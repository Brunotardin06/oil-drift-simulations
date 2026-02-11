from __future__ import annotations

import contextlib
import queue
import threading
import time
import traceback
from pathlib import Path

import flet as ft

from src.adapters.flet_ui import (
    ArtifactsViewBindings,
    ExecutionViewBindings,
    QueueWriter,
    ResultsViewBindings,
    SetupViewBindings,
    build_artifact_list,
    build_artifacts_view,
    build_execution_view,
    build_frame_list,
    extract_observed_bounds,
    build_results_view,
    build_run_id,
    build_setup_view,
    build_sidebar,
    build_value_range,
    extract_metrics,
    list_environments,
    open_path,
    parse_float,
    stage_observed_zip,
    validate_observed_zip,
)
from src.application.dto import ValidationRunRequest, ValidationRunResult
from src.application.simulation_controller import SimulationController

EMPTY_IMAGE_SRC = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="
OBSERVED_BOUNDS_PADDING_DEG = 1.0
MIN_OBSERVED_BOUNDS_SPAN_DEG = 2.0


def _enum_value(enum_name: str, member_name: str, fallback):
    enum_obj = getattr(ft, enum_name, None)
    if enum_obj is None:
        return fallback
    return getattr(enum_obj, member_name, fallback)


def _expand_bounds(
    bounds: tuple[float, float, float, float],
    padding_deg: float = OBSERVED_BOUNDS_PADDING_DEG,
    min_span_deg: float = MIN_OBSERVED_BOUNDS_SPAN_DEG,
) -> tuple[float, float, float, float]:
    min_lon, max_lon, min_lat, max_lat = bounds
    min_lon_p = min_lon - padding_deg
    max_lon_p = max_lon + padding_deg
    min_lat_p = min_lat - padding_deg
    max_lat_p = max_lat + padding_deg

    lon_span = max_lon_p - min_lon_p
    lat_span = max_lat_p - min_lat_p
    if lon_span < min_span_deg:
        lon_center = 0.5 * (min_lon_p + max_lon_p)
        half = 0.5 * min_span_deg
        min_lon_p = lon_center - half
        max_lon_p = lon_center + half
    if lat_span < min_span_deg:
        lat_center = 0.5 * (min_lat_p + max_lat_p)
        half = 0.5 * min_span_deg
        min_lat_p = lat_center - half
        max_lat_p = lat_center + half

    return (min_lon_p, max_lon_p, min_lat_p, max_lat_p)


def main(page: ft.Page) -> None:
    page.title = "Oil Spill Drift"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.min_width = 1200
    page.window.min_height = 760
    page.padding = 0
    page.bgcolor = "#D8DEE6"
    page.scroll = _enum_value("ScrollMode", "AUTO", "auto")

    project_root = Path(__file__).resolve().parents[2]
    environment_names = list_environments(project_root)

    state = {
        "selected_zip": None,
        "staged_zip": None,
        "run_id": None,
        "running": False,
        "downloading": False,
        "cancel_event": threading.Event(),
        "out_dir": None,
        "sim_path": None,
        "frames": [],
        "start_time": None,
        "next_progress_log_pct": 0,
    }
    event_queue: queue.Queue = queue.Queue()

    def show_message(message: str, *, error: bool = False) -> None:
        page.snack_bar = ft.SnackBar(
            content=ft.Text(message),
            bgcolor="#B91C1C" if error else "#1E3A8A",
        )
        page.snack_bar.open = True
        page.update()

    selected_zip_text = ft.Text("Nenhum arquivo selecionado", color="#4B6385")
    environment_download_status_text = ft.Text("Dados não baixados nesta sessão.", color="#4B6385")

    status_title = ft.Text("Idle", size=28, weight=ft.FontWeight.W_700, color="#0F172A")
    status_subtitle = ft.Text("Aguardando execução", size=16, color="#4B6385")
    progress_text = ft.Text("0%", color="#4B6385")
    progress_bar = ft.ProgressBar(value=0, bgcolor="#C9CED6", color="#030523", height=10)
    log_view = ft.ListView(expand=True, spacing=4, auto_scroll=True, height=260)

    result_score_text = ft.Text("N/A", size=34, weight=ft.FontWeight.W_700, color="#0F172A")
    result_runtime_text = ft.Text("N/A", size=26, weight=ft.FontWeight.W_600, color="#0F172A")
    best_wdf_text = ft.Text("N/A", size=24, weight=ft.FontWeight.W_600)
    best_cdf_text = ft.Text("N/A", size=24, weight=ft.FontWeight.W_600)
    best_hd_text = ft.Text("N/A", size=24, weight=ft.FontWeight.W_600)

    frame_image = ft.Image(
        src=EMPTY_IMAGE_SRC,
        fit=_enum_value("ImageFit", "CONTAIN", "contain"),
        width=980,
        height=420,
        border_radius=12,
        visible=False,
    )
    frame_label = ft.Text("Sem imagens de comparação ainda.", color="#4B6385")

    def on_frame_slider_change(e: ft.ControlEvent) -> None:
        if not state["frames"]:
            return
        idx = int(round(e.control.value))
        idx = max(0, min(idx, len(state["frames"]) - 1))
        frame_path = state["frames"][idx]
        frame_image.src = str(frame_path)
        frame_label.value = f"Step {idx + 1}/{len(state['frames'])}: {frame_path.name}"
        page.update()

    frame_slider = ft.Slider(
        min=0,
        max=0,
        divisions=1,
        disabled=True,
        on_change=on_frame_slider_change,
    )

    output_path_text = ft.Text("N/A", color="#4B6385")
    artifacts_column = ft.Column(
        spacing=8,
        scroll=_enum_value("ScrollMode", "AUTO", "auto"),
        height=340,
    )

    environment_dropdown = ft.Dropdown(
        label="Environment",
        value=environment_names[0],
        options=[ft.dropdown.Option(name) for name in environment_names],
        width=320,
    )
    start_index_field = ft.TextField(label="Start index", value="0", width=180)
    copernicus_username_field = ft.TextField(
        label="Copernicus username",
        width=320,
    )
    copernicus_password_field = ft.TextField(
        label="Copernicus password",
        password=True,
        can_reveal_password=True,
        width=320,
    )

    wdf_min_field = ft.TextField(label="WDF min", value="0.015", width=140)
    wdf_max_field = ft.TextField(label="WDF max", value="0.040", width=140)
    wdf_step_field = ft.TextField(label="WDF step", value="0.0025", width=140)

    cdf_min_field = ft.TextField(label="CDF min", value="0.5", width=140)
    cdf_max_field = ft.TextField(label="CDF max", value="1.5", width=140)
    cdf_step_field = ft.TextField(label="CDF step", value="0.1", width=140)

    hd_min_field = ft.TextField(label="HD min", value="0", width=140)
    hd_max_field = ft.TextField(label="HD max", value="10000", width=140)
    hd_step_field = ft.TextField(label="HD step", value="5000", width=140)

    stokes_switch = ft.Switch(label="Stokes Drift", value=False)
    run_opt_switch = ft.Switch(label="Optimize WDF/CDF/HD", value=True)

    def append_log(message: str) -> None:
        log_view.controls.append(ft.Text(message, size=13, color="#0F172A"))
        if len(log_view.controls) > 1000:
            log_view.controls = log_view.controls[-1000:]

    def reset_execution_panel() -> None:
        status_title.value = "Running"
        status_subtitle.value = "Executando validação..."
        progress_bar.value = 0
        progress_text.value = "0%"
        log_view.controls.clear()
        state["next_progress_log_pct"] = 0

    def refresh_artifacts() -> None:
        artifacts_column.controls.clear()
        out_dir = state["out_dir"]
        if out_dir is None:
            page.update()
            return
        for file_path in build_artifact_list(out_dir):
            artifacts_column.controls.append(
                ft.Container(
                    bgcolor="#EEF2F7",
                    border=ft.border.all(1, "#CBD5E1"),
                    border_radius=10,
                    padding=12,
                    content=ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        controls=[
                            ft.Column(
                                spacing=2,
                                controls=[
                                    ft.Text(file_path.name, weight=ft.FontWeight.W_600),
                                    ft.Text(
                                        f"{file_path.stat().st_size / 1024:.1f} KB",
                                        size=12,
                                        color="#4B6385",
                                    ),
                                ],
                            ),
                            ft.OutlinedButton(
                                "Open",
                                on_click=lambda e, p=file_path: open_path(p),
                            ),
                        ],
                    ),
                )
            )

    def refresh_results() -> None:
        out_dir = state["out_dir"]
        sim_path = state["sim_path"]
        if out_dir is None:
            return

        metrics = extract_metrics(out_dir)
        if metrics["best_skillscore"] is None:
            result_score_text.value = "N/A"
        else:
            result_score_text.value = f"{metrics['best_skillscore']:.3f}"
        best_wdf_text.value = (
            f"{metrics['best_wdf']:.4f}" if metrics["best_wdf"] is not None else "N/A"
        )
        best_cdf_text.value = (
            f"{metrics['best_cdf']:.2f}" if metrics["best_cdf"] is not None else "N/A"
        )
        best_hd_text.value = (
            f"{metrics['best_hd']:.1f}" if metrics["best_hd"] is not None else "N/A"
        )

        state["frames"] = build_frame_list(out_dir, sim_path)
        if state["frames"]:
            frame_slider.disabled = False
            frame_slider.min = 0
            frame_slider.max = len(state["frames"]) - 1
            frame_slider.divisions = max(1, len(state["frames"]) - 1)
            frame_slider.value = 0
            frame_image.src = str(state["frames"][0])
            frame_image.visible = True
            frame_label.value = f"Step 1/{len(state['frames'])}: {state['frames'][0].name}"
        else:
            frame_slider.disabled = True
            frame_slider.min = 0
            frame_slider.max = 0
            frame_slider.divisions = 1
            frame_slider.value = 0
            frame_image.src = EMPTY_IMAGE_SRC
            frame_image.visible = False
            frame_label.value = "Sem imagens de comparação ainda."

    def recover_output_from_run_id(run_id: str | None) -> tuple[Path | None, Path | None]:
        if not run_id:
            return None, None
        out_dir = project_root / "data" / "2-simulated" / run_id
        if not out_dir.exists():
            return None, None
        nc_files = sorted(out_dir.glob("*.nc"), key=lambda p: p.stat().st_mtime, reverse=True)
        sim_path = nc_files[0] if nc_files else None
        return out_dir, sim_path

    def handle_queue_event(payload: dict) -> None:
        event_type = payload.get("type")
        if event_type == "log":
            append_log(payload.get("message", ""))
        elif event_type == "env_download_log":
            append_log(payload.get("message", ""))
        elif event_type == "env_download_done":
            state["downloading"] = False
            environment = payload.get("environment", "")
            environment_download_status_text.value = (
                f"{environment}: download concluído."
            )
            append_log(f"Environment {environment}: datasets ready.")
            show_message(f"Download do ambiente {environment} concluído.")
        elif event_type == "env_download_error":
            state["downloading"] = False
            environment = payload.get("environment", "")
            environment_download_status_text.value = f"{environment}: falha no download."
            append_log(payload.get("message", "Download failed."))
            show_message(
                f"Falha no download do ambiente {environment}. Veja os logs.",
                error=True,
            )
        elif event_type == "progress":
            done = int(payload.get("done", 0))
            total = max(1, int(payload.get("total", 1)))
            pct = int(round((done / total) * 100))
            progress_bar.value = done / total
            progress_text.value = f"{pct}%"
            status_subtitle.value = f"Otimização em andamento ({done}/{total})"
            next_pct = int(state.get("next_progress_log_pct", 0))
            if done == total or pct >= next_pct:
                append_log(f"Optimization progress: {done}/{total} ({pct}%)")
                if done == total:
                    state["next_progress_log_pct"] = 101
                else:
                    while next_pct <= pct:
                        next_pct += 5
                    state["next_progress_log_pct"] = next_pct
        elif event_type == "done":
            result: ValidationRunResult = payload["result"]
            state["running"] = False
            state["out_dir"] = result.out_dir
            state["sim_path"] = result.sim_path
            duration = time.monotonic() - (state["start_time"] or time.monotonic())
            result_runtime_text.value = f"{duration:.1f} s"
            status_title.value = "Completed"
            status_subtitle.value = "Execução finalizada com sucesso"
            progress_bar.value = 1.0
            progress_text.value = "100%"
            output_path_text.value = str(result.out_dir)
            refresh_results()
            refresh_artifacts()
            show_message("Execução concluída.")
        elif event_type == "cancelled":
            state["running"] = False
            recovered_out_dir, recovered_sim_path = recover_output_from_run_id(state.get("run_id"))
            if recovered_out_dir is not None:
                state["out_dir"] = recovered_out_dir
                state["sim_path"] = recovered_sim_path
                output_path_text.value = str(recovered_out_dir)
                status_title.value = "Completed"
                status_subtitle.value = "Execução finalizada com artefatos recuperados"
                append_log(
                    f"Recovered outputs from {recovered_out_dir} after late cancellation signal."
                )
                refresh_results()
                refresh_artifacts()
                show_message("Execução finalizada e resultados recuperados.")
            else:
                status_title.value = "Cancelled"
                status_subtitle.value = "Execução cancelada pelo usuário"
                show_message("Execução cancelada.")
        elif event_type == "error":
            state["running"] = False
            status_title.value = "Failed"
            status_subtitle.value = "Falha durante a execução"
            append_log(payload.get("message", "Unknown error"))
            show_message("Falha na execução. Veja os logs.", error=True)
        page.update()

    def event_consumer_loop() -> None:
        while True:
            payload = event_queue.get()
            handle_queue_event(payload)

    consumer_thread = threading.Thread(target=event_consumer_loop, daemon=True)
    consumer_thread.start()

    def worker_download_environment(
        environment: str,
        copernicus_username: str,
        copernicus_password: str,
        observed_bounds: tuple[float, float, float, float] | None,
        force_download: bool,
    ) -> None:
        controller = SimulationController()

        def on_log(message: str) -> None:
            event_queue.put({"type": "env_download_log", "message": message})

        try:
            controller.download_environment_data(
                environment=environment,
                config_name="main",
                force=force_download,
                log_callback=on_log,
                copernicus_username=copernicus_username,
                copernicus_password=copernicus_password,
                min_long=observed_bounds[0] if observed_bounds else None,
                max_long=observed_bounds[1] if observed_bounds else None,
                min_lat=observed_bounds[2] if observed_bounds else None,
                max_lat=observed_bounds[3] if observed_bounds else None,
            )
            event_queue.put(
                {
                    "type": "env_download_done",
                    "environment": environment,
                }
            )
        except Exception:
            event_queue.put(
                {
                    "type": "env_download_error",
                    "environment": environment,
                    "message": traceback.format_exc(),
                }
            )

    def build_request(
        staged_zip: Path,
        run_id: str,
        observed_bounds: tuple[float, float, float, float],
    ) -> ValidationRunRequest:
        optimize_enabled = bool(run_opt_switch.value)
        if optimize_enabled:
            wdf_min = parse_float(wdf_min_field.value, "WDF min")
            wdf_max = parse_float(wdf_max_field.value, "WDF max")
            wdf_step = parse_float(wdf_step_field.value, "WDF step")

            cdf_min = parse_float(cdf_min_field.value, "CDF min")
            cdf_max = parse_float(cdf_max_field.value, "CDF max")
            cdf_step = parse_float(cdf_step_field.value, "CDF step")

            hd_min = parse_float(hd_min_field.value, "HD min")
            hd_max = parse_float(hd_max_field.value, "HD max")
            hd_step = parse_float(hd_step_field.value, "HD step")
            hd_values = build_value_range(hd_min, hd_max, hd_step)
        else:
            # Ranges are not used when optimization is disabled.
            wdf_min = 0.0
            wdf_max = 0.05
            wdf_step = 0.0025
            cdf_min = 0.5
            cdf_max = 1.0
            cdf_step = 0.1
            hd_values = [0.0]

        start_index = int(start_index_field.value or 0)

        return ValidationRunRequest(
            config_name="main",
            environment=environment_dropdown.value or "2019",
            shp_zip=str(staged_zip),
            min_long=observed_bounds[0],
            max_long=observed_bounds[1],
            min_lat=observed_bounds[2],
            max_lat=observed_bounds[3],
            start_index=start_index,
            optimize_wdf_stokes_cdf=optimize_enabled,
            optimize_wdf_mode="fast",
            fast_particles_per_wdf=1,
            wdf_min=wdf_min,
            wdf_max=wdf_max,
            wdf_step=wdf_step,
            cdf_min=cdf_min,
            cdf_max=cdf_max,
            cdf_step=cdf_step,
            diffusivity_values=",".join(f"{value:.6g}" for value in hd_values),
            stokes_drift="true" if stokes_switch.value else "false",
            skip_animation=False,
            skip_simulation=False,
            skip_plots=False,
            run_name=run_id,
        )

    def worker_run_validation(request: ValidationRunRequest) -> None:
        controller = SimulationController()
        writer = QueueWriter(event_queue)

        def on_progress(done: int, total: int) -> None:
            event_queue.put({"type": "progress", "done": done, "total": total})

        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                result = controller.run_validation(
                    request,
                    progress_callback=on_progress,
                    should_cancel=state["cancel_event"].is_set,
                    show_plots=False,
                )
            writer.flush()
            # If a valid result exists, treat execution as completed even if a late
            # cancel flag was set near the end of the pipeline.
            if result:
                if state["cancel_event"].is_set():
                    event_queue.put(
                        {
                            "type": "log",
                            "message": "Cancel requested late, but outputs are complete. Finalizing as done.",
                        }
                    )
                event_queue.put({"type": "done", "result": result})
                return
            if state["cancel_event"].is_set():
                event_queue.put({"type": "cancelled"})
                return
            if not result:
                event_queue.put({"type": "error", "message": "No result returned from run_validation."})
                return
        except RuntimeError as exc:
            writer.flush()
            if "cancel" in str(exc).lower():
                event_queue.put({"type": "cancelled"})
            else:
                event_queue.put({"type": "error", "message": traceback.format_exc()})
        except Exception:
            writer.flush()
            event_queue.put({"type": "error", "message": traceback.format_exc()})

    def start_execution(e: ft.ControlEvent) -> None:
        if state["running"]:
            status_title.value = "Blocked"
            status_subtitle.value = "Já existe execução em andamento."
            set_screen("Execution")
            append_log("Start blocked: there is already an execution running.")
            page.update()
            show_message("Já existe uma execução em andamento.", error=True)
            return
        if state["downloading"]:
            status_title.value = "Blocked"
            status_subtitle.value = "Download de ambiente em andamento."
            set_screen("Execution")
            append_log("Start blocked: wait for environment download to finish.")
            page.update()
            show_message(
                "Aguarde o término do download do ambiente para iniciar a execução.",
                error=True,
            )
            return

        set_screen("Execution")
        reset_execution_panel()
        status_title.value = "Preparing"
        status_subtitle.value = "Validando entradas..."
        append_log("Start requested by user.")
        page.update()

        try:
            if not state["selected_zip"]:
                raise ValueError("Selecione o ZIP com o spill observado.")
            source_zip = Path(state["selected_zip"])
            has_prj = validate_observed_zip(source_zip)
            observed_bounds_raw = extract_observed_bounds(source_zip)
            observed_bounds = _expand_bounds(observed_bounds_raw)
            if not has_prj:
                append_log("Warning: observed ZIP has no .prj file.")
            run_id = build_run_id()
            staged_zip = stage_observed_zip(project_root, run_id, source_zip)
            state["run_id"] = run_id
            state["staged_zip"] = staged_zip
            append_log(
                "Observed bounds (raw) "
                f"lon=[{observed_bounds_raw[0]:.5f},{observed_bounds_raw[1]:.5f}] "
                f"lat=[{observed_bounds_raw[2]:.5f},{observed_bounds_raw[3]:.5f}]"
            )
            append_log(
                f"Observed bounds (padded {OBSERVED_BOUNDS_PADDING_DEG:.2f} deg, "
                f"min-span {MIN_OBSERVED_BOUNDS_SPAN_DEG:.2f} deg) "
                f"lon=[{observed_bounds[0]:.5f},{observed_bounds[1]:.5f}] "
                f"lat=[{observed_bounds[2]:.5f},{observed_bounds[3]:.5f}]"
            )
            request = build_request(
                staged_zip=staged_zip,
                run_id=run_id,
                observed_bounds=observed_bounds,
            )
        except Exception as exc:
            status_title.value = "Failed"
            status_subtitle.value = "Falha na validação de entrada."
            append_log(traceback.format_exc())
            page.update()
            show_message(str(exc), error=True)
            return

        state["running"] = True
        state["cancel_event"].clear()
        state["start_time"] = time.monotonic()
        reset_execution_panel()
        append_log(f"Run ID: {state['run_id']}")
        append_log(f"Staged observed ZIP: {state['staged_zip']}")
        page.update()

        thread = threading.Thread(target=worker_run_validation, args=(request,), daemon=True)
        thread.start()

    def start_environment_download(e: ft.ControlEvent) -> None:
        if state["running"]:
            show_message("Não é possível baixar dados durante uma execução.", error=True)
            return
        if state["downloading"]:
            show_message("Já existe um download em andamento.", error=True)
            return

        environment = environment_dropdown.value or "2019"
        copernicus_username = (copernicus_username_field.value or "").strip()
        copernicus_password = copernicus_password_field.value or ""
        if not copernicus_username or not copernicus_password:
            show_message(
                "Informe usuário e senha do Copernicus para baixar os dados.",
                error=True,
            )
            return
        observed_bounds = None
        force_download = False
        if state["selected_zip"]:
            try:
                source_zip = Path(state["selected_zip"])
                validate_observed_zip(source_zip)
                observed_bounds = _expand_bounds(extract_observed_bounds(source_zip))
                force_download = True
            except Exception as exc:
                show_message(f"Falha ao ler bounds do ZIP observado: {exc}", error=True)
                return
        state["downloading"] = True
        environment_download_status_text.value = f"{environment}: baixando dados..."
        status_title.value = "Downloading"
        status_subtitle.value = f"Baixando dados do ambiente {environment}..."
        if observed_bounds:
            append_log(
                f"Environment {environment}: starting Copernicus download with observed bounds "
                f"lon=[{observed_bounds[0]:.5f},{observed_bounds[1]:.5f}] "
                f"lat=[{observed_bounds[2]:.5f},{observed_bounds[3]:.5f}] (force overwrite)."
            )
        else:
            append_log(f"Environment {environment}: starting Copernicus download.")
        set_screen("Execution")
        page.update()

        thread = threading.Thread(
            target=worker_download_environment,
            args=(
                environment,
                copernicus_username,
                copernicus_password,
                observed_bounds,
                force_download,
            ),
            daemon=True,
        )
        thread.start()

    def cancel_execution(e: ft.ControlEvent) -> None:
        if not state["running"]:
            return
        state["cancel_event"].set()
        status_title.value = "Cancelling"
        status_subtitle.value = "Aguardando ponto seguro para parar..."
        append_log("Cancel requested by user.")
        page.update()

    async def choose_zip_async() -> None:
        files = await file_picker.pick_files(
            allow_multiple=False,
            file_type=getattr(ft, "FilePickerFileType", None).CUSTOM
            if getattr(ft, "FilePickerFileType", None) is not None
            else "custom",
            allowed_extensions=["zip"],
        )
        if not files:
            return
        selected_path = files[0].path
        if not selected_path:
            return
        state["selected_zip"] = selected_path
        selected_zip_text.value = selected_path
        page.update()

    def choose_zip(e: ft.ControlEvent) -> None:
        page.run_task(choose_zip_async)

    file_picker = ft.FilePicker()
    page.services.append(file_picker)

    views = {
        "Validation Setup": build_setup_view(
            SetupViewBindings(
                choose_zip=choose_zip,
                selected_zip_text=selected_zip_text,
                environment_dropdown=environment_dropdown,
                copernicus_username_field=copernicus_username_field,
                copernicus_password_field=copernicus_password_field,
                environment_download_status_text=environment_download_status_text,
                download_environment_data=start_environment_download,
                start_index_field=start_index_field,
                run_opt_switch=run_opt_switch,
                wdf_min_field=wdf_min_field,
                wdf_max_field=wdf_max_field,
                wdf_step_field=wdf_step_field,
                cdf_min_field=cdf_min_field,
                cdf_max_field=cdf_max_field,
                cdf_step_field=cdf_step_field,
                hd_min_field=hd_min_field,
                hd_max_field=hd_max_field,
                hd_step_field=hd_step_field,
                stokes_switch=stokes_switch,
                start_execution=start_execution,
            )
        ),
        "Execution": build_execution_view(
            ExecutionViewBindings(
                status_title=status_title,
                status_subtitle=status_subtitle,
                progress_text=progress_text,
                progress_bar=progress_bar,
                log_view=log_view,
                cancel_execution=cancel_execution,
            )
        ),
        "Results": build_results_view(
            ResultsViewBindings(
                result_score_text=result_score_text,
                result_runtime_text=result_runtime_text,
                best_wdf_text=best_wdf_text,
                best_cdf_text=best_cdf_text,
                best_hd_text=best_hd_text,
                frame_image=frame_image,
                frame_label=frame_label,
                frame_slider=frame_slider,
            )
        ),
        "Artifacts": build_artifacts_view(
            ArtifactsViewBindings(
                output_path_text=output_path_text,
                artifacts_column=artifacts_column,
                open_output_folder=lambda _: open_path(state["out_dir"]) if state["out_dir"] else None,
            )
        ),
    }

    content_host = ft.Container(expand=True)
    nav_buttons: dict[str, ft.Control] = {}

    def set_screen(screen_name: str) -> None:
        content_host.content = views[screen_name]
        for label, button in nav_buttons.items():
            button.style = ft.ButtonStyle(
                bgcolor="#364B6A" if label == screen_name else "transparent",
                color="#FFFFFF" if label == screen_name else "#D2D9E6",
                shape=ft.RoundedRectangleBorder(radius=10),
            )
        page.update()

    sidebar, created_nav_buttons = build_sidebar(on_navigate=set_screen)
    nav_buttons.update(created_nav_buttons)

    page.add(
        ft.Row(
            expand=True,
            spacing=0,
            controls=[
                sidebar,
                ft.Container(expand=True, content=content_host),
            ],
        )
    )

    set_screen("Validation Setup")


if __name__ == "__main__":
    ft.app(target=main)
