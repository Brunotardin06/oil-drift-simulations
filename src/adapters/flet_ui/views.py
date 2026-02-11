from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import flet as ft


def _icon_value(name: str, fallback: str):
    icons_obj = getattr(ft, "Icons", None)
    if icons_obj is None:
        icons_obj = getattr(ft, "icons", None)
    if icons_obj is not None:
        return getattr(icons_obj, name, fallback)
    return fallback


def _enum_value(enum_name: str, member_name: str, fallback):
    enum_obj = getattr(ft, enum_name, None)
    if enum_obj is None:
        return fallback
    return getattr(enum_obj, member_name, fallback)


@dataclass(frozen=True)
class SetupViewBindings:
    choose_zip: Callable[[ft.ControlEvent], None]
    selected_zip_text: ft.Text
    environment_dropdown: ft.Dropdown
    copernicus_username_field: ft.TextField
    copernicus_password_field: ft.TextField
    environment_download_status_text: ft.Text
    download_environment_data: Callable[[ft.ControlEvent], None]
    start_index_field: ft.TextField
    run_opt_switch: ft.Switch
    wdf_min_field: ft.TextField
    wdf_max_field: ft.TextField
    wdf_step_field: ft.TextField
    cdf_min_field: ft.TextField
    cdf_max_field: ft.TextField
    cdf_step_field: ft.TextField
    hd_min_field: ft.TextField
    hd_max_field: ft.TextField
    hd_step_field: ft.TextField
    stokes_switch: ft.Switch
    start_execution: Callable[[ft.ControlEvent], None]


@dataclass(frozen=True)
class ExecutionViewBindings:
    status_title: ft.Text
    status_subtitle: ft.Text
    progress_text: ft.Text
    progress_bar: ft.ProgressBar
    log_view: ft.ListView
    cancel_execution: Callable[[ft.ControlEvent], None]


@dataclass(frozen=True)
class ResultsViewBindings:
    result_score_text: ft.Text
    result_runtime_text: ft.Text
    best_wdf_text: ft.Text
    best_cdf_text: ft.Text
    best_hd_text: ft.Text
    frame_image: ft.Image
    frame_label: ft.Text
    frame_slider: ft.Slider


@dataclass(frozen=True)
class ArtifactsViewBindings:
    output_path_text: ft.Text
    artifacts_column: ft.Column
    open_output_folder: Callable[[ft.ControlEvent], None]


def build_setup_view(bindings: SetupViewBindings) -> ft.Container:
    return ft.Container(
        padding=28,
        content=ft.Column(
            spacing=18,
            controls=[
                ft.Text("Validation Setup", size=42, weight=ft.FontWeight.W_700, color="#0F172A"),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=14,
                        controls=[
                            ft.Text("Observed Spill Data", size=28, weight=ft.FontWeight.W_600),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    ft.ElevatedButton(
                                        "Choose ZIP",
                                        icon=_icon_value("UPLOAD_FILE", "upload_file"),
                                        on_click=bindings.choose_zip,
                                    ),
                                    bindings.selected_zip_text,
                                ],
                            ),
                        ],
                    ),
                ),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=14,
                        controls=[
                            ft.Text("Environment Configuration", size=28, weight=ft.FontWeight.W_600),
                            ft.Row(spacing=16, controls=[bindings.environment_dropdown, bindings.start_index_field]),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    bindings.copernicus_username_field,
                                    bindings.copernicus_password_field,
                                ],
                            ),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    ft.OutlinedButton(
                                        "Download Environment Data",
                                        icon=_icon_value("DOWNLOAD", "download"),
                                        on_click=bindings.download_environment_data,
                                    ),
                                    bindings.environment_download_status_text,
                                ],
                            ),
                        ],
                    ),
                ),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=14,
                        controls=[
                            ft.Text("Optimization Parameters", size=28, weight=ft.FontWeight.W_600),
                            bindings.run_opt_switch,
                            ft.Text("Wind Drift Factor (WDF)", size=20, weight=ft.FontWeight.W_600),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    bindings.wdf_min_field,
                                    bindings.wdf_max_field,
                                    bindings.wdf_step_field,
                                ],
                            ),
                            ft.Text("Current Drift Factor (CDF)", size=20, weight=ft.FontWeight.W_600),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    bindings.cdf_min_field,
                                    bindings.cdf_max_field,
                                    bindings.cdf_step_field,
                                ],
                            ),
                            ft.Text("Horizontal Diffusion (HD)", size=20, weight=ft.FontWeight.W_600),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    bindings.hd_min_field,
                                    bindings.hd_max_field,
                                    bindings.hd_step_field,
                                ],
                            ),
                            bindings.stokes_switch,
                        ],
                    ),
                ),
                ft.Row(
                    alignment=ft.MainAxisAlignment.END,
                    controls=[
                        ft.FilledButton(
                            "Start Execution",
                            icon=_icon_value("PLAY_ARROW", "play_arrow"),
                            on_click=bindings.start_execution,
                        )
                    ],
                ),
            ],
        ),
    )


def build_execution_view(bindings: ExecutionViewBindings) -> ft.Container:
    return ft.Container(
        padding=28,
        content=ft.Column(
            spacing=18,
            controls=[
                ft.Text("Execution Progress", size=42, weight=ft.FontWeight.W_700, color="#0F172A"),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=16,
                        controls=[
                            ft.Row(
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                controls=[
                                    ft.Column(
                                        spacing=4,
                                        controls=[bindings.status_title, bindings.status_subtitle],
                                    ),
                                    ft.OutlinedButton(
                                        "Cancel",
                                        icon=_icon_value("CANCEL", "cancel"),
                                        style=ft.ButtonStyle(color="#B91C1C"),
                                        on_click=bindings.cancel_execution,
                                    ),
                                ],
                            ),
                            bindings.progress_text,
                            bindings.progress_bar,
                        ],
                    ),
                ),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    expand=True,
                    content=ft.Column(
                        spacing=10,
                        expand=True,
                        controls=[
                            ft.Text("Execution Log", size=26, weight=ft.FontWeight.W_600),
                            bindings.log_view,
                        ],
                    ),
                ),
            ],
        ),
    )


def build_results_view(bindings: ResultsViewBindings) -> ft.Container:
    return ft.Container(
        padding=28,
        content=ft.Column(
            spacing=18,
            controls=[
                ft.Text("Results Overview", size=42, weight=ft.FontWeight.W_700, color="#0F172A"),
                ft.Row(
                    spacing=16,
                    controls=[
                        ft.Container(
                            width=270,
                            bgcolor="#EDF1F6",
                            border=ft.border.all(1, "#C8D1DB"),
                            border_radius=12,
                            padding=18,
                            content=ft.Column(
                                spacing=6,
                                controls=[
                                    ft.Text("Skill Score", size=18, color="#4B6385"),
                                    bindings.result_score_text,
                                ],
                            ),
                        ),
                        ft.Container(
                            width=270,
                            bgcolor="#EDF1F6",
                            border=ft.border.all(1, "#C8D1DB"),
                            border_radius=12,
                            padding=18,
                            content=ft.Column(
                                spacing=6,
                                controls=[
                                    ft.Text("Duration", size=18, color="#4B6385"),
                                    bindings.result_runtime_text,
                                ],
                            ),
                        ),
                    ],
                ),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=10,
                        controls=[
                            ft.Text("Best Parameters", size=26, weight=ft.FontWeight.W_600),
                            ft.Row(
                                spacing=24,
                                controls=[
                                    ft.Column(controls=[ft.Text("WDF", color="#4B6385"), bindings.best_wdf_text]),
                                    ft.Column(controls=[ft.Text("CDF", color="#4B6385"), bindings.best_cdf_text]),
                                    ft.Column(controls=[ft.Text("HD", color="#4B6385"), bindings.best_hd_text]),
                                ],
                            ),
                        ],
                    ),
                ),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=12,
                        controls=[
                            ft.Text("Trajectory Comparison", size=26, weight=ft.FontWeight.W_600),
                            bindings.frame_image,
                            bindings.frame_label,
                            bindings.frame_slider,
                        ],
                    ),
                ),
            ],
        ),
    )


def build_artifacts_view(bindings: ArtifactsViewBindings) -> ft.Container:
    return ft.Container(
        padding=28,
        content=ft.Column(
            spacing=18,
            controls=[
                ft.Text("Artifacts & Export", size=42, weight=ft.FontWeight.W_700, color="#0F172A"),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        controls=[
                            ft.Column(
                                spacing=4,
                                controls=[
                                    ft.Text("Output Location", size=24, weight=ft.FontWeight.W_600),
                                    bindings.output_path_text,
                                ],
                            ),
                            ft.OutlinedButton(
                                "Open Folder",
                                icon=_icon_value("FOLDER_OPEN", "folder_open"),
                                on_click=bindings.open_output_folder,
                            ),
                        ],
                    ),
                ),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=10,
                        controls=[
                            ft.Text("Generated Artifacts", size=26, weight=ft.FontWeight.W_600),
                            bindings.artifacts_column,
                        ],
                    ),
                ),
            ],
        ),
    )


def build_sidebar(on_navigate: Callable[[str], None]) -> tuple[ft.Container, dict[str, ft.TextButton]]:
    nav_buttons: dict[str, ft.TextButton] = {}
    nav_column = ft.Column(spacing=10)

    for label, icon in [
        ("Validation Setup", _icon_value("DESCRIPTION_OUTLINED", "description_outlined")),
        ("Execution", _icon_value("PLAY_ARROW_OUTLINED", "play_arrow_outlined")),
        ("Results", _icon_value("BAR_CHART_OUTLINED", "bar_chart_outlined")),
        ("Artifacts", _icon_value("FOLDER_OPEN_OUTLINED", "folder_open_outlined")),
    ]:
        btn = ft.TextButton(
            content=label,
            icon=icon,
            style=ft.ButtonStyle(
                color="#D2D9E6",
                alignment=ft.Alignment(-1, 0),
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=ft.padding.symmetric(horizontal=12, vertical=14),
            ),
            on_click=lambda e, name=label: on_navigate(name),
        )
        nav_buttons[label] = btn
        nav_column.controls.append(btn)

    sidebar = ft.Container(
        width=300,
        bgcolor="#1E2C45",
        padding=20,
        content=ft.Column(
            expand=True,
            spacing=20,
            controls=[
                ft.Text("Oil Spill Drift", size=40, weight=ft.FontWeight.W_700, color="#FFFFFF"),
                ft.Text("Digital Twin Validation", size=18, color="#9DB0CE"),
                nav_column,
                ft.Container(expand=True),
                ft.Text("OpenDrift/OpenOil v2.5", color="#9DB0CE"),
            ],
        ),
    )
    return sidebar, nav_buttons
