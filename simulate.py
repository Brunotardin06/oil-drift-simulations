import click
from hydra import compose, initialize
from pathlib import Path
from datetime import datetime
import geopandas as gpd
from src.simulation import simulate_drift
import pandas as pd
import xarray as xr
import numpy as np
import json

from shapely.geometry import Point
import matplotlib.pyplot as plt
from compare_spills_gif import generate_comparison_gif
from opendrift.models.openoil import OpenOil
from src.optimization import (
    build_observed_trajectory,
    fast_grid_search_wdf_stokes_current_drift,
)

def _load_oil_types(oil_types, oil_types_file):
    def normalize_name(name):
        return " ".join(name.strip().split()).lower()

    selected = []
    if oil_types_file:
        file_path = Path(oil_types_file)
        if not file_path.exists():
            raise ValueError(f"oil types file not found: {file_path}")
        selected.extend(
            [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        )
    if oil_types:
        selected.extend([item.strip() for item in oil_types.split(",") if item.strip()])

    if not selected:
        return []

    available = OpenOil(loglevel=50).oiltypes
    by_normalized = {normalize_name(name): name for name in available}
    normalized = []
    invalid = []
    for name in selected:
        mapped = by_normalized.get(normalize_name(name))
        if mapped:
            normalized.append(mapped)
            continue
        invalid.append(name)

    if invalid:
        raise ValueError(
            "Unknown oil types: "
            + ", ".join(invalid[:10])
            + (" ..." if len(invalid) > 10 else "")
        )

    return list(dict.fromkeys(normalized))


def _parse_float_list(value, default):
    if value is None:
        return list(default)
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        return list(default)
    return [float(item) for item in parts]


def _parse_bool_list(value, default):
    if value is None:
        return list(default)
    parts = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not parts:
        return list(default)
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    parsed = []
    for item in parts:
        if item not in mapping:
            raise ValueError(f"Invalid boolean value: {item}")
        parsed.append(mapping[item])
    return list(dict.fromkeys(parsed))


class _ProgressPrinter:
    def __init__(self, total, every=15, label="Progress"):
        self.total = int(total)
        self.every = int(every)
        self.label = label
        self.count = 0

    def tick(self, n=1):
        if self.total <= 0:
            return
        self.count += int(n)
        if self.count % self.every == 0 or self.count >= self.total:
            print(f"{self.label}: {self.count}/{self.total}")

def _build_datetime_column(real_manchas):
    columns = set(real_manchas.columns)
    if "DATA_HORA1" in columns and "TEMPO_ENTR" in columns:
        real_manchas["date"] = pd.to_datetime(real_manchas["DATA_HORA1"], format="%d/%m/%Y")
        real_manchas["time"] = pd.to_datetime(real_manchas["TEMPO_ENTR"], format="%H:%M")
        real_manchas["datetime"] = real_manchas.apply(
            lambda row: datetime.combine(row["date"].date(), row["time"].time()),
            axis=1,
        )
        return

    if "Data/Hora" in columns:
        real_manchas["datetime"] = pd.to_datetime(
            real_manchas["Data/Hora"],
            dayfirst=True,
            errors="raise",
        )
        return

    raise ValueError(
        "Missing datetime fields. Expected DATA_HORA1/TEMPO_ENTR or Data/Hora."
    )

@click.command()
@click.option('--config-name', default='main')
@click.option('--skip-animation', is_flag=True, default=False)
@click.option('--skip-simulation', is_flag=False, default=False)
@click.option('--skip-plots', is_flag=True, default=False)
@click.option('--evaluation', is_flag=True, default=False)
@click.option('--optimize-wdf', is_flag=True, default=False)
@click.option('--optimize-stokes', is_flag=True, default=False)
@click.option('--optimize-wdf-stokes', is_flag=True, default=False)
@click.option('--optimize-wdf-stokes-cdf', is_flag=True, default=False)
@click.option('--optimize-physics', is_flag=True, default=False)
@click.option('--optimize-wdf-mode', type=click.Choice(['robust', 'fast'], case_sensitive=False), default='fast')
@click.option('--fast-particles-per-wdf', type=int, default=1)
@click.option('--wdf-min', type=float, default=0.0)
@click.option('--wdf-max', type=float, default=0.05)
@click.option('--wdf-step', type=float, default=0.0025)
@click.option('--cdf-min', type=float, default=0.5)
@click.option('--cdf-max', type=float, default=1.0)
@click.option('--cdf-step', type=float, default=0.1)
@click.option('--diffusivity-values', type=str, default=None)
@click.option('--dispersion-values', type=str, default=None)
@click.option('--evaporation-values', type=str, default=None)
@click.option('--optimize-cleanup', is_flag=True, default=False)
@click.option('--padding-animation-frame', type=float, default=0.1)
@click.option('--wind-drift-factor', type=float, default=None)
@click.option('--current-drift-factor', type=float, default=None)
@click.option('--stokes-drift', type=click.Choice(['true', 'false'], case_sensitive=False), default=None)
@click.option('--horizontal-diffusivity', type=float, default=None)
@click.option('--processes-dispersion', type=click.Choice(['true', 'false'], case_sensitive=False), default=None)
@click.option('--processes-evaporation', type=click.Choice(['true', 'false'], case_sensitive=False), default=None)
@click.option('--oil-types', type=str, default=None)
@click.option('--oil-types-file', type=str, default=None)
@click.option('--environment', type=str, default="2019")
@click.option('--shp-zip', type=str, default=None)
@click.option('--min-long', type=float, default=None)
@click.option('--max-long', type=float, default=None)
@click.option('--min-lat', type=float, default=None)
@click.option('--max-lat', type=float, default=None)
@click.option('--start-index', type=int, default=0)
@click.option('--optimize-cdf-hd-de', is_flag=True, default=False)
def simulate_validation(
    config_name,
    skip_animation,
    skip_simulation,
    skip_plots,
    evaluation,
    optimize_wdf,
    optimize_stokes,
    optimize_wdf_stokes,
    optimize_wdf_stokes_cdf,
    optimize_physics,
    optimize_wdf_mode,
    fast_particles_per_wdf,
    wdf_min,
    wdf_max,
    wdf_step,
    cdf_min,
    cdf_max,
    cdf_step,
    diffusivity_values,
    dispersion_values,
    evaporation_values,
    optimize_cleanup,
    padding_animation_frame,
    wind_drift_factor,
    current_drift_factor,
    stokes_drift,
    horizontal_diffusivity,
    processes_dispersion,
    processes_evaporation,
    oil_types,
    oil_types_file,
    environment,
    shp_zip,
    min_long,
    max_long,
    min_lat,
    max_lat,
    start_index,
    optimize_cdf_hd_de,
):
    if evaluation:
        skip_animation = True
        skip_plots = True
    with initialize(version_base=None, config_path="conf"):
        overrides = [
            "simulation=sim4validation",
            f"environment=\"{environment}\"",
        ]
        if shp_zip:
            overrides.append(f"paths.plataformas_shp=\"{shp_zip}\"")
        if min_long is not None:
            overrides.append(f"copernicusmarine.min_long={min_long}")
        if max_long is not None:
            overrides.append(f"copernicusmarine.max_long={max_long}")
        if min_lat is not None:
            overrides.append(f"copernicusmarine.min_lat={min_lat}")
        if max_lat is not None:
            overrides.append(f"copernicusmarine.max_lat={max_lat}")
        config = compose(config_name=config_name, overrides=overrides)


        # -- Step 1 : Importar dados reais
        # Para converter um arquivo .shp em dataframe Python -> geopandas
        zip_file = Path(config.paths.plataformas_shp)
        real_manchas = gpd.read_file(zip_file).to_crs(epsg=4326)


        # -- Step 1bis : Engenharia de features: juntar DATA e HORA em um campo só
        _build_datetime_column(real_manchas)
        offset_hours = 0.0
        try:
            offset_hours = float(config.copernicusmarine.specificities.get("datetime_offset_hours", 0))
        except Exception:
            offset_hours = 0.0
        if offset_hours:
            real_manchas["datetime"] = real_manchas["datetime"] + pd.Timedelta(hours=offset_hours)
        real_manchas.sort_values('datetime', inplace=True)
        unique_times = sorted(real_manchas["datetime"].unique())
        if start_index < 0 or start_index >= len(unique_times):
            raise ValueError(f"start-index out of range (0..{len(unique_times)-1})")
        if start_index:
            start_time = unique_times[start_index]
            real_manchas = real_manchas[real_manchas["datetime"] >= start_time].copy()
        minlon, minlat, maxlon, maxlat = real_manchas.total_bounds
        pad_lon = (maxlon - minlon) * padding_animation_frame
        pad_lat = (maxlat - minlat) * padding_animation_frame
        plot_bounds = (
            minlon - pad_lon,
            maxlon + pad_lon,
            minlat - pad_lat,
            maxlat + pad_lat,
        )
        observed_trajectory = build_observed_trajectory(real_manchas)
        stokes_override = None
        if stokes_drift is not None:
            stokes_override = stokes_drift.lower() == "true"
        stokes_drift = None
        stokes_values = [True, False] if stokes_override is None else [stokes_override]
        selected_oil_types = _load_oil_types(oil_types, oil_types_file)
        selected_oil_type = selected_oil_types[0] if selected_oil_types else None
        if horizontal_diffusivity is not None:
            horizontal_diffusivity = float(horizontal_diffusivity)
        if processes_dispersion is not None:
            processes_dispersion = processes_dispersion.lower() == "true"
        if processes_evaporation is not None:
            processes_evaporation = processes_evaporation.lower() == "true"
        auto_physics = bool(optimize_physics)
        if optimize_wdf or optimize_stokes or optimize_wdf_stokes or optimize_physics or optimize_cdf_hd_de:
            raise ValueError(
                "This codebase was simplified. Use only --optimize-wdf-stokes-cdf "
                "with fast mode (CDF/HD grid + fast WDF)."
            )

        if optimize_wdf_stokes_cdf:
            if stokes_override is not None:
                print(f"Fixing stokes_drift to {stokes_override} for optimization.")
            if optimize_wdf or optimize_stokes or optimize_wdf_stokes:
                print("Ignoring --optimize-wdf/--optimize-stokes/--optimize-wdf-stokes because --optimize-wdf-stokes-cdf is set.")
            if wind_drift_factor is not None:
                print("Ignoring --wind-drift-factor because --optimize-wdf-stokes-cdf is set.")
            if current_drift_factor is not None:
                print("Ignoring --current-drift-factor because --optimize-wdf-stokes-cdf is set.")
            if wdf_step <= 0:
                raise ValueError("wdf-step must be > 0")
            if wdf_max < wdf_min:
                raise ValueError("wdf-max must be >= wdf-min")
            if cdf_step <= 0:
                raise ValueError("cdf-step must be > 0")
            if cdf_max < cdf_min:
                raise ValueError("cdf-max must be >= cdf-min")

            wdf_values = np.arange(wdf_min, wdf_max + (wdf_step / 2), wdf_step)
            cdf_values = np.arange(cdf_min, cdf_max + (cdf_step / 2), cdf_step)
            if diffusivity_values is not None:
                hdiff_values = _parse_float_list(diffusivity_values, [0.0])
            elif horizontal_diffusivity is not None:
                hdiff_values = [float(horizontal_diffusivity)]
            else:
                hdiff_values = [0.0]
            out_dir = Path(config.paths.simulation_data) / config.simulation.name
            out_dir.mkdir(parents=True, exist_ok=True)
            mode = optimize_wdf_mode.lower()
            if mode != "fast":
                raise ValueError("Only fast mode is supported in the simplified optimizer.")
            if optimize_cleanup:
                print("Note: --optimize-cleanup is ignored in fast mode.")
            total_runs = len(cdf_values) * len(hdiff_values)
            print(
                f"Will test {len(cdf_values)} cdf x {len(hdiff_values)} hd "
                f"= {total_runs} simulations (fast; all WDFs per run, stokes fixed=False)."
            )
            progress = _ProgressPrinter(total_runs, every=15, label="Progress")
            best_row, results_df = fast_grid_search_wdf_stokes_current_drift(
                real_manchas,
                config,
                observed_trajectory,
                wdf_values,
                cdf_values,
                hdiff_values,
                particles_per_wdf=fast_particles_per_wdf,
                oil_type=selected_oil_type,
                progress=progress,
            )
            results_name = "wdf_cdf_hd_optimization_fast"

            results_df.to_csv(out_dir / f"{results_name}.csv", index=False)
            if best_row is None or pd.isna(best_row["skillscore"]):
                print("Combined optimization failed: no valid skillscore computed.")
                return

            wind_drift_factor = float(best_row["wind_drift_factor"])
            stokes_drift = False
            current_drift_factor = float(best_row["current_drift_factor"])
            if "horizontal_diffusivity" in best_row and pd.notna(best_row["horizontal_diffusivity"]):
                horizontal_diffusivity = float(best_row["horizontal_diffusivity"])
            if "oil_type" in best_row:
                selected_oil_type = str(best_row["oil_type"])
            summary = {
                "wind_drift_factor": wind_drift_factor,
                "stokes_drift": stokes_drift,
                "current_drift_factor": current_drift_factor,
                "skillscore": float(best_row["skillscore"]),
                "wdf_min": float(wdf_min),
                "wdf_max": float(wdf_max),
                "wdf_step": float(wdf_step),
                "cdf_min": float(cdf_min),
                "cdf_max": float(cdf_max),
                "cdf_step": float(cdf_step),
                "hdiff_values": hdiff_values,
                "mode": mode,
            }
            if horizontal_diffusivity is not None:
                summary["horizontal_diffusivity"] = float(horizontal_diffusivity)
            if selected_oil_type:
                summary["oil_type"] = selected_oil_type
            if mode == "fast":
                summary["particles_per_wdf"] = int(fast_particles_per_wdf)
            (out_dir / f"{results_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(
                f"Best wdf/stokes/cdf: {wind_drift_factor:.4f} "
                f"stokes={stokes_drift} "
                f"cdf={current_drift_factor:.2f} "
                f"hd={horizontal_diffusivity if horizontal_diffusivity is not None else 0.0:.2f} "
                f"oil={selected_oil_type or 'default'} "
                f"(skillscore {best_row['skillscore']:.4f})"
            )
            if mode == "fast":
                auto_physics = True

        if optimize_wdf_stokes and not optimize_wdf_stokes_cdf:
            if stokes_override is not None:
                print(f"Fixing stokes_drift to {stokes_override} for optimization.")
            if optimize_wdf or optimize_stokes:
                print("Ignoring --optimize-wdf/--optimize-stokes because --optimize-wdf-stokes is set.")
            if wind_drift_factor is not None:
                print("Ignoring --wind-drift-factor because --optimize-wdf-stokes is set.")
            if wdf_step <= 0:
                raise ValueError("wdf-step must be > 0")
            if wdf_max < wdf_min:
                raise ValueError("wdf-max must be >= wdf-min")

            wdf_values = np.arange(wdf_min, wdf_max + (wdf_step / 2), wdf_step)
            out_dir = Path(config.paths.simulation_data) / config.simulation.name
            out_dir.mkdir(parents=True, exist_ok=True)
            mode = optimize_wdf_mode.lower()
            if mode == "fast":
                if optimize_cleanup:
                    print("Note: --optimize-cleanup is ignored in fast mode.")
                best_row, results_df = fast_grid_search_wdf_stokes(
                    real_manchas,
                    config,
                    observed_trajectory,
                    wdf_values,
                    stokes_values=stokes_values,
                    particles_per_wdf=fast_particles_per_wdf,
                    current_drift_factor=current_drift_factor,
                    oil_type=selected_oil_type,
                )
                results_name = "wdf_stokes_optimization_fast"
            else:
                best_row, results_df = grid_search_wdf_stokes(
                    real_manchas,
                    config,
                    observed_trajectory,
                    wdf_values,
                    stokes_values=stokes_values,
                    padding_animation_frame=padding_animation_frame,
                    out_prefix="sim_wdf_stokes_opt",
                    cleanup=optimize_cleanup,
                    current_drift_factor=current_drift_factor,
                    oil_type=selected_oil_type,
                )
                results_name = "wdf_stokes_optimization"

            results_df.to_csv(out_dir / f"{results_name}.csv", index=False)
            if best_row is None or pd.isna(best_row["skillscore"]):
                print("Combined optimization failed: no valid skillscore computed.")
                return

            wind_drift_factor = float(best_row["wind_drift_factor"])
            stokes_drift = bool(best_row["stokes_drift"])
            summary = {
                "wind_drift_factor": wind_drift_factor,
                "stokes_drift": stokes_drift,
                "skillscore": float(best_row["skillscore"]),
                "wdf_min": float(wdf_min),
                "wdf_max": float(wdf_max),
                "wdf_step": float(wdf_step),
                "mode": mode,
            }
            if current_drift_factor is not None:
                summary["current_drift_factor"] = float(current_drift_factor)
            if selected_oil_type:
                summary["oil_type"] = selected_oil_type
            if mode == "fast":
                summary["particles_per_wdf"] = int(fast_particles_per_wdf)
            (out_dir / f"{results_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(
                f"Best wdf/stokes: {wind_drift_factor:.4f} "
                f"stokes={stokes_drift} "
                f"oil={selected_oil_type or 'default'} "
                f"(skillscore {best_row['skillscore']:.4f})"
            )

        if optimize_wdf and not optimize_wdf_stokes and not optimize_wdf_stokes_cdf:
            if wind_drift_factor is not None:
                print("Ignoring --wind-drift-factor because --optimize-wdf is set.")
            if wdf_step <= 0:
                raise ValueError("wdf-step must be > 0")
            if wdf_max < wdf_min:
                raise ValueError("wdf-max must be >= wdf-min")

            wdf_values = np.arange(wdf_min, wdf_max + (wdf_step / 2), wdf_step)
            out_dir = Path(config.paths.simulation_data) / config.simulation.name
            out_dir.mkdir(parents=True, exist_ok=True)
            mode = optimize_wdf_mode.lower()
            if mode == "fast":
                if optimize_cleanup:
                    print("Note: --optimize-cleanup is ignored in fast mode.")
                best_row, results_df = fast_grid_search_wind_drift_factor(
                    real_manchas,
                    config,
                    observed_trajectory,
                    wdf_values,
                    particles_per_wdf=fast_particles_per_wdf,
                    current_drift_factor=current_drift_factor,
                    oil_type=selected_oil_type,
                )
                results_name = "wdf_optimization_fast"
            else:
                best_row, results_df = grid_search_wind_drift_factor(
                    real_manchas,
                    config,
                    observed_trajectory,
                    wdf_values,
                    padding_animation_frame=padding_animation_frame,
                    out_prefix="sim_wdf_opt",
                    cleanup=optimize_cleanup,
                    current_drift_factor=current_drift_factor,
                    oil_type=selected_oil_type,
                )
                results_name = "wdf_optimization"

            results_df.to_csv(out_dir / f"{results_name}.csv", index=False)
            if best_row is None or pd.isna(best_row["skillscore"]):
                print("Optimization failed: no valid skillscore computed.")
                return

            wind_drift_factor = float(best_row["wind_drift_factor"])
            summary = {
                "wind_drift_factor": wind_drift_factor,
                "skillscore": float(best_row["skillscore"]),
                "wdf_min": float(wdf_min),
                "wdf_max": float(wdf_max),
                "wdf_step": float(wdf_step),
                "mode": mode,
            }
            if current_drift_factor is not None:
                summary["current_drift_factor"] = float(current_drift_factor)
            if selected_oil_type:
                summary["oil_type"] = selected_oil_type
            if mode == "fast":
                summary["particles_per_wdf"] = int(fast_particles_per_wdf)
            (out_dir / f"{results_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(
                f"Best wind drift factor: {wind_drift_factor:.4f} "
                f"oil={selected_oil_type or 'default'} "
                f"(skillscore {best_row['skillscore']:.4f})"
            )

        if optimize_stokes and not optimize_wdf_stokes and not optimize_wdf_stokes_cdf:
            if stokes_override is not None:
                print(f"Fixing stokes_drift to {stokes_override} for optimization.")
            out_dir = Path(config.paths.simulation_data) / config.simulation.name
            out_dir.mkdir(parents=True, exist_ok=True)
            best_row, results_df = grid_search_stokes_drift(
                real_manchas,
                config,
                observed_trajectory,
                stokes_values,
                padding_animation_frame=padding_animation_frame,
                cleanup=optimize_cleanup,
                wind_drift_factor=wind_drift_factor,
                current_drift_factor=current_drift_factor,
                oil_type=selected_oil_type,
            )
            results_df.to_csv(out_dir / "stokes_optimization.csv", index=False)
            if best_row is None or pd.isna(best_row["skillscore"]):
                print("Stokes optimization failed: no valid skillscore computed.")
                return

            stokes_drift = bool(best_row["stokes_drift"])
            summary = {
                "stokes_drift": stokes_drift,
                "skillscore": float(best_row["skillscore"]),
            }
            if wind_drift_factor is not None:
                summary["wind_drift_factor"] = float(wind_drift_factor)
            if current_drift_factor is not None:
                summary["current_drift_factor"] = float(current_drift_factor)
            if selected_oil_type:
                summary["oil_type"] = selected_oil_type
            (out_dir / "stokes_optimization.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(
                f"Best stokes drift: {stokes_drift} "
                f"oil={selected_oil_type or 'default'} "
                f"(skillscore {best_row['skillscore']:.4f})"
            )

        if auto_physics:
            default_diffusivity = [5.0, 10.0, 20.0, 50.0]
            default_dispersion = [False, True]
            default_evaporation = [False, True]
            hdiff_values = _parse_float_list(diffusivity_values, default_diffusivity)
            dispersion_list = _parse_bool_list(dispersion_values, default_dispersion)
            evaporation_list = _parse_bool_list(evaporation_values, default_evaporation)

            base_wdf = wind_drift_factor
            if base_wdf is None:
                base_wdf = getattr(config.simulation, "wind_drift_factor", None)
            base_stokes = stokes_drift
            if base_stokes is None:
                base_stokes = stokes_override if stokes_override is not None else getattr(config.simulation, "stokes_drift", True)
            base_cdf = current_drift_factor
            if base_cdf is None:
                base_cdf = getattr(config.simulation, "current_drift_factor", None)
            base_oil = selected_oil_type or getattr(config.simulation, "oil_type", None)
            if base_wdf is None or base_cdf is None:
                raise ValueError(
                    "optimize-physics requires wind_drift_factor and current_drift_factor. "
                    "Run --optimize-wdf-stokes-cdf first or pass values."
                )

            wind_drift_factor = float(base_wdf)
            stokes_drift = bool(base_stokes)
            current_drift_factor = float(base_cdf)
            if base_oil:
                selected_oil_type = str(base_oil)

            total_runs = len(hdiff_values) * len(dispersion_list) * len(evaporation_list)
            print(
                f"Will test {len(hdiff_values)} diffusivity x {len(dispersion_list)} dispersion x "
                f"{len(evaporation_list)} evaporation = {total_runs} simulations (robust)."
            )
            progress = _ProgressPrinter(total_runs, every=15, label="Physics progress")
            out_dir = Path(config.paths.simulation_data) / config.simulation.name
            out_dir.mkdir(parents=True, exist_ok=True)
            best_row, results_df = grid_search_physics_params(
                real_manchas,
                config,
                observed_trajectory,
                wind_drift_factor,
                stokes_drift,
                current_drift_factor,
                selected_oil_type,
                hdiff_values,
                dispersion_list,
                evaporation_list,
                padding_animation_frame=padding_animation_frame,
                out_prefix="sim_physics_opt",
                cleanup=optimize_cleanup,
                progress=progress,
            )
            results_df.to_csv(out_dir / "physics_optimization.csv", index=False)
            if best_row is None or pd.isna(best_row["skillscore"]):
                print("Physics optimization failed: no valid skillscore computed.")
                return

            horizontal_diffusivity = float(best_row["horizontal_diffusivity"])
            processes_dispersion = bool(best_row["processes_dispersion"])
            processes_evaporation = bool(best_row["processes_evaporation"])
            summary = {
                "wind_drift_factor": wind_drift_factor,
                "stokes_drift": stokes_drift,
                "current_drift_factor": current_drift_factor,
                "oil_type": selected_oil_type,
                "horizontal_diffusivity": horizontal_diffusivity,
                "processes_dispersion": processes_dispersion,
                "processes_evaporation": processes_evaporation,
                "skillscore": float(best_row["skillscore"]),
                "diffusivity_values": hdiff_values,
                "dispersion_values": dispersion_list,
                "evaporation_values": evaporation_list,
            }
            (out_dir / "physics_optimization.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(
                f"Best physics: hdiff={horizontal_diffusivity:.2f} "
                f"dispersion={processes_dispersion} evaporation={processes_evaporation} "
                f"(skillscore {best_row['skillscore']:.4f})"
            )


        #Clean and recreate output directory
        #out_dir = Path(config.paths.simulation_data) / config.simulation.name
        #rmtree(out_dir, ignore_errors=True)
        #out_dir.mkdir(parents=True, exist_ok=True)

        # -- Step 2 : Lançar as simulações se necessário
        print("Start simulation...")
        if stokes_override is not None and not optimize_stokes and not optimize_wdf_stokes and not optimize_wdf_stokes_cdf:
            stokes_drift = stokes_override
        sim_filename = f'sim_2019_P53_TEST_NOSTOKES_30WDF_75CDF'
        if wind_drift_factor is not None:
            safe_wdf = f"{wind_drift_factor:.4f}".replace(".", "p")
            sim_filename = f"{sim_filename}_wdf{safe_wdf}"
        if stokes_drift is not None:
            sim_filename = f"{sim_filename}_{'stokes' if stokes_drift else 'nostokes'}"
        if current_drift_factor is not None:
            safe_cdf = f"{current_drift_factor:.2f}".replace(".", "p")
            sim_filename = f"{sim_filename}_cdf{safe_cdf}"
        if horizontal_diffusivity is not None:
            safe_hdiff = f"{horizontal_diffusivity:.2f}".replace(".", "p")
            sim_filename = f"{sim_filename}_hd{safe_hdiff}"
        if processes_dispersion is not None:
            sim_filename = f"{sim_filename}_{'disp' if processes_dispersion else 'nodisp'}"
        if processes_evaporation is not None:
            sim_filename = f"{sim_filename}_{'evap' if processes_evaporation else 'noevap'}"

        out_dir = Path(config.paths.simulation_data) / config.simulation.name
        out_dir.mkdir(parents=True, exist_ok=True)
        run_params = {
            "environment": environment,
            "simulation_name": config.simulation.name,
        }
        if start_index:
            run_params["start_index"] = int(start_index)
        if wind_drift_factor is not None:
            run_params["wind_drift_factor"] = float(wind_drift_factor)
        if stokes_drift is not None:
            run_params["stokes_drift"] = bool(stokes_drift)
        if current_drift_factor is not None:
            run_params["current_drift_factor"] = float(current_drift_factor)
        if selected_oil_type:
            run_params["oil_type"] = selected_oil_type
        if horizontal_diffusivity is not None:
            run_params["horizontal_diffusivity"] = float(horizontal_diffusivity)
        if processes_dispersion is not None:
            run_params["processes_dispersion"] = bool(processes_dispersion)
        if processes_evaporation is not None:
            run_params["processes_evaporation"] = bool(processes_evaporation)
        (out_dir / f"{sim_filename}.json").write_text(
            json.dumps(run_params, indent=2), encoding="utf-8"
        )
        if not skip_simulation:
            simulate_drift(
                real_manchas,
                sim_filename,
                config,
                skip_animation,
                padding_animation_frame,
                wind_drift_factor=wind_drift_factor,
                stokes_drift=stokes_drift,
                current_drift_factor=current_drift_factor,
                oil_type=selected_oil_type,
                horizontal_diffusivity=horizontal_diffusivity,
                processes_dispersion=processes_dispersion,
                processes_evaporation=processes_evaporation,
            )
            print(f"The results have been generated in {Path(config.paths.simulation_data) / config.simulation.name}")
        else:
            print(f"The results are probably already present in {Path(config.paths.simulation_data) / config.simulation.name}")


        # -- Step 3 : Importar os resultados Opendrift gerados
        # Para converter um arquivo .nc em dataframe Python -> xarray
        sim_path = Path(config.paths.simulation_data) / config.simulation.name / f"{sim_filename}.nc"

        if not skip_animation:
            compare_gif = Path(config.paths.simulation_data) / config.simulation.name / f"{sim_filename}_compare.gif"
            try:
                real_steps = int(real_manchas["datetime"].nunique())
                generate_comparison_gif(
                    sim_nc=sim_path,
                    shp_zip=config.paths.plataformas_shp,
                    out=compare_gif,
                    extent=",".join(f"{v:.6f}" for v in plot_bounds),
                    datetime_offset_hours=offset_hours,
                    real_steps=real_steps,
                )
            except Exception as exc:
                print(f"Failed to generate comparison GIF: {exc}")

        if skip_plots:
            return

        ds_result = xr.open_dataset(sim_path, engine="netcdf4")


        # -- Step 4 : Fazer o plot a cada snapshot
        snapshot_datetimes = list(observed_trajectory['time'])
        for dt_snapshot in snapshot_datetimes:

            # Step 4.1 : Find closest simulation time index to the real snapshot datetime
            # O output time step que foi definido durante a simulação Openrift pode não cair perfeitamente na data da mancha real
            sim_times = pd.to_datetime(ds_result['time'].values)
            idx_time = (abs(sim_times - dt_snapshot)).argmin()

            # Step 4.2 : Extract all particle positions at this specific instant
            lons = ds_result['lon'].isel(time=idx_time).values
            lats = ds_result['lat'].isel(time=idx_time).values

            gdf_sim = gpd.GeoDataFrame(
                {'lon': lons, 'lat': lats},
                geometry=[Point(xy) for xy in zip(lons, lats)],
                crs="EPSG:4326"
            )

            # -- Step 4.3 : Select real spill polygon for this same datetime
            manchas_at_time = real_manchas[real_manchas['datetime'] == dt_snapshot]

            # -- Step 4.4 : Plot both real and simulated results
            _, ax = plt.subplots(figsize=(8, 6))
            gdf_sim.plot(ax=ax, color='blue', markersize=5, label='Simulated particles')
            manchas_at_time.plot(ax=ax, color='red', alpha=0.5, label='Observed spill')

            plt.title(f"Oil spill comparison at {dt_snapshot}")
            plt.legend()
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.grid(True)
            plt.show()


if __name__ == '__main__':
    simulate_validation()
