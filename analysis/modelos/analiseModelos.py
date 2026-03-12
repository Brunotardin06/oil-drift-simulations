import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as crt
from scipy import stats
import os


def load_data(file_ref, file_cmp, datetime_str):
    ref = xr.open_dataset(file_ref)
    cmp = xr.open_dataset(file_cmp)

    ref_slice = ref.sel(time=datetime_str, method="nearest")
    cmp_slice = cmp.sel(time=datetime_str, method="nearest")

    actual_time_ref = str(ref_slice.time.values).replace('T', ' ').split('.')[0]
    actual_time_cmp = str(cmp_slice.time.values).replace('T', ' ').split('.')[0]
    requested_time_str = datetime_str.replace('T', ' ')

    print("\n" + "=" * 60)
    print("VALIDAÇÃO DO TIMESTAMP SELECIONADO:")
    print("=" * 60)
    print(f"Timestamp solicitado: {requested_time_str}")
    print(f"Timestamp REF real:   {actual_time_ref}")
    print(f"Timestamp CMP real:   {actual_time_cmp}")

    if actual_time_ref == actual_time_cmp:
        print("REF e CMP estão no MESMO timestamp")
    else:
        print("AVISO: REF e CMP têm timestamps DIFERENTES")

    return ref, cmp, ref_slice, cmp_slice, actual_time_ref, actual_time_cmp


def extract_components(slice_data):
    # Extracts u, v components and coordinates from a data slice
    # Args: slice_data (xarray dataset filtered by time)
    # Returns: (u, v, lon, lat)
    u = slice_data.sw_cur_u.squeeze(drop=True)
    v = slice_data.sw_cur_v.squeeze(drop=True)
    lon = slice_data.longitude
    lat = slice_data.latitude

    return u, v, lon, lat


def calculate_spatial_crop(lat, lon, lat_min_req=None, lat_max_req=None, lon_min_req=None, lon_max_req=None, n_expand=3):
    lat_vals = lat.values
    lon_vals = lon.values
    n_expand = max(0, int(n_expand))

    def compute_axis_indices(axis_vals, req_min, req_max):
        if req_min is None and req_max is None:
            return np.arange(len(axis_vals))

        axis_min = float(np.min(axis_vals))
        axis_max = float(np.max(axis_vals))

        lo = axis_min if req_min is None else req_min
        hi = axis_max if req_max is None else req_max
        lo, hi = (lo, hi) if lo <= hi else (hi, lo)

        mask = (axis_vals >= lo) & (axis_vals <= hi)

        if mask.any():
            axis_in = np.where(mask)[0]
            min_idx = max(0, axis_in.min() - n_expand)
            max_idx = min(len(axis_vals) - 1, axis_in.max() + n_expand)
        else:
            min_idx = int(np.argmin(np.abs(axis_vals - lo)))
            max_idx = int(np.argmin(np.abs(axis_vals - hi)))
            if min_idx > max_idx:
                min_idx, max_idx = max_idx, min_idx
            min_idx = max(0, min_idx - n_expand)
            max_idx = min(len(axis_vals) - 1, max_idx + n_expand)

        return np.arange(min_idx, max_idx + 1)

    lat_indices = compute_axis_indices(lat_vals, lat_min_req, lat_max_req)
    lon_indices = compute_axis_indices(lon_vals, lon_min_req, lon_max_req)

    lat_crop_vals = lat_vals[lat_indices]
    lon_crop_vals = lon_vals[lon_indices]

    lat_min = float(np.min(lat_crop_vals))
    lat_max = float(np.max(lat_crop_vals))
    lon_min = float(np.min(lon_crop_vals))
    lon_max = float(np.max(lon_crop_vals))

    return lat_indices, lon_indices, lat_min, lat_max, lon_min, lon_max


def apply_crop(u, v, lat, lon, lat_indices, lon_indices):
    u_crop = u.isel(latitude=lat_indices, longitude=lon_indices)
    v_crop = v.isel(latitude=lat_indices, longitude=lon_indices)
    lat_crop = lat.isel(latitude=lat_indices)
    lon_crop = lon.isel(longitude=lon_indices)
    return u_crop, v_crop, lat_crop, lon_crop


def ensure_dimensions(u, v):
    if u.dims != ("latitude", "longitude"):
        if set(u.dims) == {"latitude", "longitude"}:
            u = u.transpose("latitude", "longitude")
        else:
            raise ValueError(f"Dimensões inesperadas para u: {u.dims}")

    if v.dims != ("latitude", "longitude"):
        if set(v.dims) == {"latitude", "longitude"}:
            v = v.transpose("latitude", "longitude")
        else:
            raise ValueError(f"Dimensões inesperadas para v: {v.dims}")

    return u, v


def align_grids(u_cmp, v_cmp, lon_cmp, lat_cmp, lon_ref, lat_ref):
    if not (np.array_equal(lon_ref.values, lon_cmp.values) and np.array_equal(lat_ref.values, lat_cmp.values)):
        u_cmp_aligned = u_cmp.interp(longitude=lon_ref, latitude=lat_ref)
        v_cmp_aligned = v_cmp.interp(longitude=lon_ref, latitude=lat_ref)
    else:
        u_cmp_aligned = u_cmp
        v_cmp_aligned = v_cmp
    return u_cmp_aligned, v_cmp_aligned


def calculate_metrics(u_ref, v_ref, u_cmp, v_cmp):
    diff_u = u_ref - u_cmp
    diff_v = v_ref - v_cmp

    rmse_u = float(np.sqrt(np.nanmean((diff_u.values) ** 2)))
    rmse_v = float(np.sqrt(np.nanmean((diff_v.values) ** 2)))

    mag_ref = np.sqrt(u_ref.values**2 + v_ref.values**2)
    mag_diff = np.sqrt(diff_u.values**2 + diff_v.values**2)

    rmse_vector = float(np.sqrt(np.nanmean(mag_diff**2)))
    mean_vector_error = float(np.nanmean(mag_diff))
    max_vector_error = float(np.nanmax(mag_diff))
    mean_mag_ref = float(np.nanmean(mag_ref))
    relative_error_pct = (mean_vector_error / mean_mag_ref * 100) if mean_mag_ref > 0 else np.nan

    return {
        'rmse_u': rmse_u,
        'rmse_v': rmse_v,
        'rmse_vector': rmse_vector,
        'mean_vector_error': mean_vector_error,
        'max_vector_error': max_vector_error,
        'mean_mag_ref': mean_mag_ref,
        'relative_error_pct': relative_error_pct,
        'diff_u': diff_u,
        'diff_v': diff_v
    }


def plot_comparison(u_ref, v_ref, u_cmp, v_cmp, diff_u, diff_v,
                    lon, lat, lon_min, lon_max, lat_min, lat_max, datetime_str,
                    label_ref="REF", label_cmp="CMP"):
    lon2d, lat2d = np.meshgrid(lon.values, lat.values)

    fig, axes = plt.subplots(1, 3, figsize=(28, 10), subplot_kw={'projection': crt.PlateCarree()})

    skip = (slice(None, None, 1), slice(None, None, 1))

    ax_ref = axes[0]
    ax_ref.coastlines()
    ax_ref.set_extent([lon_min - 0.1, lon_max + 0.1, lat_min - 0.1, lat_max + 0.1], crs=crt.PlateCarree())
    ax_ref.quiver(lon2d[skip], lat2d[skip], u_ref.values[skip], v_ref.values[skip],
                  color="red", angles="xy", scale_units="xy", scale=1.0, width=0.004,
                  transform=crt.PlateCarree())
    gl1 = ax_ref.gridlines(draw_labels=True, alpha=0.4, linestyle='--')
    gl1.top_labels = False
    gl1.right_labels = False
    ax_ref.set_title(f'Dados {label_ref}', fontsize=14, weight='bold')

    ax_cmp = axes[1]
    ax_cmp.coastlines()
    ax_cmp.set_extent([lon_min - 0.1, lon_max + 0.1, lat_min - 0.1, lat_max + 0.1], crs=crt.PlateCarree())
    ax_cmp.quiver(lon2d[skip], lat2d[skip], u_cmp.values[skip], v_cmp.values[skip],
                  color="blue", angles="xy", scale_units="xy", scale=1.0, width=0.004,
                  transform=crt.PlateCarree())
    gl2 = ax_cmp.gridlines(draw_labels=True, alpha=0.4, linestyle='--')
    gl2.top_labels = False
    gl2.right_labels = False
    ax_cmp.set_title(f'Dados {label_cmp}', fontsize=14, weight='bold')

    ax_diff = axes[2]
    ax_diff.coastlines()
    ax_diff.set_extent([lon_min - 0.1, lon_max + 0.1, lat_min - 0.1, lat_max + 0.1], crs=crt.PlateCarree())
    ax_diff.quiver(lon2d[skip], lat2d[skip], diff_u.values[skip], diff_v.values[skip],
                   color="green", angles="xy", scale_units="xy", scale=1.0, width=0.004,
                   transform=crt.PlateCarree())
    gl3 = ax_diff.gridlines(draw_labels=True, alpha=0.4, linestyle='--')
    gl3.top_labels = False
    gl3.right_labels = False
    ax_diff.set_title(f'Diferença ({label_ref} - {label_cmp})', fontsize=14, weight='bold')

    datetime_display = datetime_str.replace("T", " ")
    plt.suptitle(
        f'Comparação Vetorial - Região Selecionada ({datetime_display})\n'
        f'{label_ref} vs {label_cmp}',
        fontsize=16,
        weight='bold'
    )
    fig.text(
        0.5,
        0.01,
        f'Janela espacial: lat [{lat_min:.4f}, {lat_max:.4f}] | '
        f'lon [{lon_min:.4f}, {lon_max:.4f}] | grade: {len(lat)} x {len(lon)}',
        ha='center',
        fontsize=10
    )
    plt.tight_layout()
    return fig, axes


def analyze_all_timestamps_distribution(ref, cmp, lat_indices_ref, lon_indices_ref, lat_indices_cmp, lon_indices_cmp):
    times_ref = ref.time.values
    results = []

    print("\n" + "=" * 60)
    print("ANÁLISE TEMPORAL DA DISTRIBUIÇÃO DAS DIFERENÇAS:")
    print("=" * 60)
    print(f"Iterando sobre {len(times_ref)} timestamps\n")

    for i, t in enumerate(times_ref):
        t_str = str(t).replace('T', ' ').split('.')[0]
        try:
            ref_slice = ref.sel(time=t, method="nearest")
            cmp_slice = cmp.sel(time=t, method="nearest")

            u_ref, v_ref, lon_ref, lat_ref = extract_components(ref_slice)
            u_cmp, v_cmp, lon_cmp, lat_cmp = extract_components(cmp_slice)

            u_ref, v_ref, lat_c, lon_c = apply_crop(u_ref, v_ref, lat_ref, lon_ref, lat_indices_ref, lon_indices_ref)
            u_cmp, v_cmp, lat_cmp_c, lon_cmp_c = apply_crop(u_cmp, v_cmp, lat_cmp, lon_cmp, lat_indices_cmp, lon_indices_cmp)

            u_ref, v_ref = ensure_dimensions(u_ref, v_ref)
            u_cmp, v_cmp = ensure_dimensions(u_cmp, v_cmp)

            u_cmp_al, v_cmp_al = align_grids(u_cmp, v_cmp, lon_cmp_c, lat_cmp_c, lon_c, lat_c)

            du = (u_ref - u_cmp_al).values.flatten()
            dv = (v_ref - v_cmp_al).values.flatten()
            du = du[~np.isnan(du)]
            dv = dv[~np.isnan(dv)]

            if len(du) < 3 or len(dv) < 3:
                continue

            mean_u, std_u = float(np.mean(du)), float(np.std(du))
            mean_v, std_v = float(np.mean(dv)), float(np.std(dv))

            if len(du) <= 5000:
                test_name = "Shapiro-Wilk"
                _, p_sw_u = stats.shapiro(du)
                _, p_sw_v = stats.shapiro(dv)
            else:
                test_name = "D'Agostino-Pearson"
                _, p_sw_u = stats.normaltest(du)
                _, p_sw_v = stats.normaltest(dv)

            results.append({
                'time': t,
                'time_str': t_str,
                'mean_u': mean_u, 'std_u': std_u,
                'mean_v': mean_v, 'std_v': std_v,
                'normality_test': test_name,
                'p_sw_u': p_sw_u, 'p_sw_v': p_sw_v,
                'diff_u': du, 'diff_v': dv,
                'n_points': len(du)
            })

            # Output detalhado por timestamp desativado temporariamente.

        except Exception as e:
            print(f"  [{i+1:3d}/{len(times_ref)}] {t_str} - ERRO: {e}")

    return results


def plot_all_timestamps_histogram(ts_results):
    if not ts_results:
        print("Nenhum resultado para plotar.")
        return None

    all_du = np.concatenate([r['diff_u'] for r in ts_results])
    all_dv = np.concatenate([r['diff_v'] for r in ts_results])
    n_ts = len(ts_results)
    n_pts = len(all_du)

    mu_u, sig_u = stats.norm.fit(all_du)
    mu_v, sig_v = stats.norm.fit(all_dv)

    if n_pts <= 5000:
        test_name = "Shapiro-Wilk"
        _, p_u = stats.shapiro(all_du)
        _, p_v = stats.shapiro(all_dv)
    else:
        test_name = "D'Agostino-Pearson"
        _, p_u = stats.normaltest(all_du)
        _, p_v = stats.normaltest(all_dv)

    result_u = "Gaussiana" if p_u > 0.05 else "Não-Gaussiana"
    result_v = "Gaussiana" if p_v > 0.05 else "Não-Gaussiana"

    # Practical normality assessment (important for very large N)
    skew_u = float(stats.skew(all_du))
    skew_v = float(stats.skew(all_dv))
    kurt_u = float(stats.kurtosis(all_du))
    kurt_v = float(stats.kurtosis(all_dv))
    qq_r_u = float(stats.probplot(all_du, dist="norm")[1][2])
    qq_r_v = float(stats.probplot(all_dv, dist="norm")[1][2])

    approx_u = (abs(skew_u) < 0.5) and (abs(kurt_u) < 1.0) and (qq_r_u > 0.995)
    approx_v = (abs(skew_v) < 0.5) and (abs(kurt_v) < 1.0) and (qq_r_v > 0.995)
    practical_u = "Aproximadamente Gaussiana" if approx_u else "Não-Gaussiana"
    practical_v = "Aproximadamente Gaussiana" if approx_v else "Não-Gaussiana"

    print("\n" + "=" * 60)
    print("HISTOGRAMA AGREGADO (TODOS OS TIMESTAMPS):")
    print("=" * 60)
    print(f"  Timestamps incluídos: {n_ts}")
    print(f"  Total de pontos:      {n_pts}")
    print(f"  Teste de normalidade: {test_name}")
    print("  Observação: com N muito grande, p-value tende a rejeitar normalidade por desvios mínimos.")
    print(f"\n  U | μ={mu_u:+.6f}  σ={sig_u:.6f}  p={p_u:.10f}  [{result_u}]")
    print(f"    Avaliação prática: [{practical_u}] | skew={skew_u:+.4f}, kurt={kurt_u:+.4f}, QQ-r={qq_r_u:.6f}")
    print(f"  V | μ={mu_v:+.6f}  σ={sig_v:.6f}  p={p_v:.10f}  [{result_v}]")
    print(f"    Avaliação prática: [{practical_v}] | skew={skew_v:+.4f}, kurt={kurt_v:+.4f}, QQ-r={qq_r_v:.6f}")
    print("=" * 60)

    n_bins = int(np.ceil(np.log2(n_pts) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Distribuição Agregada das Diferenças (REF − CMP) – {n_ts} timestamps, {n_pts} pontos', fontsize=14, weight='bold')

    for ax, (comp, data, mu, sig, p_val, color) in zip(axes, [
        ('U', all_du, mu_u, sig_u, p_u, 'steelblue'),
        ('V', all_dv, mu_v, sig_v, p_v, 'coral'),
    ]):
        ax.hist(data, bins=n_bins, density=True, color=color, alpha=0.7, edgecolor='black', linewidth=0.4,
                label=f'{n_pts} amostras')

        x_fit = np.linspace(data.min(), data.max(), 400)
        ax.plot(x_fit, stats.norm.pdf(x_fit, mu, sig), 'k-', linewidth=2.0, label=f'N(μ={mu:+.4f}, σ={sig:.4f})')

        ax.axvline(0, color='black', linestyle='--', linewidth=0.9, label='zero')
        ax.axvline(mu, color='darkred', linestyle=':', linewidth=1.2, label=f'μ = {mu:+.4f}')
        if comp == 'U':
            result_str = f"{result_u} | prática: {practical_u}"
            color_title = 'darkgreen' if approx_u else 'darkred'
        else:
            result_str = f"{result_v} | prática: {practical_v}"
            color_title = 'darkgreen' if approx_v else 'darkred'
        ax.set_title(f'Componente {comp}  |  {test_name}: p = {p_val:.10f}  [{result_str}]',
                     fontsize=11, weight='bold', color=color_title)
        ax.set_xlabel('Diferença (m/s)', fontsize=11)
        ax.set_ylabel('Densidade de probabilidade', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def get_data_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)


def print_information(ref_dataset, cmp_dataset, metrics):
    print("=" * 60)
    print("INFORMAÇÕES DO ARQUIVO REF:")
    print("=" * 60)
    print("\nDIMENSÕES:")
    print(ref_dataset.sizes)
    print("\nVARIÁVEIS:")
    for var in ref_dataset.data_vars:
        print(f"  - {var}: {ref_dataset[var].dims} | Shape: {ref_dataset[var].shape}")

    print("\n" + "=" * 60)
    print("INFORMAÇÕES DO ARQUIVO CMP:")
    print("=" * 60)
    print("\nDIMENSÕES:")
    print(cmp_dataset.sizes)
    print("\nVARIÁVEIS:")
    for var in cmp_dataset.data_vars:
        print(f"  - {var}: {cmp_dataset[var].dims} | Shape: {cmp_dataset[var].shape}")

    print("\n" + "=" * 60)
    print("MÉTRICAS DE ERRO (REF - CMP):")
    print("=" * 60)
    print("\n1. RMSE por componente:")
    print(f"  - RMSE u: {metrics['rmse_u']:.6f} m/s")
    print(f"  - RMSE v: {metrics['rmse_v']:.6f} m/s")
    print("\n2. Erro Vetorial:")
    print(f"  - RMSE vetorial: {metrics['rmse_vector']:.6f} m/s")
    print(f"  - Erro médio vetorial: {metrics['mean_vector_error']:.6f} m/s")
    print(f"  - Erro máximo vetorial: {metrics['max_vector_error']:.6f} m/s")


def main():
    file_ref = get_data_path('corrente_cmems_mod_glo_phy_anfc_0.083deg_RUN_2025-06-12.nc')
    file_cmp = get_data_path('corrente_remo-hycom124v2_2d_map-runs-remo_hycom124v2_2d_map_RUN_2025-06-12.nc')
    label_ref = "CMEMS"
    label_cmp = "REMO-HYCOM"
    datetime_str = "2025-06-13T12:00:00"

    lat_min_req, lat_max_req = None, None
    lon_min_req, lon_max_req = None, None
    n_expand = 3

    print(f"Arquivo REF: {file_ref}")
    print(f"Arquivo CMP: {file_cmp}")
    ref, cmp, ref_slice, cmp_slice, _, _ = load_data(file_ref, file_cmp, datetime_str)

    u_ref, v_ref, lon_ref, lat_ref = extract_components(ref_slice)
    u_cmp, v_cmp, lon_cmp, lat_cmp = extract_components(cmp_slice)

    lat_indices_ref, lon_indices_ref, lat_min, lat_max, lon_min, lon_max = calculate_spatial_crop(
        lat_ref, lon_ref, lat_min_req, lat_max_req, lon_min_req, lon_max_req, n_expand
    )

    lat_indices_cmp, lon_indices_cmp, _, _, _, _ = calculate_spatial_crop(
        lat_cmp, lon_cmp, lat_min, lat_max, lon_min, lon_max, 0
    )

    if len(lat_indices_ref) == 0 or len(lon_indices_ref) == 0 or len(lat_indices_cmp) == 0 or len(lon_indices_cmp) == 0:
        print("AVISO: Nenhum ponto encontrado no recorte!")
        return

    u_ref, v_ref, lat_ref, lon_ref = apply_crop(u_ref, v_ref, lat_ref, lon_ref, lat_indices_ref, lon_indices_ref)
    u_cmp, v_cmp, lat_cmp, lon_cmp = apply_crop(u_cmp, v_cmp, lat_cmp, lon_cmp, lat_indices_cmp, lon_indices_cmp)

    u_ref, v_ref = ensure_dimensions(u_ref, v_ref)
    u_cmp, v_cmp = ensure_dimensions(u_cmp, v_cmp)

    u_cmp_aligned, v_cmp_aligned = align_grids(u_cmp, v_cmp, lon_cmp, lat_cmp, lon_ref, lat_ref)
    metrics = calculate_metrics(u_ref, v_ref, u_cmp_aligned, v_cmp_aligned)

    plot_comparison(
        u_ref, v_ref, u_cmp_aligned, v_cmp_aligned,
        metrics['diff_u'], metrics['diff_v'],
        lon_ref, lat_ref, lon_min, lon_max, lat_min, lat_max, datetime_str,
        label_ref=label_ref, label_cmp=label_cmp
    )

    print_information(ref, cmp, metrics)

    ts_results = analyze_all_timestamps_distribution(
        ref, cmp,
        lat_indices_ref, lon_indices_ref,
        lat_indices_cmp, lon_indices_cmp
    )
    plot_all_timestamps_histogram(ts_results)

    plt.show()


if __name__ == "__main__":
    main()

