import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as crt
from scipy import stats
import os


def load_data(file_nrt, file_my, datetime_str):
    # Loads NRT and MY datasets and selects the specified time instant
    # Args: file_nrt (NetCDF path), file_my (NetCDF path), datetime_str (ISO format)
    # Returns: (nrt_dataset, my_dataset, nrt_slice, my_slice, actual_time_nrt, actual_time_my)
    nrt = xr.open_dataset(file_nrt)
    my = xr.open_dataset(file_my)
    
    # Select time using nearest method
    nrt_slice = nrt.sel(time=datetime_str, method="nearest")
    my_slice = my.sel(time=datetime_str, method="nearest")
    
    # Get actual selected times as strings
    actual_time_nrt = str(nrt_slice.time.values).replace('T', ' ').split('.')[0]
    actual_time_my = str(my_slice.time.values).replace('T', ' ').split('.')[0]
    requested_time_str = datetime_str.replace('T', ' ')
    
    # Validation and warnings
    print("\n" + "=" * 60)
    print("VALIDAÇÃO DO TIMESTAMP SELECIONADO:")
    print("=" * 60)
    print(f"Timestamp solicitado: {requested_time_str}")
    print(f"Timestamp NRT real:   {actual_time_nrt}")
    print(f"Timestamp MY real:    {actual_time_my}")
    
    # Check if NRT and MY have same timestamp
    if actual_time_nrt == actual_time_my:
        print("NRT e MY estão no MESMO timestamp")
    else:
        print(f"AVISO: NRT e MY têm timestamps DIFERENTES")
    
    return nrt, my, nrt_slice, my_slice, actual_time_nrt, actual_time_my


def extract_components(slice_data):
    # Extracts u, v components and coordinates from a data slice
    # Args: slice_data (xarray dataset filtered by time)
    # Returns: (u, v, lon, lat)
    u = slice_data.uo.squeeze(drop=True)
    v = slice_data.vo.squeeze(drop=True)
    lon = slice_data.longitude
    lat = slice_data.latitude
    
    return u, v, lon, lat

"""
def calculate_spatial_crop(lat, lon, lat_min_req, lat_max_req, lon_min_req, lon_max_req, n_expand=3):
    # Calculates indices for spatial crop based on center and expansion
    # Args: lat, lon (dataset coordinates), lat/lon_min/max_req (requested limits), n_expand (extra points, default: 3)
    # Returns: (lat_indices, lon_indices, lat_min, lat_max, lon_min, lon_max)
    lat_center = (lat_min_req + lat_max_req) / 2
    lon_center = (lon_min_req + lon_max_req) / 2

    print(f"Debug: {lat.values}" )
    print(f"Debug 3:  {lon.values}" )
    print(f"Debug 2:  {lat.values.shape}" )

    
    lat_center_idx = np.argmin(np.abs(lat.values - lat_center))
    lon_center_idx = np.argmin(np.abs(lon.values - lon_center))
    
    lat_min_idx = max(0, lat_center_idx - n_expand)
    lat_max_idx = min(len(lat.values) - 1, lat_center_idx + n_expand)
    lon_min_idx = max(0, lon_center_idx - n_expand)
    lon_max_idx = min(len(lon.values) - 1, lon_center_idx + n_expand)
    
    lat_indices = np.arange(lat_min_idx, lat_max_idx + 1)
    lon_indices = np.arange(lon_min_idx, lon_max_idx + 1)
    
    lat_min = lat.values[lat_min_idx]
    lat_max = lat.values[lat_max_idx]
    lon_min = lon.values[lon_min_idx]
    lon_max = lon.values[lon_max_idx]
    
    return lat_indices, lon_indices, lat_min, lat_max, lon_min, lon_max
"""

def calculate_spatial_crop(lat, lon, lat_min_req, lat_max_req, lon_min_req, lon_max_req, n_expand=3):
    # Calculates indices for spatial crop based on requested bounds and expansion
    # Args: lat, lon (dataset coordinates), lat/lon_min/max_req (requested limits), n_expand (extra points, default: 3)
    # Returns: (lat_indices, lon_indices, lat_min, lat_max, lon_min, lon_max)

    lat_vals = lat.values
    lon_vals = lon.values

    # Normalize requested bounds
    lat_lo, lat_hi = (lat_min_req, lat_max_req) if lat_min_req <= lat_max_req else (lat_max_req, lat_min_req)
    lon_lo, lon_hi = (lon_min_req, lon_max_req) if lon_min_req <= lon_max_req else (lon_max_req, lon_min_req)

    # Build mask of points inside requested bounds
    lat_mask = (lat_vals >= lat_lo) & (lat_vals <= lat_hi)
    lon_mask = (lon_vals >= lon_lo) & (lon_vals <= lon_hi)

    print(f"Debug 1: {np.where(lat_mask)[0]}")

    if lat_mask.any():
        lat_in = np.where(lat_mask)[0]
        lat_min_idx = max(0, lat_in.min() - n_expand)
        lat_max_idx = min(len(lat_vals) - 1, lat_in.max() + n_expand)
    else:
        # fallback: nearest points to requested bounds
        lat_min_idx = int(np.argmin(np.abs(lat_vals - lat_lo)))
        lat_max_idx = int(np.argmin(np.abs(lat_vals - lat_hi)))
        if lat_min_idx > lat_max_idx:
            lat_min_idx, lat_max_idx = lat_max_idx, lat_min_idx
        lat_min_idx = max(0, lat_min_idx - n_expand)
        lat_max_idx = min(len(lat_vals) - 1, lat_max_idx + n_expand)

    if lon_mask.any():
        lon_in = np.where(lon_mask)[0]
        lon_min_idx = max(0, lon_in.min() - n_expand)
        lon_max_idx = min(len(lon_vals) - 1, lon_in.max() + n_expand)
    else:
        # fallback: nearest points to requested bounds
        lon_min_idx = int(np.argmin(np.abs(lon_vals - lon_lo)))
        lon_max_idx = int(np.argmin(np.abs(lon_vals - lon_hi)))
        if lon_min_idx > lon_max_idx:
            lon_min_idx, lon_max_idx = lon_max_idx, lon_min_idx
        lon_min_idx = max(0, lon_min_idx - n_expand)
        lon_max_idx = min(len(lon_vals) - 1, lon_max_idx + n_expand)

    lat_indices = np.arange(lat_min_idx, lat_max_idx + 1)
    lon_indices = np.arange(lon_min_idx, lon_max_idx + 1)

    lat_min = lat_vals[lat_min_idx]
    lat_max = lat_vals[lat_max_idx]
    lon_min = lon_vals[lon_min_idx]
    lon_max = lon_vals[lon_max_idx]

    return lat_indices, lon_indices, lat_min, lat_max, lon_min, lon_max



def apply_crop(u, v, lat, lon, lat_indices, lon_indices):
    # Applies spatial crop to components and coordinates
    # Args: u, v (velocity components), lat, lon (coordinates), lat_indices, lon_indices (crop indices)
    # Returns: (u_crop, v_crop, lat_crop, lon_crop)
    u_crop = u.isel(latitude=lat_indices, longitude=lon_indices)
    v_crop = v.isel(latitude=lat_indices, longitude=lon_indices)
    lat_crop = lat.isel(latitude=lat_indices)
    lon_crop = lon.isel(longitude=lon_indices)
    
    return u_crop, v_crop, lat_crop, lon_crop


def ensure_dimensions(u, v):
    # Ensures components have dimensions (latitude, longitude)
    # Args: u, v (velocity components)
    # Returns: (u, v) with correct dimensions
    if u.dims != ("latitude", "longitude"):
        if set(u.dims) == {"latitude", "longitude"}:
            u = u.transpose("latitude", "longitude")
        else:
            raise ValueError(f"Dimensoes inesperadas para u: {u.dims}")
    
    if v.dims != ("latitude", "longitude"):
        if set(v.dims) == {"latitude", "longitude"}:
            v = v.transpose("latitude", "longitude")
        else:
            raise ValueError(f"Dimensoes inesperadas para v: {v.dims}")
    
    return u, v


def align_grids(u_my, v_my, lon_my, lat_my, lon_ref, lat_ref):
    # Aligns MY grid to reference grid (NRT) through interpolation
    # Args: u_my, v_my (MY components), lon_my, lat_my (MY coordinates), lon_ref, lat_ref (reference coordinates)
    # Returns: (u_my_aligned, v_my_aligned)
    if not (np.array_equal(lon_ref.values, lon_my.values) and 
            np.array_equal(lat_ref.values, lat_my.values)):
        u_my_aligned = u_my.interp(longitude=lon_ref, latitude=lat_ref)
        v_my_aligned = v_my.interp(longitude=lon_ref, latitude=lat_ref)
    else:
        u_my_aligned = u_my
        v_my_aligned = v_my
    
    return u_my_aligned, v_my_aligned


def calculate_metrics(u_nrt, v_nrt, u_my, v_my):
    # Calculates RMSE and vector error metrics for u and v components
    # Args: u_nrt, v_nrt (NRT components), u_my, v_my (components to evaluate - MY)
    # Returns: dict with comprehensive metrics
    diff_u = u_nrt - u_my
    diff_v = v_nrt - v_my
    
    # Component-wise RMSE
    rmse_u = float(np.sqrt(np.nanmean((diff_u.values) ** 2)))
    rmse_v = float(np.sqrt(np.nanmean((diff_v.values) ** 2)))
    
    # Vector magnitude-based metrics
    # Magnitude of NRT vectors
    mag_nrt = np.sqrt(u_nrt.values**2 + v_nrt.values**2)
    # Magnitude of MY vectors
    mag_my = np.sqrt(u_my.values**2 + v_my.values**2)
    # Magnitude of difference vectors
    mag_diff = np.sqrt(diff_u.values**2 + diff_v.values**2)
    
    # Vector RMSE (based on magnitude of difference vectors)
    rmse_vector = float(np.sqrt(np.nanmean(mag_diff**2)))
    
    # Mean vector error (average magnitude of difference)
    mean_vector_error = float(np.nanmean(mag_diff))
    
    # Maximum vector error
    max_vector_error = float(np.nanmax(mag_diff))
    
    # Mean magnitude of NRT vectors (for context)
    mean_mag_nrt = float(np.nanmean(mag_nrt))
    
    # Relative error (percentage)
    relative_error_pct = (mean_vector_error / mean_mag_nrt * 100) if mean_mag_nrt > 0 else np.nan
    
    return {
        'rmse_u': rmse_u,
        'rmse_v': rmse_v,
        'rmse_vector': rmse_vector,
        'mean_vector_error': mean_vector_error,
        'max_vector_error': max_vector_error,
        'mean_mag_nrt': mean_mag_nrt,
        'relative_error_pct': relative_error_pct,
        'diff_u': diff_u,
        'diff_v': diff_v
    }


def plot_comparison(u_nrt, v_nrt, u_my, v_my, diff_u, diff_v,
                     lon, lat, lon_min, lon_max, lat_min, lat_max, datetime_str):
    # Plots comparison between NRT, MY and vector difference
    # Args: u_nrt, v_nrt (NRT components), u_my, v_my (MY aligned components), diff_u, diff_v (vector differences),
    #       lon, lat (coordinates), lon/lat_min/max (spatial limits), datetime_str (date/time string)
    # Returns: (fig, axes)
    lon2d, lat2d = np.meshgrid(lon.values, lat.values)
    
    fig, axes = plt.subplots(1, 3, figsize=(28, 10), 
                             subplot_kw={'projection': crt.PlateCarree()})
    
    skip = (slice(None, None, 1), slice(None, None, 1))
    
    # Subplot NRT
    ax_nrt = axes[0]
    ax_nrt.coastlines()
    ax_nrt.set_extent([lon_min - 0.1, lon_max + 0.1, lat_min - 0.1, lat_max + 0.1], 
                      crs=crt.PlateCarree())
    ax_nrt.quiver(lon2d[skip], lat2d[skip], u_nrt.values[skip], v_nrt.values[skip],
                  color="red", angles="xy", scale_units="xy", scale=1.0, width=0.004,
                  transform=crt.PlateCarree())
    gl1 = ax_nrt.gridlines(draw_labels=True, alpha=0.4, linestyle='--')
    gl1.top_labels = False
    gl1.right_labels = False
    ax_nrt.set_title('Dados NRT', fontsize=14, weight='bold')
    
    # Subplot MY
    ax_my = axes[1]
    ax_my.coastlines()
    ax_my.set_extent([lon_min - 0.1, lon_max + 0.1, lat_min - 0.1, lat_max + 0.1], 
                     crs=crt.PlateCarree())
    ax_my.quiver(lon2d[skip], lat2d[skip], u_my.values[skip], v_my.values[skip],
                 color="blue", angles="xy", scale_units="xy", scale=1.0, width=0.004,
                 transform=crt.PlateCarree())
    gl2 = ax_my.gridlines(draw_labels=True, alpha=0.4, linestyle='--')
    gl2.top_labels = False
    gl2.right_labels = False
    ax_my.set_title('Dados MY', fontsize=14, weight='bold')
    
    # Subplot Difference
    ax_diff = axes[2]
    ax_diff.coastlines()
    ax_diff.set_extent([lon_min - 0.1, lon_max + 0.1, lat_min - 0.1, lat_max + 0.1], 
                       crs=crt.PlateCarree())
    ax_diff.quiver(lon2d[skip], lat2d[skip], diff_u.values[skip], diff_v.values[skip],
                   color="green", angles="xy", scale_units="xy", scale=1.0, width=0.004,
                   transform=crt.PlateCarree())
    gl3 = ax_diff.gridlines(draw_labels=True, alpha=0.4, linestyle='--')
    gl3.top_labels = False
    gl3.right_labels = False
    ax_diff.set_title('Diferença (NRT - MY)', fontsize=14, weight='bold')
    
    # Format datetime for display (replace T with space)
    datetime_display = datetime_str.replace("T", " ")
    
    plt.suptitle(f'Comparação Vetorial - Região Selecionada ({datetime_display})', 
                 fontsize=16, weight='bold')
    plt.tight_layout()
    
    return fig, axes

def analyze_error_distribution(diff_u, diff_v):
    # Analyzes error distribution to explain bin frequency patterns
    # Args: diff_u, diff_v (difference arrays)
    # Returns: None (prints analysis)
    from scipy import stats
    
    # Flatten and remove NaN
    diff_u_flat = diff_u.values.flatten()
    diff_v_flat = diff_v.values.flatten()
    diff_u_flat = diff_u_flat[~np.isnan(diff_u_flat)]
    diff_v_flat = diff_v_flat[~np.isnan(diff_v_flat)]
    
    print("\n" + "=" * 60)
    print("ANÁLISE DA DISTRIBUIÇÃO DE ERROS:")
    print("=" * 60)
    
    # 1. Simetria
    print("\n1. SIMETRIA DOS DADOS:")
    skew_u = stats.skew(diff_u_flat)
    skew_v = stats.skew(diff_v_flat)
    print(f"  Assimetria (Skewness) u: {skew_u:.6f}")
    print(f"  Assimetria (Skewness) v: {skew_v:.6f}")
    if abs(skew_u) < 0.5:
        print(f"    → u é aproximadamente SIMÉTRICO")
    elif skew_u > 0:
        print(f"    → u tem ASSIMETRIA POSITIVA (cauda direita)")
    else:
        print(f"    → u tem ASSIMETRIA NEGATIVA (cauda esquerda)")
    
    # 2. Curtose
    print("\n2. CURTOSE (CONCENTRAÇÃO NOS EXTREMOS):")
    kurt_u = stats.kurtosis(diff_u_flat)
    kurt_v = stats.kurtosis(diff_v_flat)
    print(f"  Curtose u: {kurt_u:.6f}")
    print(f"  Curtose v: {kurt_v:.6f}")
    if abs(kurt_u) < 0.5:
        print(f"    → u segue distribuição similar à GAUSSIANA")
    elif kurt_u > 0:
        print(f"    → u tem PICOS ACENTUADOS (leptocúrtica)")
    else:
        print(f"    → u é mais ACHATADA (platicúrtica)")
    
    # 3. Correlação
    print("\n3. CORRELAÇÃO ENTRE u E v:")
    corr = np.corrcoef(diff_u_flat, diff_v_flat)[0, 1]
    print(f"  Correlação de Pearson: {corr:.6f}")
    if abs(corr) < 0.3:
        print(f"    → u e v são INDEPENDENTES")
    elif abs(corr) < 0.7:
        print(f"    → u e v têm CORRELAÇÃO MODERADA")
    else:
        print(f"    → u e v são ALTAMENTE CORRELACIONADOS")
    
    # 4. Valores Únicos
    print("\n4. PRECISÃO NUMÉRICA:")
    unique_u = len(np.unique(diff_u_flat))
    unique_v = len(np.unique(diff_v_flat))
    print(f"  Valores únicos em u: {unique_u} de {len(diff_u_flat)} (densidade: {unique_u/len(diff_u_flat):.2%})")
    print(f"  Valores únicos em v: {unique_v} de {len(diff_v_flat)} (densidade: {unique_v/len(diff_v_flat):.2%})")
    if unique_u < len(diff_u_flat) * 0.5:
        print(f"    → u tem VALORES REPETIDOS (arredondamento?)")
    if unique_v < len(diff_v_flat) * 0.5:
        print(f"    → v tem VALORES REPETIDOS (arredondamento?)")
    
    # 5. Distribuição de Valores
    print("\n5. DISTRIBUIÇÃO DE FREQUÊNCIA DE VALORES:")
    value_counts_u = {}
    for val in diff_u_flat:
        val_round = round(val, 8)
        value_counts_u[val_round] = value_counts_u.get(val_round, 0) + 1
    
    top_values_u = sorted(value_counts_u.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top 5 valores mais frequentes em u:")
    for val, count in top_values_u:
        print(f"    {val:.8f}: {count} vezes")

def analyze_gaussian_fit(diff_u, diff_v):
    # Analyzes if error distribution follows a Gaussian (normal) distribution
    # Performs normality tests and displays ideal Gaussian parameters
    # Args: diff_u, diff_v (difference arrays)
    # Returns: None (prints analysis and generates Q-Q plots)
    
    # Flatten and remove NaN
    diff_u_flat = diff_u.values.flatten()
    diff_v_flat = diff_v.values.flatten()
    diff_u_flat = diff_u_flat[~np.isnan(diff_u_flat)]
    diff_v_flat = diff_v_flat[~np.isnan(diff_v_flat)]
    
    print("\n" + "=" * 60)
    print("TESTE DE DISTRIBUIÇÃO GAUSSIANA:")
    print("=" * 60)
    
    # Calculate parameters
    mean_u = np.mean(diff_u_flat)
    std_u = np.std(diff_u_flat)
    mean_v = np.mean(diff_v_flat)
    std_v = np.std(diff_v_flat)
    
    # Display parameters
    print("\nPARÂMETROS DOS DADOS:")
    print(f"\n  Componente U:")
    print(f"    Media (mu):     {mean_u:+.8f}")
    print(f"    Desvio (sigma): {std_u:.8f}")
    
    print(f"\n  Componente V:")
    print(f"    Media (mu):     {mean_v:+.8f}")
    print(f"    Desvio (sigma): {std_v:.8f}")
    
    # Shapiro-Wilk Test
    print("\n" + "-" * 60)
    print("SHAPIRO-WILK TEST (melhor para n < 50):")
    print("  H0: Distribuicao eh GAUSSIANA")
    print("  Rejeita H0 se p-value <= 0.05")
    print("-" * 60)
    
    stat_u_sw, p_u_sw = stats.shapiro(diff_u_flat)
    stat_v_sw, p_v_sw = stats.shapiro(diff_v_flat)
    
    result_u_sw = "GAUSSIANA" if p_u_sw > 0.05 else "NAO GAUSSIANA"
    result_v_sw = "GAUSSIANA" if p_v_sw > 0.05 else "NAO GAUSSIANA"
    
    print(f"\n  Componente U: p-value = {p_u_sw:.6f}  [{result_u_sw}]")
    print(f"  Componente V: p-value = {p_v_sw:.6f}  [{result_v_sw}]")
    
    # Jarque-Bera Test
    print("\n" + "-" * 60)
    print("JARQUE-BERA TEST (baseado em assimetria + curtose):")
    print("  H0: Distribuicao eh GAUSSIANA")
    print("  Rejeita H0 se p-value <= 0.05")
    print("-" * 60)
    
    stat_u_jb, p_u_jb = stats.jarque_bera(diff_u_flat)
    stat_v_jb, p_v_jb = stats.jarque_bera(diff_v_flat)
    
    result_u_jb = "GAUSSIANA" if p_u_jb > 0.05 else "NAO GAUSSIANA"
    result_v_jb = "GAUSSIANA" if p_v_jb > 0.05 else "NAO GAUSSIANA"
    
    print(f"\n  Componente U: p-value = {p_u_jb:.6f}  [{result_u_jb}]")
    print(f"\n  Componente V: p-value = {p_v_jb:.6f}  [{result_v_jb}]")
    
    # Summary
    print("\n" + "=" * 60)
    print("Q-Q PLOTS (comparacao visual)...")
    
    fig_qq, axes_qq = plt.subplots(1, 2, figsize=(12, 4.5))
    fig_qq.suptitle('Q-Q Plot: Comparação com Distribuição Gaussiana', fontsize=14, weight='bold')
    
    stats.probplot(diff_u_flat, dist="norm", plot=axes_qq[0])
    axes_qq[0].set_title('Componente U', fontsize=12, weight='bold')
    axes_qq[0].set_xlabel('Quantis Teóricos', fontsize=10)
    axes_qq[0].set_ylabel('Quantis Observados', fontsize=10)
    axes_qq[0].grid(True, alpha=0.3)
    
    stats.probplot(diff_v_flat, dist="norm", plot=axes_qq[1])
    axes_qq[1].set_title('Componente V', fontsize=12, weight='bold')
    axes_qq[1].set_xlabel('Quantis Teóricos', fontsize=10)
    axes_qq[1].set_ylabel('Quantis Observados', fontsize=10)
    axes_qq[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("Q-Q plots gerados (pontos na diagonal = gaussiana)")
    print("=" * 60)
    
    return fig_qq

def plot_histogramdif(diff_u, diff_v, u_my, v_my):
    # Plots histograms for u, v component differences and vector magnitude difference
    # Normalizes by maximum magnitude of MY vectors
    # Args: diff_u, diff_v (difference arrays), u_my, v_my (MY velocity components)
    # Returns: fig
    
    # Calculate magnitude of MY vectors
    mag_my = np.sqrt(u_my.values**2 + v_my.values**2)
    mag_my_max = float(np.nanmax(mag_my))
    
    print(f"\nMagnitude máxima dos vetores MY: {mag_my_max:.6f} m/s")
    
    # Flatten arrays -> To change a multidimensional array into a 1D array for histogram plotting
    diff_u_flat = diff_u.values.flatten()
    diff_v_flat = diff_v.values.flatten()
    
    # Normalize by maximum MY magnitude
    diff_u_norm = diff_u_flat / mag_my_max
    diff_v_norm = diff_v_flat / mag_my_max
    
    # Calculate vector magnitude difference (normalized)
    mag_diff_norm = np.sqrt(diff_u_norm**2 + diff_v_norm**2)
    
    # Remove NaN values
    diff_u_norm = diff_u_norm[~np.isnan(diff_u_norm)]
    diff_v_norm = diff_v_norm[~np.isnan(diff_v_norm)]
    mag_diff_norm = mag_diff_norm[~np.isnan(mag_diff_norm)]
    
    # Calculate optimal bins using Sturges' rule: k = ceil(log2(n) + 1)
    n_points = len(mag_diff_norm)
    n_bins = int(np.ceil(np.log2(n_points) + 1))
    
    print("\n" + "=" * 60)
    print("INFORMAÇÕES DOS HISTOGRAMAS:")
    print("=" * 60)
    print(f"Número de pontos: {n_points}")
    print(f"Número de bins (Regra de Sturges): {n_bins}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram diff_u (normalized)
    counts_u, bins_u, patches_u = axes[0].hist(diff_u_norm, bins=n_bins, color='red', alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Diferença u normalizada (adimensional)', fontsize=12)
    axes[0].set_ylabel('Frequência', fontsize=12)
    axes[0].set_title('Histograma Diferença u (Normalizada)', fontsize=14, weight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Print bin edges for diff_u
    print("\nBins Diferença u:")
    for i in range(len(counts_u)):
        left_bracket = "["
        right_bracket = "]" if i == len(counts_u) - 1 else ")"
        print(f"  Bin {i+1}: {left_bracket}{bins_u[i]:.6f}, {bins_u[i+1]:.6f}{right_bracket} → {int(counts_u[i])} valores")
    
    # Histogram diff_v (normalized)
    counts_v, bins_v, patches_v = axes[1].hist(diff_v_norm, bins=n_bins, color='blue', alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Diferença v normalizada (adimensional)', fontsize=12)
    axes[1].set_ylabel('Frequência', fontsize=12)
    axes[1].set_title('Histograma Diferença v (Normalizada)', fontsize=14, weight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Print bin edges for diff_v
    print("\nBins Diferença v:")
    for i in range(len(counts_v)):
        left_bracket = "["
        right_bracket = "]" if i == len(counts_v) - 1 else ")"
        print(f"  Bin {i+1}: {left_bracket}{bins_v[i]:.6f}, {bins_v[i+1]:.6f}{right_bracket} → {int(counts_v[i])} valores")
    
    # Histogram vector magnitude (normalized)
    counts_mag, bins_mag, patches_mag = axes[2].hist(mag_diff_norm, bins=n_bins, color='green', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Magnitude da Diferença Normalizada (adimensional)', fontsize=12)
    axes[2].set_ylabel('Frequência', fontsize=12)
    axes[2].set_title('Histograma Erro Vetorial (Normalizado)', fontsize=14, weight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # Print bin edges for magnitude
    print("\nBins Magnitude Vetorial:")
    for i in range(len(counts_mag)):
        left_bracket = "["
        right_bracket = "]" if i == len(counts_mag) - 1 else ")"
        print(f"  Bin {i+1}: {left_bracket}{bins_mag[i]:.6f}, {bins_mag[i+1]:.6f}{right_bracket} → {int(counts_mag[i])} valores")
    print("=" * 60)
    
    plt.suptitle(f'Distribuição dos Erros Normalizados (NRT - MY) | Norm: mag_MY_max = {mag_my_max:.4f} m/s', fontsize=16, weight='bold')
    plt.tight_layout()
    
    return fig


def analyze_all_timestamps_distribution(nrt, my, lat_indices, lon_indices):
    # Iterates over all NRT timestamps and analyses if NRT-MY spatial differences
    # follow a Gaussian distribution for each instant.
    # Args: nrt, my (full xarray datasets), lat_indices, lon_indices (pre-computed crop indices)
    # Returns: list of dicts with per-timestamp statistics
    from scipy import stats as sp_stats

    times_nrt = nrt.time.values
    results = []

    print("\n" + "=" * 60)
    print("ANÁLISE TEMPORAL DA DISTRIBUIÇÃO DAS DIFERENÇAS:")
    print("=" * 60)
    print(f"Iterando sobre {len(times_nrt)} timestamps...\n")

    for i, t in enumerate(times_nrt):
        t_str = str(t).replace('T', ' ').split('.')[0]
        try:
            nrt_slice = nrt.sel(time=t, method="nearest")
            my_slice  = my.sel(time=t, method="nearest")

            u_nrt, v_nrt, lon_nrt, lat_nrt = extract_components(nrt_slice)
            u_my,  v_my,  lon_my,  lat_my  = extract_components(my_slice)

            # Apply same spatial crop
            u_nrt, v_nrt, lat_c,    lon_c    = apply_crop(u_nrt, v_nrt, lat_nrt, lon_nrt, lat_indices, lon_indices)
            u_my,  v_my,  lat_my_c, lon_my_c = apply_crop(u_my,  v_my,  lat_my,  lon_my,  lat_indices, lon_indices)

            # Ensure correct dimensions
            u_nrt, v_nrt = ensure_dimensions(u_nrt, v_nrt)
            u_my,  v_my  = ensure_dimensions(u_my,  v_my)

            # Align MY grid to NRT grid
            u_my_al, v_my_al = align_grids(u_my, v_my, lon_my_c, lat_my_c, lon_c, lat_c)

            # Flatten and remove NaN
            du = (u_nrt - u_my_al).values.flatten()
            dv = (v_nrt - v_my_al).values.flatten()
            du = du[~np.isnan(du)]
            dv = dv[~np.isnan(dv)]

            if len(du) < 3 or len(dv) < 3:
                continue

            mean_u, std_u = float(np.mean(du)), float(np.std(du))
            mean_v, std_v = float(np.mean(dv)), float(np.std(dv))
            skew_u = float(sp_stats.skew(du))
            skew_v = float(sp_stats.skew(dv))
            kurt_u = float(sp_stats.kurtosis(du))
            kurt_v = float(sp_stats.kurtosis(dv))

            # Shapiro-Wilk: best for small samples (n < 5000)
            _, p_sw_u = sp_stats.shapiro(du)
            _, p_sw_v = sp_stats.shapiro(dv)

            results.append({
                'time': t,
                'time_str': t_str,
                'mean_u': mean_u, 'std_u': std_u,
                'mean_v': mean_v, 'std_v': std_v,
                'skew_u': skew_u, 'skew_v': skew_v,
                'kurt_u': kurt_u, 'kurt_v': kurt_v,
                'p_sw_u': p_sw_u, 'p_sw_v': p_sw_v,
                'diff_u': du,     'diff_v': dv,
                'n_points': len(du)
            })

            tag_u = "GAUSSIANA"     if p_sw_u > 0.05 else "NAO-GAUSSIANA"
            tag_v = "GAUSSIANA"     if p_sw_v > 0.05 else "NAO-GAUSSIANA"
            print(f"  [{i+1:3d}/{len(times_nrt)}] {t_str} | "
                  f"μu={mean_u:+.4f} σu={std_u:.4f} [{tag_u}] | "
                  f"μv={mean_v:+.4f} σv={std_v:.4f} [{tag_v}]")

        except Exception as e:
            print(f"  [{i+1:3d}/{len(times_nrt)}] {t_str} - ERRO: {e}")

    total = len(results)
    if total > 0:
        n_gauss_u = sum(1 for r in results if r['p_sw_u'] > 0.05)
        n_gauss_v = sum(1 for r in results if r['p_sw_v'] > 0.05)
        print(f"\nRESUMO FINAL:")
        print(f"  U: {n_gauss_u}/{total} timestamps com distribuição Gaussiana "
              f"({100 * n_gauss_u / total:.1f}%)")
        print(f"  V: {n_gauss_v}/{total} timestamps com distribuição Gaussiana "
              f"({100 * n_gauss_v / total:.1f}%)")

    return results


def plot_all_timestamps_histogram(ts_results):
    # Aggregates differences from ALL timestamps into a single histogram per component
    # and checks if the combined distribution follows a Gaussian.
    # Uses Shapiro-Wilk for n <= 5000, D'Agostino-Pearson (normaltest) for larger samples.
    # Args: ts_results (list of dicts returned by analyze_all_timestamps_distribution)
    # Returns: fig
    from scipy import stats as sp_stats

    if not ts_results:
        print("Nenhum resultado para plotar.")
        return None

    # Concatenate all timestamps into single arrays
    all_du = np.concatenate([r['diff_u'] for r in ts_results])
    all_dv = np.concatenate([r['diff_v'] for r in ts_results])
    n_ts   = len(ts_results)
    n_pts  = len(all_du)

    # MLE Gaussian fit
    mu_u, sig_u = sp_stats.norm.fit(all_du)
    mu_v, sig_v = sp_stats.norm.fit(all_dv)

    # Normality test: Shapiro-Wilk up to 5000 samples, D'Agostino-Pearson above
    if n_pts <= 5000:
        test_name = "Shapiro-Wilk"
        _, p_u = sp_stats.shapiro(all_du)
        _, p_v = sp_stats.shapiro(all_dv)
    else:
        test_name = "D'Agostino-Pearson"
        _, p_u = sp_stats.normaltest(all_du)
        _, p_v = sp_stats.normaltest(all_dv)

    result_u = "Gaussiana"     if p_u > 0.05 else "Não-Gaussiana"
    result_v = "Gaussiana"     if p_v > 0.05 else "Não-Gaussiana"

    print("\n" + "=" * 60)
    print("HISTOGRAMA AGREGADO (TODOS OS TIMESTAMPS):")
    print("=" * 60)
    print(f"  Timestamps incluídos: {n_ts}")
    print(f"  Total de pontos:      {n_pts}")
    print(f"  Teste de normalidade: {test_name}")
    print(f"\n  U | μ={mu_u:+.6f}  σ={sig_u:.6f}  p={p_u:.6f}  [{result_u}]")
    print(f"  V | μ={mu_v:+.6f}  σ={sig_v:.6f}  p={p_v:.6f}  [{result_v}]")
    print("=" * 60)

    # Bins using Sturges' rule on the combined sample
    n_bins = int(np.ceil(np.log2(n_pts) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Distribuição Agregada das Diferenças (NRT − MY) – {n_ts} timestamps, {n_pts} pontos',
        fontsize=14, weight='bold'
    )

    for ax, (comp, data, mu, sig, p_val, color) in zip(axes, [
        ('U', all_du, mu_u, sig_u, p_u, 'steelblue'),
        ('V', all_dv, mu_v, sig_v, p_v, 'coral'),
    ]):
        ax.hist(data, bins=n_bins, density=True,
                color=color, alpha=0.7, edgecolor='black', linewidth=0.4,
                label=f'{n_pts} amostras')

        # Fitted Gaussian curve
        x_fit = np.linspace(data.min(), data.max(), 400)
        ax.plot(x_fit, sp_stats.norm.pdf(x_fit, mu, sig),
                'k-', linewidth=2.0, label=f'N(μ={mu:+.4f}, σ={sig:.4f})')

        ax.axvline(0,  color='black',   linestyle='--', linewidth=0.9, label='zero')
        ax.axvline(mu, color='darkred', linestyle=':',  linewidth=1.2, label=f'μ = {mu:+.4f}')

        result_str = "Gaussiana" if p_val > 0.05 else "Não-Gaussiana"
        color_title = 'darkgreen' if p_val > 0.05 else 'darkred'
        ax.set_title(
            f'Componente {comp}  |  {test_name}: p = {p_val:.4f}  [{result_str}]',
            fontsize=11, weight='bold', color=color_title
        )
        ax.set_xlabel('Diferença (m/s)', fontsize=11)
        ax.set_ylabel('Densidade de probabilidade', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def get_data_path(filename):
    # Builds a path relative to the script directory
    # Args: filename (str) - name of the file in the analysis directory
    # Returns: absolute path to the file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, filename)
    return data_path

def print_information(nrt_dataset, metrics):
    # Prints dataset information and calculated metrics
    # Args: nrt_dataset (original NRT dataset), metrics (dict with calculated metrics)
    

    print("=" * 60)
    print("INFORMAÇÕES DO ARQUIVO:")
    print("=" * 60)
    print("\nDIMENSÕES:")
    print(nrt_dataset.sizes)
    print("\nVARIÁVEIS:")
    for var in nrt_dataset.data_vars:
        print(f"  - {var}: {nrt_dataset[var].dims} | Shape: {nrt_dataset[var].shape}")
    print("\nCOORDENADAS:")
    for coord in nrt_dataset.coords:
        print(f"  - {coord}: {nrt_dataset[coord].shape}")
    
    print("\n" + "=" * 60)
    print("MÉTRICAS DE ERRO (NRT - MY):")
    print("=" * 60)
    
    print("\n1. RMSE por componente:")
    print(f"  - RMSE u: {metrics['rmse_u']:.6f} m/s")
    print(f"  - RMSE v: {metrics['rmse_v']:.6f} m/s")
    
    print("\n2. Erro Vetorial:")
    print(f"  - RMSE vetorial: {metrics['rmse_vector']:.6f} m/s")
    print(f"  - Erro médio vetorial: {metrics['mean_vector_error']:.6f} m/s")
    print(f"  - Erro máximo vetorial: {metrics['max_vector_error']:.6f} m/s")

def main():
    # Main function - executes complete analysis
    
    # Input parameters
    file_nrt = get_data_path('cmems_obs-mob_glo_phy-cur_nrt_0.25deg_PT1H-i_1771818690342.nc')
    file_my = get_data_path('cmems_obs-mob_glo_phy-cur_my_0.25deg_PT1H-i_1771818740018.nc')
    datetime_str = "2025-01-02T00:00:00"
    lat_min_req, lat_max_req = -25.28, -25.18
    lon_min_req, lon_max_req = -43.00, -42.70
    n_expand = 3
    
    # 1. Load data
    print(f"  Arquivo NRT: {file_nrt}")
    print(f"  Arquivo MY:  {file_my}")
    nrt, my, nrt_slice, my_slice, actual_time_nrt, actual_time_my = load_data(file_nrt, file_my, datetime_str)
    
    # Display available timesteps
    print("\n" + "=" * 60)
    print("TIMESTEPS DISPONÍVEIS NO ARQUIVO NRT:")
    print("=" * 60)
    print(f"Total de timesteps: {len(nrt.time.values)}")
    print("\nLista de datas/horas:")
    for i, time in enumerate(nrt.time.values):
        time_str = str(time).replace('T', ' ').split('.')[0]  # Format: YYYY-MM-DD HH:MM:SS
        print(f"  {i+1:3d}. {time_str}")
    print("=" * 60)
    print(f"\nTimestep selecionado para análise: {datetime_str.replace('T', ' ')}\n")
    
    # 2. Extract components
    u, v, lon, lat = extract_components(nrt_slice)
    u_my, v_my, lon_my, lat_my = extract_components(my_slice)
    
    # 3. Calculate spatial crop
    print(f"\nIntervalo solicitado: lat [{lat_min_req}, {lat_max_req}], "
          f"lon [{lon_min_req}, {lon_max_req}]")
    lat_indices, lon_indices, lat_min, lat_max, lon_min, lon_max = calculate_spatial_crop(
        lat, lon, lat_min_req, lat_max_req, lon_min_req, lon_max_req, n_expand
    )
    print(f"Intervalo ampliado: lat [{lat_min:.4f}, {lat_max:.4f}], "
          f"lon [{lon_min:.4f}, {lon_max:.4f}]")
    print(f"Pontos encontrados: {len(lat_indices)} lat x {len(lon_indices)} lon")
    
    # 4. Apply crop
    if len(lat_indices) > 0 and len(lon_indices) > 0:
        u, v, lat, lon = apply_crop(u, v, lat, lon, lat_indices, lon_indices)
        u_my, v_my, lat_my, lon_my = apply_crop(u_my, v_my, lat_my, lon_my, 
                                                       lat_indices, lon_indices)
    else:
        print("AVISO: Nenhum ponto encontrado no recorte!")
        return
    
    # 5. Ensure correct dimensions
    u, v = ensure_dimensions(u, v)
    u_my, v_my = ensure_dimensions(u_my, v_my)
    
    # 6. Align grids
    u_my_aligned, v_my_aligned = align_grids(u_my, v_my, lon_my, lat_my, lon, lat)
    
    # 7. Calculate metrics
    metrics = calculate_metrics(u, v, u_my_aligned, v_my_aligned)
    
    # 8. Plot
    fig, axes = plot_comparison(u, v, u_my_aligned, v_my_aligned,
                                   metrics['diff_u'], metrics['diff_v'],
                                   lon, lat, lon_min, lon_max, lat_min, lat_max,
                                   datetime_str)
    
    # 9. Print information
    print_information(nrt, metrics)
    
    # 9.5. Analyze error distribution
    analyze_error_distribution(metrics['diff_u'], metrics['diff_v'])
    
    # 9.6. Analyze Gaussian fit
    fig_qq = analyze_gaussian_fit(metrics['diff_u'], metrics['diff_v'])
    
    # 10. Plot histograms
    fig_hist = plot_histogramdif(metrics['diff_u'], metrics['diff_v'], u_my_aligned, v_my_aligned)

    # 11. Analyse distribution over all timestamps
    ts_results = analyze_all_timestamps_distribution(nrt, my, lat_indices, lon_indices)

    # 12. Plot aggregated histogram across all timestamps with Gaussian fit
    fig_agg_hist = plot_all_timestamps_histogram(ts_results)

    plt.show()

if __name__ == "__main__":
    main()
