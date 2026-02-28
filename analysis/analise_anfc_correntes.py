import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as crt
from scipy import stats
import os
import pandas as pd

def get_data_path(filename):
    """Build path relative to script directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)

def load_data(anfc_path, cmems_my_path, datetime_str):
    """
    Carrega datasets ANFC e CMEMS MY
    ANFC sera degradado de 0.083 deg para 0.25 deg (resolucao CMEMS)
    para comparacao sem interpolacao do CMEMS
    """
    print("\n" + "=" * 80)
    print("CARREGANDO DADOS ANFC (0.083 deg) E CMEMS MY (0.25 deg)")
    print("=" * 80)
    
    try:
        anfc = xr.open_dataset(anfc_path)
        
        cmems_my = xr.open_dataset(cmems_my_path)
        
        # Debug: Mostrar cobertura temporal dos datasets
        print("\n" + "-" * 80)
        print("COBERTURA TEMPORAL DOS DATASETS:")
        print("-" * 80)
        
        anfc_times = pd.to_datetime(anfc.time.values)
        cmems_my_times = pd.to_datetime(cmems_my.time.values)
        
        print(f"ANFC (aguaANFC.nc):")
        print(f"  Primeiro timestamp: {anfc_times[0]}")
        print(f"  Último timestamp:   {anfc_times[-1]}")
        print(f"  Total de timestamps: {len(anfc_times)}")
        print(f"  Período: {(anfc_times[-1] - anfc_times[0]).total_seconds() / 3600:.1f} horas")
        
        print(f"\nCMEMS MY (aguaMultiYear.nc):")
        print(f"  Primeiro timestamp: {cmems_my_times[0]}")
        print(f"  Último timestamp:   {cmems_my_times[-1]}")
        print(f"  Total de timestamps: {len(cmems_my_times)}")
        print(f"  Período: {(cmems_my_times[-1] - cmems_my_times[0]).total_seconds() / 3600:.1f} horas")
        
        # Verificar se os períodos se sobrepõem
        overlap_start = max(anfc_times[0], cmems_my_times[0])
        overlap_end = min(anfc_times[-1], cmems_my_times[-1])
        
        if overlap_start <= overlap_end:
            print(f"\nPERÍODO DE SOBREPOSIÇÃO:")
            print(f"  Início: {overlap_start}")
            print(f"  Fim:    {overlap_end}")
            print(f"  Duração: {(overlap_end - overlap_start).total_seconds() / 3600:.1f} horas")
        else:
            print(f"\nAVISO: Datasets NÃO se sobrepõem temporalmente!")
            print(f"  ANFC:     {anfc_times[0]} a {anfc_times[-1]}")
            print(f"  CMEMS MY: {cmems_my_times[0]} a {cmems_my_times[-1]}")
        
        # Selecionar tempo mais próximo
        anfc_slice = anfc.sel(time=datetime_str, method="nearest")
        cmems_my_slice = cmems_my.sel(time=datetime_str, method="nearest")
        
        # Obter tempos reais
        time_anfc = str(anfc_slice.time.values).replace('T', ' ').split('.')[0]
        time_my = str(cmems_my_slice.time.values).replace('T', ' ').split('.')[0]
        
        print("\n" + "-" * 80)
        print("TIMESTAMPS SELECIONADOS:")
        print("-" * 80)
        print(f"Timestamp solicitado: {datetime_str.replace('T', ' ')}")
        print(f"ANFC real:           {time_anfc}")
        print(f"CMEMS MY real:       {time_my}")
        
        return anfc, cmems_my, anfc_slice, cmems_my_slice
        
    except Exception as e:
        print(f"\nErro ao carregar dados: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None

def extract_components(slice_data):
    """Extrai componentes u, v e coordenadas"""
    u = slice_data.uo.squeeze(drop=True)
    v = slice_data.vo.squeeze(drop=True)
    lon = slice_data.longitude
    lat = slice_data.latitude
    
    return u, v, lon, lat

def ensure_dimensions(u, v):
    """
    Garante que u, v têm dimensões (latitude, longitude)
    Se tiverem 3D com depth, extrai o primeiro nível (superfície)
    """
    # Se tiver dimensão depth, selecionar o primeiro nível (superfície)
    if 'depth' in u.dims:
        print(f"Extraindo superfície (depth=0) de u: {u.dims} para ", end="")
        u = u.isel(depth=0, drop=True)
        print(f"{u.dims}")
    
    if 'depth' in v.dims:
        print(f"Extraindo superfície (depth=0) de v: {v.dims} para ", end="")
        v = v.isel(depth=0, drop=True)
        print(f"{v.dims}")
    
    # Verificar dimensões finais
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

def find_common_geographic_area(lon_anfc, lat_anfc, lon_cmems, lat_cmems):
    """
    Encontra a area geografica comum entre ANFC e CMEMS
    Retorna os limites da intersecao
    """
    print("\n" + "-" * 80)
    print("VERIFICANDO COBERTURA GEOGRAFICA")
    print("-" * 80)
    
    # Coberturas originais
    lon_anfc_min, lon_anfc_max = float(lon_anfc.values.min()), float(lon_anfc.values.max())
    lat_anfc_min, lat_anfc_max = float(lat_anfc.values.min()), float(lat_anfc.values.max())
    
    lon_cmems_min, lon_cmems_max = float(lon_cmems.values.min()), float(lon_cmems.values.max())
    lat_cmems_min, lat_cmems_max = float(lat_cmems.values.min()), float(lat_cmems.values.max())
    
    print(f"Cobertura ANFC original:")
    print(f"  lon: [{lon_anfc_min:.4f}, {lon_anfc_max:.4f}]")
    print(f"  lat: [{lat_anfc_min:.4f}, {lat_anfc_max:.4f}]")
    
    print(f"\nCobertura CMEMS original:")
    print(f"  lon: [{lon_cmems_min:.4f}, {lon_cmems_max:.4f}]")
    print(f"  lat: [{lat_cmems_min:.4f}, {lat_cmems_max:.4f}]")
    
    # Calcular intersecao
    lon_common_min = max(lon_anfc_min, lon_cmems_min)
    lon_common_max = min(lon_anfc_max, lon_cmems_max)
    lat_common_min = max(lat_anfc_min, lat_cmems_min)
    lat_common_max = min(lat_anfc_max, lat_cmems_max)
    
    # Verificar se ha intersecao
    if lon_common_min >= lon_common_max or lat_common_min >= lat_common_max:
        print("\nERRO: Nao ha intersecao geografica entre os datasets!")
        return None
    
    print(f"\nArea comum (intersecao):")
    print(f"  lon: [{lon_common_min:.4f}, {lon_common_max:.4f}]")
    print(f"  lat: [{lat_common_min:.4f}, {lat_common_max:.4f}]")
    
    # Calcular areas
    area_anfc = (lon_anfc_max - lon_anfc_min) * (lat_anfc_max - lat_anfc_min)
    area_cmems = (lon_cmems_max - lon_cmems_min) * (lat_cmems_max - lat_cmems_min)
    area_common = (lon_common_max - lon_common_min) * (lat_common_max - lat_common_min)
    
    print(f"\nPercentual de cobertura comum:")
    print(f"  ANFC:  {area_common/area_anfc*100:.1f}% da area original")
    print(f"  CMEMS: {area_common/area_cmems*100:.1f}% da area original")
    
    return {
        'lon_min': lon_common_min,
        'lon_max': lon_common_max,
        'lat_min': lat_common_min,
        'lat_max': lat_common_max
    }

def degrade_anfc_to_cmems_resolution(u_anfc, v_anfc, lon_anfc, lat_anfc, lon_cmems, lat_cmems, common_area=None):
    """
    Degrada resolucao ANFC (0.083 deg) para resolucao CMEMS (0.25 deg)
    usando coarse-graining (media de blocos)
    Se common_area fornecido, recorta ambos datasets para area comum
    """
    print("\n" + "-" * 80)
    print("DEGRADANDO RESOLUCAO ANFC PARA CMEMS")
    print("-" * 80)
    print(f"Grade ANFC original:  {len(lon_anfc)} x {len(lat_anfc)} pontos (0.083 deg)")
    print(f"Grade CMEMS alvo:     {len(lon_cmems)} x {len(lat_cmems)} pontos (0.25 deg)")
    
    # Se area comum fornecida, recortar CMEMS para essa area
    if common_area is not None:
        print(f"\nRecortando CMEMS para area comum...")
        lon_cmems_filtered = lon_cmems.sel(
            longitude=slice(common_area['lon_min'], common_area['lon_max'])
        )
        lat_cmems_filtered = lat_cmems.sel(
            latitude=slice(common_area['lat_min'], common_area['lat_max'])
        )
        print(f"Grade CMEMS apos recorte: {len(lon_cmems_filtered)} x {len(lat_cmems_filtered)} pontos")
    else:
        lon_cmems_filtered = lon_cmems
        lat_cmems_filtered = lat_cmems
    
    # Debug: valores originais
    print(f"\nDEBUG - ANFC antes da degradacao:")
    print(f"  u_anfc: min={np.nanmin(u_anfc.values):.4f}, max={np.nanmax(u_anfc.values):.4f}, NaNs={np.isnan(u_anfc.values).sum()}")
    print(f"  v_anfc: min={np.nanmin(v_anfc.values):.4f}, max={np.nanmax(v_anfc.values):.4f}, NaNs={np.isnan(v_anfc.values).sum()}")
    
    # Interpolar ANFC para grade CMEMS (degradacao)
    u_anfc_degraded = u_anfc.interp(longitude=lon_cmems_filtered, latitude=lat_cmems_filtered, method='linear')
    v_anfc_degraded = v_anfc.interp(longitude=lon_cmems_filtered, latitude=lat_cmems_filtered, method='linear')
    
    # Debug: valores apos degradacao
    print(f"\nDEBUG - ANFC apos degradacao para 0.25 deg:")
    print(f"  u_anfc_degraded: min={np.nanmin(u_anfc_degraded.values):.4f}, max={np.nanmax(u_anfc_degraded.values):.4f}, NaNs={np.isnan(u_anfc_degraded.values).sum()}")
    print(f"  v_anfc_degraded: min={np.nanmin(v_anfc_degraded.values):.4f}, max={np.nanmax(v_anfc_degraded.values):.4f}, NaNs={np.isnan(v_anfc_degraded.values).sum()} \n")
    
    print(f"Grade final ANFC degradado: {len(u_anfc_degraded.longitude)} x {len(u_anfc_degraded.latitude)} pontos\n")
    
    # Verificar se grades sao identicas agora
    if (np.array_equal(u_anfc_degraded.longitude.values, lon_cmems_filtered.values) and 
        np.array_equal(u_anfc_degraded.latitude.values, lat_cmems_filtered.values)):
        print("Grades idênticas")
    else:
        print("AVISO: Grades ainda nao sao identicas")
        print(f"  Diferenca lon: {np.abs(u_anfc_degraded.longitude.values - lon_cmems_filtered.values).max():.6f} deg")
        print(f"  Diferenca lat: {np.abs(u_anfc_degraded.latitude.values - lat_cmems_filtered.values).max():.6f} deg")
    
    return u_anfc_degraded, v_anfc_degraded, lon_cmems_filtered, lat_cmems_filtered

def calculate_metrics(u_anfc, v_anfc, u_cmems, v_cmems):
    """Calcula métricas de erro entre ANFC e CMEMS"""
    
    # Debug: Valores antes do cálculo
    print(f"\nDEBUG - Valores antes de calcular diferenças:")
    print(f"  u_anfc: min={np.nanmin(u_anfc.values):.4f}, max={np.nanmax(u_anfc.values):.4f}")
    print(f"  v_anfc: min={np.nanmin(v_anfc.values):.4f}, max={np.nanmax(v_anfc.values):.4f}")
    print(f"  u_cmems: min={np.nanmin(u_cmems.values):.4f}, max={np.nanmax(u_cmems.values):.4f}")
    print(f"  v_cmems: min={np.nanmin(v_cmems.values):.4f}, max={np.nanmax(v_cmems.values):.4f}")
    
    diff_u = u_anfc - u_cmems
    diff_v = v_anfc - v_cmems
    
    # Debug: Diferenças
    print(f"\nDEBUG - Diferenças:")
    print(f"  diff_u: min={np.nanmin(diff_u.values):.4f}, max={np.nanmax(diff_u.values):.4f}, NaNs={np.isnan(diff_u.values).sum()}")
    print(f"  diff_v: min={np.nanmin(diff_v.values):.4f}, max={np.nanmax(diff_v.values):.4f}, NaNs={np.isnan(diff_v.values).sum()}")
    
    if np.isnan(diff_u.values).sum() > len(diff_u.values) * 0.1:
        print(f"AVISO: Mais de 10% dos valores diff_u são NaN!")
    if np.isnan(diff_v.values).sum() > len(diff_v.values) * 0.1:
        print(f"AVISO: Mais de 10% dos valores diff_v são NaN!")
    
    # RMSE por componente
    rmse_u = float(np.sqrt(np.nanmean((diff_u.values) ** 2)))
    rmse_v = float(np.sqrt(np.nanmean((diff_v.values) ** 2)))
    
    # Magnitude dos vetores
    mag_anfc = np.sqrt(u_anfc.values**2 + v_anfc.values**2)
    mag_cmems = np.sqrt(u_cmems.values**2 + v_cmems.values**2)
    mag_diff = np.sqrt(diff_u.values**2 + diff_v.values**2)
    
    # Métricas vetoriais
    rmse_vector = float(np.sqrt(np.nanmean(mag_diff**2)))
    mean_vector_error = float(np.nanmean(mag_diff))
    max_vector_error = float(np.nanmax(mag_diff))
    mean_mag_anfc = float(np.nanmean(mag_anfc))
    
    # Erro relativo
    relative_error_pct = (mean_vector_error / mean_mag_anfc * 100) if mean_mag_anfc > 0 else np.nan
    
    print(f"\nDEBUG - Magnitudes:")
    print(f"  mag_anfc: min={np.nanmin(mag_anfc):.4f}, max={np.nanmax(mag_anfc):.4f}, média={mean_mag_anfc:.4f}")
    print(f"  mag_cmems: min={np.nanmin(mag_cmems):.4f}, max={np.nanmax(mag_cmems):.4f}, média={np.nanmean(mag_cmems):.4f}")
    print(f"  mag_diff: min={np.nanmin(mag_diff):.4f}, max={np.nanmax(mag_diff):.4f}")
    
    return {
        'rmse_u': rmse_u,
        'rmse_v': rmse_v,
        'rmse_vector': rmse_vector,
        'mean_vector_error': mean_vector_error,
        'max_vector_error': max_vector_error,
        'mean_mag_anfc': mean_mag_anfc,
        'relative_error_pct': relative_error_pct,
        'diff_u': diff_u,
        'diff_v': diff_v,
        'mag_anfc': mag_anfc,
        'mag_cmems': mag_cmems,
    }

def plot_comparison(u_anfc, v_anfc, u_my, v_my, diff_u, diff_v,
                    lon, lat, datetime_str):
    """Plota comparação ANFC vs CMEMS MY"""
    lon2d, lat2d = np.meshgrid(lon.values, lat.values)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             subplot_kw={'projection': crt.PlateCarree()})
    
    skip = (slice(None, None, 2), slice(None, None, 2))
    
    # ANFC degradado
    ax = axes[0]
    ax.coastlines()
    ax.quiver(lon2d[skip], lat2d[skip], u_anfc.values[skip], v_anfc.values[skip],
              color="red", angles="xy", scale_units="xy", scale=1.0, width=0.004,
              transform=crt.PlateCarree())
    ax.gridlines(draw_labels=True, alpha=0.4, linestyle='--')
    ax.set_title('ANFC Degradado (0.083 para 0.25 deg)', fontsize=12, weight='bold')
    
    # CMEMS MY
    ax = axes[1]
    ax.coastlines()
    ax.quiver(lon2d[skip], lat2d[skip], u_my.values[skip], v_my.values[skip],
              color="blue", angles="xy", scale_units="xy", scale=1.0, width=0.004,
              transform=crt.PlateCarree())
    ax.gridlines(draw_labels=True, alpha=0.4, linestyle='--')
    ax.set_title('CMEMS MY (0.25 deg)', fontsize=12, weight='bold')
    
    # Diferença ANFC - MY
    ax = axes[2]
    ax.coastlines()
    ax.quiver(lon2d[skip], lat2d[skip], diff_u.values[skip], diff_v.values[skip],
              color="green", angles="xy", scale_units="xy", scale=1.0, width=0.004,
              transform=crt.PlateCarree())
    ax.gridlines(draw_labels=True, alpha=0.4, linestyle='--')
    ax.set_title('Diferenca (ANFC - MY)', fontsize=12, weight='bold')
    
    datetime_display = datetime_str.replace("T", " ")
    plt.suptitle(f'Comparacao: ANFC (degradado para 0.25 deg) vs CMEMS MY ({datetime_display})',
                 fontsize=14, weight='bold')
    plt.tight_layout()
    
    return fig

def plot_original_resolutions(u_anfc_orig, v_anfc_orig, lon_anfc, lat_anfc,
                               u_my_orig, v_my_orig, lon_my, lat_my,
                               datetime_str):
    """
    Plota ANFC e CMEMS MY nas suas RESOLUCOES ORIGINAIS
    ANFC: 0.083 deg (16x23 pontos) - alta resolucao
    CMEMS: 0.25 deg (5x8 pontos) - baixa resolucao
    Sem interpolacao ou degradacao - mostra os dados brutos
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             subplot_kw={'projection': crt.PlateCarree()})
    
    # ANFC Original (alta resolução)
    ax = axes[0]
    ax.coastlines()
    lon2d_anfc, lat2d_anfc = np.meshgrid(lon_anfc.values, lat_anfc.values)
    # Mostrar todos os pontos ANFC (sem skip)
    ax.quiver(lon2d_anfc, lat2d_anfc, u_anfc_orig.values, v_anfc_orig.values,
              color="red", angles="xy", scale_units="xy", scale=1.0, width=0.003,
              transform=crt.PlateCarree(), alpha=0.8)
    ax.gridlines(draw_labels=True, alpha=0.4, linestyle='--')
    ax.set_title(f'ANFC Original (0.083 deg)\n{len(lon_anfc)}x{len(lat_anfc)} = {len(lon_anfc)*len(lat_anfc)} pontos', 
                 fontsize=12, weight='bold', color='red')
    ax.set_extent([lon_anfc.values.min()-0.1, lon_anfc.values.max()+0.1,
                   lat_anfc.values.min()-0.1, lat_anfc.values.max()+0.1],
                  crs=crt.PlateCarree())
    
    # CMEMS MY Original (baixa resolução)
    ax = axes[1]
    ax.coastlines()
    lon2d_my, lat2d_my = np.meshgrid(lon_my.values, lat_my.values)
    # Mostrar todos os pontos CMEMS
    ax.quiver(lon2d_my, lat2d_my, u_my_orig.values, v_my_orig.values,
              color="blue", angles="xy", scale_units="xy", scale=1.0, width=0.005,
              transform=crt.PlateCarree(), alpha=0.8)
    ax.gridlines(draw_labels=True, alpha=0.4, linestyle='--')
    ax.set_title(f'CMEMS MY Original (0.25 deg)\n{len(lon_my)}x{len(lat_my)} = {len(lon_my)*len(lat_my)} pontos', 
                 fontsize=12, weight='bold', color='blue')
    ax.set_extent([lon_my.values.min()-0.1, lon_my.values.max()+0.1,
                   lat_my.values.min()-0.1, lat_my.values.max()+0.1],
                  crs=crt.PlateCarree())
    
    datetime_display = datetime_str.replace("T", " ")
    plt.suptitle(f'Resolucoes Originais - Sem Interpolacao ({datetime_display})',
                 fontsize=14, weight='bold')
    plt.tight_layout()
    
    return fig

def print_metrics(metrics_my):
    """Imprime métricas de erro ANFC vs CMEMS MY"""
    print("\n" + "=" * 80)
    print("MÉTRICAS DE ERRO: ANFC VS CMEMS MY")
    print("=" * 80)
    print(f"\nRMSE u:              {metrics_my['rmse_u']:.6f} m/s")
    print(f"RMSE v:              {metrics_my['rmse_v']:.6f} m/s")
    print(f"RMSE vetorial:       {metrics_my['rmse_vector']:.6f} m/s")
    print(f"Erro médio vetorial: {metrics_my['mean_vector_error']:.6f} m/s")
    print(f"Erro máximo vetorial: {metrics_my['max_vector_error']:.6f} m/s")
    print(f"\nMagnitude média ANFC: {metrics_my['mean_mag_anfc']:.6f} m/s")
    print(f"Erro relativo:       {metrics_my['relative_error_pct']:.2f}%")
    print("=" * 80)

def main():
    """Função principal"""
    
    # Paths
    anfc_path = get_data_path('aguaANFC.nc')
    cmems_my_path = get_data_path('aguaMultiYear.nc')
    
    datetime_str = "2025-04-05T00:00:00"
    
    print("\n" + "=" * 80)
    print("ANALISE: ANFC (0.083 degradado para 0.25 deg) vs CMEMS MY (0.25 deg)")
    print("ESTRATEGIA: Degradar ANFC para evitar erros de interpolacao")
    print("=" * 80)
    print(f"Timestamp: {datetime_str}")
    
    # 1. Carregar dados
    anfc, cmems_my, anfc_slice, cmems_my_slice = \
        load_data(anfc_path, cmems_my_path, datetime_str)
    
    if anfc is None:
        return
    
    # 2. Extrair componentes
    u_anfc, v_anfc, lon_anfc, lat_anfc = extract_components(anfc_slice)
    u_my, v_my, lon_my, lat_my = extract_components(cmems_my_slice)
    
    # 3. Garantir dimensões corretas
    u_anfc, v_anfc = ensure_dimensions(u_anfc, v_anfc)
    u_my, v_my = ensure_dimensions(u_my, v_my)
    
    # 4. Encontrar area geografica comum
    common_area = find_common_geographic_area(lon_anfc, lat_anfc, lon_my, lat_my)
    if common_area is None:
        print("ERRO: Nao e possivel comparar - areas geograficas nao se sobrepoem!")
        return
    
    # 5. Degradar ANFC de 0.083 deg para 0.25 deg (resolucao CMEMS) na area comum
    u_anfc_degraded, v_anfc_degraded, lon_my_filtered, lat_my_filtered = degrade_anfc_to_cmems_resolution(
        u_anfc, v_anfc, lon_anfc, lat_anfc, lon_my, lat_my, common_area)
    
    # 6. Recortar CMEMS para a mesma area comum
    u_my_filtered = u_my.sel(
        longitude=slice(common_area['lon_min'], common_area['lon_max']),
        latitude=slice(common_area['lat_min'], common_area['lat_max'])
    )
    v_my_filtered = v_my.sel(
        longitude=slice(common_area['lon_min'], common_area['lon_max']),
        latitude=slice(common_area['lat_min'], common_area['lat_max'])
    )
    print(f"Grade CMEMS MY apos recorte: {len(u_my_filtered.longitude)} x {len(u_my_filtered.latitude)} pontos")
    
    # 7. Calcular métricas (grades idênticas NA MESMA AREA, sem interpolação!)
    metrics_my = calculate_metrics(u_anfc_degraded, v_anfc_degraded, u_my_filtered, v_my_filtered)
    
    # 8. Imprimir métricas
    print_metrics(metrics_my)
    
    # 9. Plotar comparação com resolução degradada
    fig1 = plot_comparison(u_anfc_degraded, v_anfc_degraded, u_my_filtered, v_my_filtered,
                          metrics_my['diff_u'], metrics_my['diff_v'],
                          lon_my_filtered, lat_my_filtered, datetime_str)
    
    # 10. Plotar resoluções originais lado a lado (sem degradação)
    print("\n" + "=" * 80)
    print("PLOTANDO RESOLUCOES ORIGINAIS (SEM INTERPOLACAO)")
    print("=" * 80)
    print(f"ANFC:     {len(lon_anfc)} x {len(lat_anfc)} = {len(lon_anfc)*len(lat_anfc)} pontos (0.083 deg)")
    print(f"CMEMS MY: {len(lon_my)} x {len(lat_my)} = {len(lon_my)*len(lat_my)} pontos (0.25 deg)")
    
    fig2 = plot_original_resolutions(u_anfc, v_anfc, lon_anfc, lat_anfc,
                                     u_my, v_my, lon_my, lat_my,
                                     datetime_str)
    
    plt.show()
    
    # Fechar datasets
    anfc.close()
    cmems_my.close()

if __name__ == "__main__":
    main()
