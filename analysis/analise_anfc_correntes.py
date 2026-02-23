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
    ANFC será usado como referência (0.083°)
    CMEMS MY será interpolado para grade do ANFC
    """
    print("\n" + "=" * 80)
    print("CARREGANDO DADOS ANFC (0.083°) E CMEMS MY (0.25°)")
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
        print(f"Extraindo superfície (depth=0) de u: {u.dims} → ", end="")
        u = u.isel(depth=0, drop=True)
        print(f"{u.dims}")
    
    if 'depth' in v.dims:
        print(f"Extraindo superfície (depth=0) de v: {v.dims} → ", end="")
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

def align_to_anfc(u_cmems, v_cmems, lon_cmems, lat_cmems, lon_anfc, lat_anfc):
    """
    Interpola dados CMEMS (0.25°) para grade ANFC (0.083°)
    """
    print("\n" + "-" * 80)
    print("ALINHANDO GRADES")
    print("-" * 80)
    print(f"Grade CMEMS: {len(lon_cmems)} x {len(lat_cmems)} pontos")
    print(f"Grade ANFC:  {len(lon_anfc)} x {len(lat_anfc)} pontos")
    
    # Debug: Verificar valores antes da interpolação
    print(f"\n🔍 DEBUG - Antes da interpolação:")
    print(f"  u_cmems: min={np.nanmin(u_cmems.values):.4f}, max={np.nanmax(u_cmems.values):.4f}, NaNs={np.isnan(u_cmems.values).sum()}")
    print(f"  v_cmems: min={np.nanmin(v_cmems.values):.4f}, max={np.nanmax(v_cmems.values):.4f}, NaNs={np.isnan(v_cmems.values).sum()}")
    
    # Verificar cobertura geográfica
    print(f"\nCobertura CMEMS:")
    print(f"  lon: [{lon_cmems.values.min():.4f}, {lon_cmems.values.max():.4f}]")
    print(f"  lat: [{lat_cmems.values.min():.4f}, {lat_cmems.values.max():.4f}]")
    print(f"Cobertura ANFC:")
    print(f"  lon: [{lon_anfc.values.min():.4f}, {lon_anfc.values.max():.4f}]")
    print(f"  lat: [{lat_anfc.values.min():.4f}, {lat_anfc.values.max():.4f}]")
    
    # Verificar se coberturas são sobrepostas
    lon_overlap = (lon_anfc.values.min() >= lon_cmems.values.min() and 
                   lon_anfc.values.max() <= lon_cmems.values.max())
    lat_overlap = (lat_anfc.values.min() >= lat_cmems.values.min() and 
                   lat_anfc.values.max() <= lat_cmems.values.max())
    print(f"  Overlap lon: {lon_overlap}, lat: {lat_overlap}")
    if not (lon_overlap and lat_overlap):
        print(f"AVISO: Há extrapolação (dados fora da cobertura CMEMS)!")
    
    # Verificar se são iguais
    if not (np.array_equal(lon_cmems.values, lon_anfc.values) and 
            np.array_equal(lat_cmems.values, lat_anfc.values)):
        print("\nGrades diferentes - interpolando CMEMS para ANFC...")
        u_aligned = u_cmems.interp(longitude=lon_anfc, latitude=lat_anfc, method='linear')
        v_aligned = v_cmems.interp(longitude=lon_anfc, latitude=lat_anfc, method='linear')
        
        # Debug: Verificar valores após interpolação
        print(f"\nDEBUG - Após a interpolação:")
        print(f"  u_aligned: min={np.nanmin(u_aligned.values):.4f}, max={np.nanmax(u_aligned.values):.4f}, NaNs={np.isnan(u_aligned.values).sum()}")
        print(f"  v_aligned: min={np.nanmin(v_aligned.values):.4f}, max={np.nanmax(v_aligned.values):.4f}, NaNs={np.isnan(v_aligned.values).sum()}")
        
        if np.isnan(u_aligned.values).sum() > len(u_aligned.values) * 0.5:
            print(f"AVISO: Mais de 50% dos valores são NaN após interpolação!")
        
        print("CMEMS interpolado com sucesso para grade ANFC (0.083°)")
    else:
        print("\nGrades já estão alinhadas")
        u_aligned = u_cmems
        v_aligned = v_cmems
    
    return u_aligned, v_aligned

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
    
    # ANFC
    ax = axes[0]
    ax.coastlines()
    ax.quiver(lon2d[skip], lat2d[skip], u_anfc.values[skip], v_anfc.values[skip],
              color="red", angles="xy", scale_units="xy", scale=1.0, width=0.004,
              transform=crt.PlateCarree())
    ax.gridlines(draw_labels=True, alpha=0.4, linestyle='--')
    ax.set_title('ANFC (0.083°)', fontsize=12, weight='bold')
    
    # CMEMS MY (interpolado)
    ax = axes[1]
    ax.coastlines()
    ax.quiver(lon2d[skip], lat2d[skip], u_my.values[skip], v_my.values[skip],
              color="blue", angles="xy", scale_units="xy", scale=1.0, width=0.004,
              transform=crt.PlateCarree())
    ax.gridlines(draw_labels=True, alpha=0.4, linestyle='--')
    ax.set_title('CMEMS MY (0.25° → 0.083°)', fontsize=12, weight='bold')
    
    # Diferença ANFC - MY
    ax = axes[2]
    ax.coastlines()
    ax.quiver(lon2d[skip], lat2d[skip], diff_u.values[skip], diff_v.values[skip],
              color="green", angles="xy", scale_units="xy", scale=1.0, width=0.004,
              transform=crt.PlateCarree())
    ax.gridlines(draw_labels=True, alpha=0.4, linestyle='--')
    ax.set_title('Diferença (ANFC - MY)', fontsize=12, weight='bold')
    
    datetime_display = datetime_str.replace("T", " ")
    plt.suptitle(f'Comparação de Correntes: ANFC vs CMEMS MY ({datetime_display})',
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
    print("ANÁLISE: ANFC (0.083°) vs CMEMS MY (0.25°)")
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
    
    # 4. Alinhar CMEMS MY para grade ANFC
    u_my_aligned, v_my_aligned = align_to_anfc(u_my, v_my, lon_my, lat_my, 
                                                 lon_anfc, lat_anfc)
    
    # 5. Calcular métricas
    metrics_my = calculate_metrics(u_anfc, v_anfc, u_my_aligned, v_my_aligned)
    
    # 6. Imprimir métricas
    print_metrics(metrics_my)
    
    # 7. Plotar
    fig = plot_comparison(u_anfc, v_anfc, u_my_aligned, v_my_aligned,
                          metrics_my['diff_u'], metrics_my['diff_v'],
                          lon_anfc, lat_anfc, datetime_str)
    
    plt.show()
    
    # Fechar datasets
    anfc.close()
    cmems_my.close()

if __name__ == "__main__":
    main()
