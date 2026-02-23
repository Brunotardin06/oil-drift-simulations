import xarray as xr

FILE_PATH = r"c:\Users\prmorais\Documents\GitHub\oil-drift-simulations\analysis\water_042025.nc"

def main():
    ds = xr.open_dataset(FILE_PATH)

    print("=== RESUMO DO DATASET ===")
    print(ds)

    print("\n=== ATRIBUTOS GLOBAIS ===")
    for k, v in ds.attrs.items():
        print(f"{k}: {v}")

    print("\n=== VARIÁVEIS ===")
    for name, da in ds.data_vars.items():
        print(f"\n-- {name} --")
        print(f"dims: {da.dims}")
        print(f"shape: {da.shape}")
        print(f"attrs: {da.attrs}")
        try:
            stats = da.load().to_numpy()
            print(f"min: {float(stats.min())} | max: {float(stats.max())}")
        except Exception as e:
            print(f"estatísticas indisponíveis: {e}")

    print("\n=== COORDENADAS ===")
    for name, da in ds.coords.items():
        print(f"{name}: dims={da.dims}, shape={da.shape}, attrs={da.attrs}")

if __name__ == "__main__":
    main()