import os, zipfile, hashlib, unicodedata
from pathlib import Path

def listar_zips(path):
    return [nome for nome in os.listdir(path) if nome.lower().endswith(".zip")]


def listar_pastas(path):
    return [nome for nome in os.listdir(path) if (Path(path) / nome).is_dir()]


def para_caminho_longo_windows(path):
    path_str = str(Path(path).resolve())
    if os.name != "nt":
        return path_str
    if path_str.startswith("\\\\?\\"):
        return path_str
    if path_str.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path_str[2:]
    return "\\\\?\\" + path_str


def nome_destino_curto(caminho_zip):
    stem = Path(caminho_zip).stem
    prefixo = stem[:40].rstrip()
    sufixo = hashlib.md5(stem.encode("utf-8")).hexdigest()[:8]
    return f"{prefixo}_{sufixo}"


def chave_flexivel(texto):
    texto_norm = unicodedata.normalize("NFKD", texto)
    sem_acentos = "".join(ch for ch in texto_norm if not unicodedata.combining(ch))
    return "".join(ch for ch in sem_acentos.lower() if ch.isalnum())


def resolver_caminho_entrada(path):
    p = Path(path)
    if p.exists():
        return p

    parent = p.parent
    alvo = p.name
    if not parent.exists() or not parent.is_dir():
        return p

    chave_alvo = chave_flexivel(alvo)
    for candidato in parent.iterdir():
        if chave_flexivel(candidato.name) == chave_alvo:
            return candidato

    return p


def extrair_zip_no_mesmo_nivel(path, nome_zip):
    caminho_zip = Path(path) / nome_zip
    pasta_destino = caminho_zip.parent / nome_destino_curto(caminho_zip)
    pasta_destino.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(para_caminho_longo_windows(caminho_zip), "r") as zip_ref:
        zip_ref.extractall(para_caminho_longo_windows(pasta_destino))

    print(f"Extraído: {nome_zip}")
    print(f"Destino: {pasta_destino}\n")


def menu_extracao(path):
    path = resolver_caminho_entrada(path)
    if not os.path.isdir(path):
        print(f"Caminho inválido: {path}")
        return

    atual = Path(path)
    raiz = Path(path)

    while True:
        pastas = listar_pastas(atual)
        arquivos = listar_zips(atual)

        print("\n=== MENU DE NAVEGAÇÃO E EXTRAÇÃO ===")
        print(f"Nível atual: {atual}")

        if pastas:
            print("\nPastas:")
            for i, nome in enumerate(pastas, start=1):
                print(f"P{i}: {nome}")
        else:
            print("\nPastas: nenhuma")

        if arquivos:
            print("\nZIPs:")
            for i, nome in enumerate(arquivos, start=1):
                print(f"Z{i}: {nome}")
        else:
            print("\nZIPs: nenhum")

        print("\n1 - Entrar em pasta (ID P)")
        print("2 - Voltar um nível")
        print("3 - Extrair todos os .zip deste nível")
        print("4 - Extrair um .zip por ID (Z)")
        print("0 - Sair")
        opcao = input("Escolha uma opção: ").strip()

        if opcao == "0":
            print("Encerrado.")
            break

        if opcao == "1":
            if not pastas:
                print("Nenhuma pasta para entrar neste nível.")
                continue

            try:
                index_pasta = int(input("Digite o ID da pasta (número após P): "))
            except ValueError:
                print("ID inválido.")
                continue

            if index_pasta < 1 or index_pasta > len(pastas):
                print("ID fora do intervalo.")
                continue

            atual = atual / pastas[index_pasta - 1]
            continue

        if opcao == "2":
            if atual == raiz:
                print("Você já está no diretório raiz.")
            else:
                atual = atual.parent
            continue

        if opcao == "3":
            if not arquivos:
                print("Nenhum arquivo .zip encontrado neste nível.")
                continue

            for nome_zip in arquivos:
                extrair_zip_no_mesmo_nivel(atual, nome_zip)
            print("Extração concluída.")

        elif opcao == "4":
            if not arquivos:
                print("Nenhum arquivo .zip encontrado neste nível.")
                continue

            try:
                index_extract = int(input("Digite o ID do zip (número após Z): "))
            except ValueError:
                print("ID inválido.")
                continue

            if index_extract < 1 or index_extract > len(arquivos):
                print("ID fora do intervalo.")
                continue

            extrair_zip_no_mesmo_nivel(atual, arquivos[index_extract - 1])

        else:
            print("Opção inválida.")


if __name__ == "__main__":
    caminho = input("Informe o diretório raiz: ").strip()
    menu_extracao(caminho)