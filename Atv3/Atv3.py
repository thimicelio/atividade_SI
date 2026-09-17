from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PASTA_DO_PROGRAMA = Path(__file__).resolve().parent
ARQUIVO_CSV = PASTA_DO_PROGRAMA / "clientes_ecommerce.csv"
SEMENTE = 42
NUM_INICIALIZACOES = 30
MAX_ITERACOES = 300
TOLERANCIA = 1e-6


def padronizar(dados):
    medias = dados.mean(axis=0)
    desvios = dados.std(axis=0)
    if np.any(desvios == 0):
        raise ValueError("Existe uma coluna com desvio-padrao igual a zero.")
    return (dados - medias) / desvios, medias, desvios


def inicializar_kmeans_mais_mais(dados, k, gerador):
    n_amostras = dados.shape[0]
    centroides = np.empty((k, dados.shape[1]), dtype=float)

    primeiro_indice = gerador.integers(n_amostras)
    centroides[0] = dados[primeiro_indice]

    distancias_minimas_quadrado = np.sum(
        (dados - centroides[0]) ** 2, axis=1
    )

    for indice_centroide in range(1, k):
        soma_distancias = distancias_minimas_quadrado.sum()

        if soma_distancias == 0:
            proximo_indice = gerador.integers(n_amostras)
        else:
            probabilidades = distancias_minimas_quadrado / soma_distancias
            proximo_indice = gerador.choice(n_amostras, p=probabilidades)

        centroides[indice_centroide] = dados[proximo_indice]

        novas_distancias_quadrado = np.sum(
            (dados - centroides[indice_centroide]) ** 2, axis=1
        )
        distancias_minimas_quadrado = np.minimum(
            distancias_minimas_quadrado,
            novas_distancias_quadrado,
        )

    return centroides


def atribuir_clusters(dados, centroides):
    distancias_quadrado = np.sum(
        (dados[:, np.newaxis, :] - centroides[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    rotulos = np.argmin(distancias_quadrado, axis=1)
    return rotulos, distancias_quadrado


def executar_kmeans(
    dados,
    k,
    semente,
    max_iteracoes=MAX_ITERACOES,
    tolerancia=TOLERANCIA,
):
    gerador = np.random.default_rng(semente)
    centroides = inicializar_kmeans_mais_mais(dados, k, gerador)

    for iteracao in range(1, max_iteracoes + 1):
        rotulos, distancias_quadrado = atribuir_clusters(dados, centroides)
        novos_centroides = np.empty_like(centroides)

        for cluster in range(k):
            pontos_cluster = dados[rotulos == cluster]

            if len(pontos_cluster) == 0:
                # Se um cluster ficar vazio, usa o ponto mais distante de seu
                # centroide atual como novo centroide.
                distancia_ao_mais_proximo = np.min(
                    distancias_quadrado, axis=1
                )
                indice_mais_distante = np.argmax(distancia_ao_mais_proximo)
                novos_centroides[cluster] = dados[indice_mais_distante]
            else:
                novos_centroides[cluster] = pontos_cluster.mean(axis=0)

        deslocamento = np.max(
            np.linalg.norm(novos_centroides - centroides, axis=1)
        )
        centroides = novos_centroides

        if deslocamento < tolerancia:
            break

    rotulos, distancias_quadrado = atribuir_clusters(dados, centroides)
    wcss = np.sum(distancias_quadrado[np.arange(len(dados)), rotulos])

    return centroides, rotulos, float(wcss), iteracao


def melhor_de_varias_execucoes(dados, k, n_execucoes, semente_base):
    melhor_resultado = None

    for repeticao in range(n_execucoes):
        resultado = executar_kmeans(
            dados,
            k,
            semente=semente_base + repeticao,
        )

        if melhor_resultado is None or resultado[2] < melhor_resultado[2]:
            melhor_resultado = resultado

    return melhor_resultado


def main():
    tabela = pd.read_csv(ARQUIVO_CSV)
    colunas_analise = ["idade", "gasto_anual_mil_reais"]

    if tabela[colunas_analise].isnull().any().any():
        raise ValueError("O arquivo possui valores ausentes nas colunas analisadas.")

    # cliente_id e apenas um identificador e nao entra no calculo de distancia.
    dados_originais = tabela[colunas_analise].to_numpy(dtype=float)
    dados_padronizados, medias, desvios = padronizar(dados_originais)

    valores_k = list(range(1, 11))
    valores_wcss = []

    for k in valores_k:
        _, _, wcss, _ = melhor_de_varias_execucoes(
            dados_padronizados,
            k,
            NUM_INICIALIZACOES,
            SEMENTE + k * 1000,
        )
        valores_wcss.append(wcss)
        print(f"k={k:2d} | menor WCSS={wcss:.6f}")

    plt.figure(figsize=(8, 5))
    plt.plot(valores_k, valores_wcss, marker="o")
    plt.xticks(valores_k)
    plt.xlabel("Numero de clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Metodo do cotovelo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PASTA_DO_PROGRAMA / "metodo_cotovelo.png", dpi=150)
    plt.close()

    # Definido depois da observacao do grafico do metodo do cotovelo.
    k_escolhido = 4
    centroides_padronizados, rotulos, wcss_final, iteracoes = (
        melhor_de_varias_execucoes(
            dados_padronizados,
            k_escolhido,
            NUM_INICIALIZACOES,
            SEMENTE + 9999,
        )
    )

    centroides_originais = centroides_padronizados * desvios + medias
    tabela["cluster"] = rotulos

    resumo = (
        tabela.groupby("cluster")
        .agg(
            quantidade_clientes=("cliente_id", "count"),
            idade_media=("idade", "mean"),
            gasto_medio_mil_reais=("gasto_anual_mil_reais", "mean"),
        )
        .sort_values(["idade_media", "gasto_medio_mil_reais"])
    )

    print(f"\nk escolhido: {k_escolhido}")
    print(f"WCSS final: {wcss_final:.6f}")
    print(f"Iteracoes da melhor execucao: {iteracoes}")
    print("\nCentroides na escala original:")
    for cluster, centroide in enumerate(centroides_originais):
        print(
            f"Cluster {cluster}: idade={centroide[0]:.2f}, "
            f"gasto={centroide[1]:.2f} mil reais"
        )
    print("\nResumo dos clusters:")
    print(resumo.to_string(float_format=lambda valor: f"{valor:.2f}"))

    plt.figure(figsize=(9, 6))
    cores = plt.get_cmap("tab10", k_escolhido)

    for cluster in range(k_escolhido):
        mascara = rotulos == cluster
        plt.scatter(
            dados_originais[mascara, 0],
            dados_originais[mascara, 1],
            color=cores(cluster),
            label=f"Cluster {cluster}",
            alpha=0.75,
            s=38,
        )

    plt.scatter(
        centroides_originais[:, 0],
        centroides_originais[:, 1],
        color="black",
        marker="X",
        s=220,
        label="Centroides",
        edgecolors="white",
        linewidths=1.2,
    )
    plt.xlabel("Idade")
    plt.ylabel("Gasto anual (mil reais)")
    plt.title(f"Clusterizacao final com k={k_escolhido}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PASTA_DO_PROGRAMA / "clusters_clientes.png", dpi=150)
    plt.close()

    tabela.to_csv(
        PASTA_DO_PROGRAMA / "clientes_ecommerce_clusterizados.csv",
        index=False,
    )
    resumo.to_csv(PASTA_DO_PROGRAMA / "resumo_clusters.csv")

    print("\nArquivos gerados:")
    print("- metodo_cotovelo.png")
    print("- clusters_clientes.png")
    print("- clientes_ecommerce_clusterizados.csv")
    print("- resumo_clusters.csv")


if __name__ == "__main__":
    main()
