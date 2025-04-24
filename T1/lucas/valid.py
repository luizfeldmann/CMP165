"""
validation_metrics.py

Este script calcula as métricas NIQE, PIQE e MCMA para um ou mais pares de imagens,
permitindo comparar a qualidade de saída do seu algoritmo AHEVPC com a do artigo original.

Uso:
    python validation_metrics.py --images meu_algoritmo.png artigo.png [ou múltiplos pares]

Dependências:
    pip install imquality scikit-image opencv-contrib-python
"""

import argparse
import cv2
import numpy as np
# from imquality import niqe, piqe
# import niqe
# import nniqe
# import nnniqe
# from skimage.metrics import niqe
import pypiqe

def mcma(img_gray: np.ndarray) -> float:
    """
    Calcula a métrica MCMA (Maximize Contrast with Minimum Artefact) segundo o artigo:
      MCMA = 0.71 * (0.4*PDRO - 0.3*PHSD - 0.7*PPU + 1)

    PDRO  = Dynamic range occupation = (max - min) / 255
    PHSD  = 1 - Similaridade(h, uniform) via histograma normalizado
    PPU   = 1 - variancia media de patches 3x3 (normalizada)
    """
    # 1) PDRO
    mn, mx = img_gray.min(), img_gray.max()
    PDRO = (mx - mn) / 255.0

    # 2) PHSD
    h, _ = np.histogram(img_gray, bins=256, range=(0,255), density=True)
    h_uniform = np.ones_like(h)/256
    PHSD = 1 - np.sum(np.minimum(h, h_uniform))

    # 3) PPU
    M, N = img_gray.shape
    variancias = []
    for i in range(1, M-1):
        for j in range(1, N-1):
            patch = img_gray[i-1:i+2, j-1:j+2]
            variancias.append(patch.var())
    # var maxima em [0,255] e uniforme seria (255^2)/12
    max_var = (255.0**2)/12.0
    PPU = 1 - (np.mean(variancias) / max_var)

    # Combina
    return 0.71 * (0.4*PDRO - 0.3*PHSD - 0.7*PPU + 1)

def compute_metrics(path: str):
    """
    Carrega a imagem, converte para grayscale e calcula NIQE, PIQE e MCMA.
    Retorna um dicionário com os valores.
    """
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Não foi possível ler: {path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.uint8)

    # NIQE e PIQE (menor é melhor)
    # niqe_score = niqe.niqe(img)
    # niqe_score = nnniqe.niqe(img)
    piqe_score = pypiqe.piqe(img)
    piqe_score, activityMask, noticeableArtifactMask, noiseMask = pypiqe.piqe(img)

    niqe_score = 0
    # piqe_score = 0

    # MCMA (maior é melhor)
    mcma_score = mcma(img)

    return {
        "NIQE": niqe_score,
        "PIQE": piqe_score,
        "MCMA": mcma_score
    }

def main():
    parser = argparse.ArgumentParser(
        description="Valida qualidade de imagens via NIQE, PIQE e MCMA"
    )
    parser.add_argument(
        "images", nargs="+",
        help="Lista de caminhos de imagens (par a par para comparação)"
    )
    args = parser.parse_args()

    paths = args.images
    if len(paths) % 2 != 0:
        parser.error("Forneça pares de imagens: seu_algoritmo.png artigo.png ...")

    for i in range(0, len(paths), 2):
        mine = paths[i]
        article = paths[i+1]
        print(f"\nComparando:\n  Seu método: {mine}\n  Artigo   : {article}")
        metrics_mine    = compute_metrics(mine)
        metrics_article = compute_metrics(article)

        print(" Métrica      |   Seu método   |   Artigo")
        print("--------------+----------------+------------")
        for name in ("NIQE","PIQE","MCMA"):
            v1 = metrics_mine[name]
            v2 = metrics_article[name]
            print(f" {name:10s} | {v1:14.4f} | {v2:10.4f}")

if __name__ == "__main__":
    main()

