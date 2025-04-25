"""
metrics.py

Este script calcula as métricas NIQE, PIQE e MCMA para um ou mais pares de imagens,
permitindo comparar a qualidade de saída do seu algoritmo AHEVPC com a do artigo original.

Uso:
    python metrics.py
"""

import os
import pandas as pd
import cv2
import numpy as np
import niqe
import pypiqe

def resize_proportional(img, max_dim=800):
    h, w = img.shape[:2]
    # calcula o fator de escala para que a maior dimensão vire max_dim
    scale = max_dim / max(h, w)
    # se a imagem já é menor, não escala
    if scale >= 1.0:
        return img.copy()
    new_w = int(w * scale)
    new_h = int(h * scale)
    # INTER_AREA é recomendado para encolher
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


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

    # img_bgr = resize_proportional(img_bgr, max_dim=400)
    # cv2.imshow("img_bgr", img_bgr)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    if img_bgr is None:
        raise FileNotFoundError(f"Não foi possível ler: {path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.uint8)

    # NIQE e PIQE (menor é melhor)
    niqe_score = niqe.niqe(img)
    # niqe_score = nnniqe.niqe(img)
    piqe_score = pypiqe.piqe(img)
    piqe_score, activityMask, noticeableArtifactMask, noiseMask = pypiqe.piqe(img)

    # niqe_score = 0
    # piqe_score = 0

    # MCMA (maior é melhor)
    mcma_score = mcma(img)

    return {
        "NIQE": niqe_score,
        "PIQE": piqe_score,
        "MCMA": mcma_score
    }

def main():

    path = os.path.abspath(__file__)
    print(path)
    
    files = [
        os.path.join(root, file)
        for root, dirs, files_list in os.walk(path.replace("metrics.py", "image_results"))
        for file in files_list if "jpg_enhanced.png" in file or "jpg_gray.png" in file
    ]


    results = {}
    for f in files:
        key = f.split("image_results/")[-1][:5]
        if key not in results:
            results[key] = {
                "path_gray": None,
                "path_enhanced": None,
                "NIQE_original": 0,
                "NIQE_enhanced": 0,
                "PIQE_original": 0,
                "PIQE_enhanced": 0,
                "MCMA_original": 0,
                "MCMA_enhanced": 0,
            }
        if f.endswith("jpg_gray.png"):
            results[key]["path_gray"] = f
        elif f.endswith("jpg_enhanced.png"):
            results[key]["path_enhanced"] = f

    # 3) processa cada par de imagens e atualiza métricas
    for key, info in results.items():
        gray_path     = info["path_gray"]
        enh_path      = info["path_enhanced"]
        metrics_gray  = compute_metrics(gray_path)
        metrics_enh   = compute_metrics(enh_path)
        for m in ("NIQE", "PIQE", "MCMA"):
            results[key][f"{m}_original"] = metrics_gray[m]
            results[key][f"{m}_enhanced"] = metrics_enh[m]

    # 4) monta o DataFrame
    df = pd.DataFrame.from_dict(results, orient="index")
    df.drop(columns=["path_gray", "path_enhanced"], inplace=True)
    print(df)

    df.to_csv("results.csv")

if __name__ == "__main__":
    main()

