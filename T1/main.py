""" Trabalho 1 - CPM165
Adaptive histogram equalization with visual perception consistency

Imagem a0317-IMG_0647.jpg corresponde a Marsh

Algoritmo:

1. Entrada de uma imagem e obtenção de seu histograma normalizado pela média hist usando a Fórmula (1).
2. Encontrar o melhor parâmetro de correção global y1 para hist usando as Fórmulas (2) e (3), 
obtendo assim o histograma ajustado hist1 pela Fórmula (4).
3. Calcular o coeficiente de correção gama secundário y2 usando a Fórmula (6).
4. Ajustar os dados pequenos em hist1 para obter hist2 pela Fórmula (5).
5. Calcular os vieses de brilho 𝛿1, 𝛿2 e o fator de ponderação ω pelas Fórmulas (9), (10) e (11).
6. Obter o histograma de saída do modelo baseado em CDF hist3 pelas Fórmulas (7) e (8).
7. Calcular o limite de truncamento Th usando a Fórmula (13) e obter o histograma truncado hist4 pela Fórmula (14).
8. Equalizar hist4 para obter a imagem aprimorada.
"""


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def step1_normalized_histogram(image) -> tuple:
    """
    Passo 1 (Fórmula 1): hist = h x L
    - h: histograma de probabilidade (cada bin dividido pelo total de pixels)
    - L: alcance dinâmico do nível de cinza (normalmente 255) - variação do valor de cinza

    No Texto:
        - h: probability distribution histogram h is calculated by scaling down 
            the vertical axis of the histogram in proportion to
            the total number of pixels.
        - L: the gray value variation range.
            
    Retornamos:
        gray: imagem em tons de cinza
        hist: histograma normalizado multiplicado por L
        Im: brilho médio da imagem
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h_raw, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256]) # quantos pixels de cada nível de cinza (0–255)
    h = h_raw / h_raw.sum() # Normaliza para obter probabilidades h (E h = 1)
    L = 255 # alcance dos níveis de cinza, que em imagens de 8 bits vai de 0 a 255
    hist = h * L # Fórmula 1
    Im = np.mean(gray) # Brilho médio - média simples dos pixels em cinza
    
    return gray, hist, Im

def step2_gamma1_adjustment(hist, Im):
    """
    Passo 2 - Fórmulas (2), (3) e (4):
        (2) erro_abs = argmin_y |CDF(Im) - lambda|     s.t. y ∈ [0,1]
        (3) lambda = I_t / 255
        (4) hist1(k) = hist(k)^y_1
    Retorna:
        best_gamma: y_1 que minimiza erro_abs
        best_hist1: hist1 ajustado pelo y_1 ótimo
    """

    # Fórmula 3: lambda = I_t / 255
    lambda_ = Im / 255

    # Iniciar a busca
    # min_diff = armazenar o erro absoluto
    # best_gamma o y_1 correspondente
    # best_hist1 o histograma resultante

    min_diff = float('inf')
    best_gamma = 0.0
    best_hist1 = hist.copy()

    # Varre y E [0,1]
    # for gamma in np.linspace(0.0, 1.0, 100):
    for gamma in np.linspace(0.5, 1.0, 100):

        # Fórmula 4 hist1(k) = hist(k)^γ
        hist1 = hist ** gamma
        hist1 = hist1 / hist1.sum() # normaliza para soma = 1 - para que seja distribuicao de probabilidade
        cdf = np.cumsum(hist1) # # CDF(hist1, Im) = E_{i=0}^{Im} hist1(i)
        diff = abs(cdf[int(Im)] - lambda_) # Fórmula 2  erro = |CDF(Im) − lambda|

        # atualiza o melhor y se este diff for menor que o erro_abs encontrado até agora
        if diff < min_diff:
            min_diff = diff
            best_gamma = gamma
            best_hist1 = hist1
    # print(f"{best_gamma=}")

    # devolve y_1 ótimo e o histograma ajustado correspondente
    return best_gamma, best_hist1


def step3_gamma2(gamma1):
    """
    Passo 3 - Fórmula 6:
        y2 = a x (1 - y1)
        onde a (peso) = 1 por padrão no artigo.
    
    No texto:
        - (a) where Î ± is the weight coefficient and the default value is 1
    Retorna:
        y2: coeficiente de correção secundário para "small data"
    """
    
    # y2 = a x (1 − y1)
    # aqui a = 1 (valor default do artigo), logo y2 = 1 * (1 − y1)
    
    return 1 * (1 - gamma1)

def step4_hist2_small_values(hist1: np.ndarray, gamma2: float) -> np.ndarray:
    """
    Passo 4 Fórmula 5 - Detail protection:
        hist2(k) = {
            hist1(k)    , if hist1(k) >= 1
            hist1(k)^g2 , if hist1(k) <  1
        }

    No Texto (secao 3.1.2 Detail protection):
        A correcao secundaria de gamma e aplicada apenas aos valores
        de histograma de baixa probabilidade (small data).

    Retorna:
        hist2: vetor de histograma com small data corrigidos por gamma2
    """

    # copia para nao alterar o original
    hist2 = hist1.copy()

    # máscara: True para todos os bins cujo valor é < 1.0
    mask = hist1 < 1.0

    # aplica gamma2 apenas aos small data
    hist2[mask] = hist1[mask] ** gamma2

    # renormaliza para garantir soma igual a 1
    hist2 = hist2 / hist2.sum()

    return hist2

def step5_bias_and_weights(hist, hist2, Im, It=127):
    """
    Passo 5 - Formulas 9, 10 e 11:

        histl  = sum(hist2[0 : Im_index])
        histt  = sum(hist2)                # total do histograma ajustado (deveria ser L = 255)
        kl     = Im_index                  # numero de bins abaixo de Im
        kr     = 256 - Im_index            # numero de bins acima de Im

        Formula  9: delta1 = (It * histt - L * histl) / (kl * (L - It))
        Formula 10: delta2 = (L * histl - It * histt) / (kr * It)
        Formula 11: omega  = exp( - (Im - It)**2 / (2 * sigma**2) )

    No texto (secao 3.2.1 Adaptive brightness control):
        It is the target average value 127
        where sigma defaults to 30
        "omega = exp(-(Im - It)^2 / (2 sigma^2))"

    Observação:
        - 1e-8, margem de segurança par evitar ZeroDivisionError

    Retorna:
        delta1: viés de brilho para bins abaixo de Im
        delta2: viés de brilho para bins acima de Im
        w     : fator de ponderacao omega
    """
    # converte Im para indice inteiro de bin
    m = int(Im)

    # soma dos bins abaixo de Im (small data) no histograma ajustado hist2
    histl = hist2[:m].sum()

    # soma total do histograma ajustado; deve ser igual a L (255) se hist2 for escala L
    histt = hist2.sum()

    # L e It do artigo; L e o alcance dinamico = 255, It valor medio alvo = 127 por padrao
    L = 255.0

    # kl e kr de acordo com a definicao
    kl = m
    kr = 256 - m

    # calcula delta1 conforme Formula (9)
    # (It * histt - L * histl) / (kl * (L - It))
    delta1 = (It * histt - L * histl) / (kl * (L - It) + 1e-8)

    # calcula delta2 conforme Formula (10)
    # (L * histl - It * histt) / (kr * It)
    delta2 = (L * histl - It * histt) / (kr * It + 1e-8)

    # constante sigma do artigo
    sigma = 30.0

    # calcula omega conforme Formula (11)
    # exp(- (Im - It)^2 / (2 sigma^2))
    w = np.exp(-((Im - It) ** 2) / (2 * sigma * sigma))

    return delta1, delta2, w



def step6_cdf_model(hist2, delta1, delta2, Im, lambda_):
    """
    Passo 6 - Formulas 7 e 8:
        Formula 7: hist3(k) = hist2(k) + delta1   para k < Im_index
        Formula 8: hist3(k) = hist2(k) + delta2   para k >= Im_index

    No Texto (secao 3.1.3 Brightness control):
        "A luminance adjustment factor is introduced to correct hist2.
        quando CDF(Im) <= lambda, add delta1 aos bins da parte esquerda,
        quando CDF(Im) > lambda, add delta2 aos bins da parte direita."

    Retorna:
        hist3: histograma apos correcao de brilho, reclipado e renormalizado
    """
    # converte Im para indice inteiro de bin
    
    m = int(Im)
    hist3 = hist2.copy()

    # calcula CDF(hist2, Im)
    cdf2 = np.cumsum(hist2)
    if cdf2[m] <= lambda_:
        # Formula 7: adiciona delta1 apenas aos bins < Im
        for i in range(m):
            hist3[i] += delta1
    else:
        # Formula 8: adiciona delta2 apenas aos bins >= Im
        for i in range(m, len(hist3)):
            hist3[i] += delta2

    # remove possiveis valores negativos e renormaliza
    hist3 = np.clip(hist3, 0, None)
    hist3 /= hist3.sum()
    return hist3


def step7_truncate_hist(hist3: np.ndarray, a: float = 2.0, b: float = 5.0) -> np.ndarray:
    """
    Passo 7 - Formulas 13 e 14:
        Formula 13: Th[g] = a * Rg[g] + b, para cada nivel de cinza g
        Formula 14: hist4[g] = min(hist3[g], Th[g])

    No Texto (secao 3.2.2 Visual restriction):
        "Rg e definido por uma funcao em quatro partes (Formula 12) que 
        reflete a resolucao do olho humano em cada nivel de cinza. 
        Depois, Th = a*Rg + b e hist4 = min(hist3, Th)."
    Retorna:
        hist4: histograma truncado e renormalizado
    """

    # numero de niveis de cinza
    n = len(hist3)  # deve ser 256

    # calcula a funcao Rg para cada nivel g de 0 a n-1 - Formula 12
    Rg = np.zeros(n)
    for g in range(n):
        if   0 <= g <  32:
            Rg[g] = -g/8 + 6
        elif 32 <= g <  64:
            Rg[g] = -g/32 + 3
        elif 64 <= g < 192:
            Rg[g] =  g/128 + 0.5
        else:  # 192 <= g < 256
            Rg[g] =  g/64 - 1

    # Formula 13: calcula o limiar Th para cada nivel
    Th = a * Rg + b

    # Formula 14: trunca cada bin de hist3 pelo limiar Th correspondente
    hist4 = np.minimum(hist3, Th)

    # renormaliza para que a soma seja 1
    # não ser mais 1, o que quebraria a interpretação como distribuição de probabilidade.
    total = hist4.sum()
    if total > 0:
        hist4 = hist4 / total

    return hist4


def step8_equalize(gray, hist):
    """
    Passo 8: Equalizar imagem usando cdf do histograma final (hist4).

    No texto (secao 3.1 final):
        "Equalizar hist4 para obter a imagem aprimorada."
    """

    # calcula a cdf acumulada do histograma
    cdf = np.cumsum(hist)
    # escala a cdf para o intervalo 0 a 255
    cdf = 255 * cdf / cdf[-1]
    # faz a remapeamento dos pixels usando a cdf escalada
    equalized = np.interp(gray.flatten(), np.arange(256), cdf)
    # remodela ao formato original e converte para uint8
    # print(f"{equalized=}")

    return equalized.reshape(gray.shape).astype(np.uint8)



if __name__ == "__main__":

    path = os.path.abspath(__file__)
    os.makedirs("image_results", exist_ok=True)
    
    files = [
        os.path.join(root, file)
        for root, dirs, files_list in os.walk(path.replace("main.py", "images"))
        for file in files_list
    ]

    for index, image_path in enumerate(files):

        print(f"Precessing: {index+1}/{len(files)}")

        filename = "image_results/" + image_path.split("/")[-1]
        image = cv2.imread(image_path)

        gray,  hist, Im     = step1_normalized_histogram(image)
        gamma1, hist1       = step2_gamma1_adjustment(hist, Im)
        gamma2              = step3_gamma2(gamma1)
        hist2               = step4_hist2_small_values(hist1, gamma2)
        delta1, delta2, w   = step5_bias_and_weights(hist, hist2, Im)
        hist3               = step6_cdf_model(hist2, delta1, delta2, w, Im)
        hist4               = step7_truncate_hist(hist3)
        enhanced            = step8_equalize(gray, hist4)

        # outros
        he = step8_equalize(gray, hist)

        #  Initial adjustment (passo 2)
        initial = step8_equalize(gray, hist1)


        # Visualizar
        fig, axs = plt.subplots(1, 5, figsize=(30, 5))
        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original")
        axs[1].imshow(gray, cmap='gray')
        axs[1].set_title("Gray Scale")
        axs[2].imshow(he, cmap='gray')
        axs[2].set_title("HE")
        axs[3].imshow(initial, cmap='gray')
        axs[3].set_title("Initial Adjustment")
        axs[4].imshow(enhanced, cmap='gray')
        axs[4].set_title("Enhanced (AHEVPC)")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{filename}_final.png")


        # save one img
        cv2.imwrite(f"{filename}_gray.png", gray)
        cv2.imwrite(f"{filename}_enhanced.png", enhanced)

