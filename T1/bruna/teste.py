import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_gamma(hist, target_cdf, im_mean, gamma_range=np.linspace(0.1, 2.0, 100)):
    """Busca iterativa para encontrar gamma_1 ótimo (Eq. 2-3 do artigo)"""
    min_diff = float('inf')
    best_gamma = 1.0
    for gamma in gamma_range:
        hist_gamma = hist ** gamma
        cdf = np.cumsum(hist_gamma) / hist_gamma.sum()
        current_cdf = cdf[int(im_mean)]
        diff = abs(current_cdf - target_cdf)
        if diff < min_diff:
            min_diff = diff
            best_gamma = gamma
    return best_gamma

def adaptive_histogram_equalization(image, target_mean=127, alpha=1.0, a=2, b=5):
    # Converter para escala de cinza se for colorida
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Passo 1: Normalizar o histograma (Eq. 1)
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    hist = hist.astype(np.float32)
    hist_normalized = hist * 255 / hist.sum()  # Eq. 1: hist = h × L
    
    # Passo 2: Ajuste inicial baseado em gamma (Eq. 2-4)
    im_mean = np.mean(image)
    lambda_t = target_mean / 255  # Quantil do brilho alvo
    
    gamma_1 = compute_gamma(hist_normalized/255, lambda_t, im_mean)
    hist_1 = (hist_normalized / 255) ** gamma_1 * 255  # Eq. 4
    
    # Passo 3: Proteção de detalhes (Eq. 5-6)
    mask_small_data = hist_1 < 1
    gamma_2 = alpha * (1 - gamma_1)  # Eq. 6
    hist_2 = np.where(mask_small_data, hist_1 ** gamma_2, hist_1)  # Eq. 5
    
    # Passo 4: Controle de brilho (Eq. 7-10)
    # Separar sub-histogramas esquerdo/direito
    hist_l = hist_2[:int(im_mean)+1]
    hist_r = hist_2[int(im_mean)+1:]
    k_l = len(hist_l[hist_l > 0])  # Número de bins não-zero à esquerda
    k_r = len(hist_r[hist_r > 0])  # Número de bins não-zero à direita
    
    cdf_m_2 = np.cumsum(hist_2 / hist_2.sum())[int(im_mean)]
    if cdf_m_2 <= lambda_t:
        # Calcular delta_1 (Eq. 9)
        delta_1 = (target_mean * hist_2.sum() - 255 * hist_2.sum()) / (k_l * (255 - target_mean))
        hist_3 = np.where(hist_2 > 0, hist_2 + delta_1, hist_2)  # Eq. 7
    else:
        # Calcular delta_2 (Eq. 10)
        delta_2 = (255 * hist_2.sum() - target_mean * hist_2.sum()) / (k_r * target_mean)
        hist_3 = np.where(hist_2 > 0, hist_2 + delta_2, hist_2)  # Eq. 8
    
    # Passo 5: Restrição visual (Eq. 12-14)
    gray_levels = np.arange(256)
    Rg = np.piecewise(gray_levels,
                    [gray_levels < 32, (32 <= gray_levels) & (gray_levels < 64),
                    (64 <= gray_levels) & (gray_levels < 192), gray_levels >= 192],
                    [lambda x: -x/8 + 6, lambda x: -x/32 + 3,
                    lambda x: x/128 + 0.5, lambda x: x/64 - 1])  # Eq. 12
    Th = a * Rg + b  # Limiar de truncamento (Eq. 13)
    hist_4 = np.minimum(hist_3, Th)  # Eq. 14
    
    #Passo 6: Equalização final
    cdf = np.cumsum(hist_4)
    cdf_normalized = cdf * 255 / cdf[-1]  # Não renormalizar (preserva brilho)
    enhanced_image = np.interp(image.flatten(), bins[:-1], cdf_normalized).reshape(image.shape)
    
    return enhanced_image.astype(np.uint8), gamma_1, gamma_2

#Exemplo de uso
if __name__ == "__main__":
    image = cv2.imread('fig3.jpg', cv2.IMREAD_GRAYSCALE)
    enhanced_img, gamma1, gamma2 = adaptive_histogram_equalization(image)
    
    # Visualização
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title(f'AHEVPC (γ1={gamma1:.2f}, γ2={gamma2:.2f})')
    plt.imshow(enhanced_img, cmap='gray')
    # plt.show()
    plt.savefig("resultado.png")

