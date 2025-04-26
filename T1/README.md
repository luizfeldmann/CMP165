# Trabalho 1

Implementação de um algoritmo de correção de constrate.

---

## Execução

Em ambiente linux.

1. Instalar requisitos

```bash
pip install -r requirements.txt
```

2. Executar método de equalização de histograma

```bash
python3 main.py
```

3. Obter métricas

```bash
python3 metrics.py
```

## Resultados obtidos

| Image | NIQE_gray   | NIQE_enhanced | PIQE_gray   | PIQE_enhanced | MCMA_gray   | MCMA_enhanced |
|-------|-------------|---------------|-------------|---------------|-------------|---------------|
| a4760 | 11.92348877 | 11.23369181   | 28.82517521 | 26.46253617   | 0.39668571  | 0.394447435   |
| a2689 | 16.17844943 | 13.99741502   | 15.65956191 | 9.365395681   | 0.329547263 | 0.329816735   |
| a1601 | 16.47559118 | 12.17043349   | 13.37303375 | 10.83519593   | 0.344269089 | 0.35449118    |
| a0762 | 12.39261008 | 11.98698533   | 37.37449414 | 31.13251144   | 0.372347786 | 0.37146506    |
| a0317 | 16.00125307 | 15.38291279   | 45.48022362 | 34.5025943    | 0.200597684 | 0.323719361   |
| a3804 | 13.32739238 | 13.38245933   | 43.35032685 | 39.70327709   | 0.371947842 | 0.370970186   |
| a0386 | 11.67138082 | 9.775781437   | 37.44001256 | 41.53529743   | 0.361247142 | 0.358276579   |
| a0639 | 10.99032239 | 10.20978007   | 69.84139396 | 67.97385684   | 0.394088993 | 0.40449568    |

**NIQE**: quanto menor melhor
**PIQE**: quanto menor melhor
**MCMA**: quanto menor melhor

## Referências

YUAN, Qi; DAI, Shengkui. **Adaptive histogram equalization with visual perception consistency.** Information Sciences. [S. l.]: Elsevier BV, maio 2024. DOI 10.1016/j.ins.2024.120525. Disponível em: [http://dx.doi.org/10.1016/j.ins.2024.120525](https://doi.org/10.1016/j.ins.2024.120525).
