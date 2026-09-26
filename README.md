# CC3092 — Laboratorio 7: Forecasting sobre ETTh1 (LSTM y Transformer desde cero)

Implementación completa del Laboratorio 7 un sistema de forecasting sobre el dataset **ETTh1** (Electricity Transformer Temperature) usando dos arquitecturas construidas con tensores PyTorch puros — un **LSTM many-to-one** y un **Transformer encoder con self-attention** — evaluadas sobre los horizontes H=24 y H=48 horas.

## Contenido

- `lab7.ipynb`: 

## Cómo correrlo

```bash
pip install torch numpy matplotlib jupyter nbconvert nbformat
jupyter nbconvert --to notebook --execute --inplace lab7.ipynb
```

El notebook descarga `ETTh1.csv` directamente desde GitHub la primera vez que se ejecuta. Corre en CPU o GPU sin cambios de código, y las semillas (`torch.manual_seed`, `np.random.seed`) están fijas para reproducibilidad.

## Restricciones respetadas

Todo el código de LSTM y Transformer está implementado con tensores puros de PyTorch (`nn.Linear`, `nn.Parameter`, `torch.optim.Adam`). No se usa `nn.LSTM`, `nn.MultiheadAttention` ni `nn.LayerNorm` en ningún punto — la celda LSTM, el multi-head attention y el LayerNorm están escritos a mano.

## Resultados (validación, escala normalizada)

| Modelo      | Horizonte | MAE    | RMSE   | Ratio vs naive |
|-------------|-----------|--------|--------|----------------|
| LSTM        | 24h       | 0.2026 | 0.2726 | 1.015          |
| LSTM        | 48h       | 0.2771 | 0.3623 | 1.064          |
| Transformer | 24h       | 0.1910 | 0.2495 | 0.957          |
| Transformer | 48h       | 0.2378 | 0.3079 | 0.913          |
| Naive       | 24h       | 0.1996 | 0.2672 | 1.000          |
| Naive       | 48h       | 0.2604 | 0.3359 | 1.000          |

El Transformer supera al baseline naive y al LSTM en ambos horizontes, y la ventaja crece con el horizonte de predicción. El análisis completo (ACF, justificación de la función de pérdida, mapas de atención, comparación matemática de flujo de gradiente entre ambas arquitecturas) está en las celdas Markdown del notebook.
