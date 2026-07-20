# Runs MLflow

Snapshot dos runs registrados na célula de comparação Baseline vs Thompson do `notebooks/01_eda.ipynb`. Os arquivos brutos ficam em `mlruns/` na raiz do repo.

Pra abrir a UI:

```powershell
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

`http://localhost:5000` → experimento `datathon-fase05-bandit`.

## Run mais recente

- run_id: `19e709e605ba4ad4aa9b83261eda5df0`
- run_name: `baseline_vs_thompson`
- start_time: 2026-07-12 22:22:14 UTC

### Parâmetros

| Parâmetro | Valor |
| --- | --- |
| n_rounds | 500 |
| seed | 75 |
| n_arms | 4 |
| scenario | drift_savings_premium_boost |

### Métricas

| Métrica | Valor |
| --- | --- |
| baseline_conversion_rate | 0.0260 |
| thompson_conversion_rate | 0.0300 |
| baseline_total_regret | 12.5000 |
| thompson_total_regret | 8.2520 |
| thompson_exploration_rate | 0.5800 |
| delta_conversion_rate | +0.0040 |
| delta_regret | +4.2480 |

### Tags

| Tag | Valor |
| --- | --- |
| model_name | baseline_vs_thompson |
| model_type | bandit |
| framework | custom |
| owner | grupo-75 |
| risk_level | medium |
| phase | datathon-fase05 |
| group | grupo-75 |

## Interpretação

Thompson bateu o baseline em conversão (+0.4 p.p.) e evitou 4.25 unidades de regret nas 500 rodadas. O cenário tem drift proposital: a taxa real do `savings_premium` foi setada em 4.5% enquanto o histórico registrava 2.2%. O baseline continua escolhendo pelo histórico e perde a virada; o Thompson explora, sente a nova distribuição e migra pra oferta certa.
