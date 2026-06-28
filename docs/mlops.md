# MLOps — Infraestrutura local e quality gates

## Subindo a infra local

```powershell
docker compose up -d mlflow prometheus grafana
```

Sobe três serviços:

| Serviço | URL | Login |
| --- | --- | --- |
| MLflow UI | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

O backend do MLflow é SQLite persistido no volume `mlflow-data`. Para zerar tudo: `docker compose down -v`.

## Quality gates

Roda local antes de abrir PR:

```powershell
ruff check src tests
black --check src tests
mypy src --ignore-missing-imports
bandit -r src -c pyproject.toml
pytest tests --cov=src --cov-fail-under=30
```

Mesmo pipeline roda no GitHub Actions (`.github/workflows/ci.yml`) a cada push/PR para `main` ou `develop`. Falha em qualquer gate bloqueia o merge.

Para automatizar localmente:

```powershell
pre-commit install
```

A partir daí ruff/black/bandit rodam antes de cada commit.

## Tracking MLflow

Treinos passam pelo helper `train_and_log` em `src/models/train.py`, que aplica o conjunto mínimo de tags exigido pela banca:

- `model_name`, `model_type`, `framework`
- `owner` (email do responsável)
- `risk_level` (low / medium / high / critical)
- `fairness_checked` (bool)
- `training_data_version`
- `git_sha`, `phase`, `group`

Métricas padrão registradas para classificação: `auc`, `precision`, `recall`, `f1`.

## Drift detection

`src/monitoring/drift.py` calcula PSI por coluna entre referência (treino) e dados atuais:

- PSI < 0.10 → estável
- 0.10 ≤ PSI < 0.20 → warning
- PSI ≥ 0.20 → trigger de retraining

`DriftReport.to_metrics_dict()` formata o resultado para logar direto no MLflow junto com as métricas do modelo. Para relatório visual há `run_evidently_drift_report` (Evidently `DataDriftPreset`).
