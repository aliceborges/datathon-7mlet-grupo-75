# Datathon 7MLET - Grupo 75

Plataforma de recomendação de ofertas em canais digitais bancários usando Multi-Armed Bandit. A ideia é fugir das regras fixas e dos testes A/B longos, deixando o próprio sistema equilibrar exploração e explotação a partir do que vai observando.

A base de referência é o [telemarketing-jyb-dataset](https://www.kaggle.com/datasets/aguado/telemarketing-jyb-dataset/data) do Kaggle. Removemos colunas que geram vazamento temporal (`duration` é o caso clássico) antes de qualquer feature engineering.

## O que tem no repo

Duas políticas de decisão convivem:

- Um **baseline determinístico** que sempre pega a oferta com maior conversão histórica registrada em `DEFAULT_HISTORICAL_CONVERSION`.
- Um **Thompson Sampling** com priors Beta(1, 1) por braço, que aprende com recompensa binária (converteu ou não).

Ambas rodam por trás de uma API FastAPI em dois endpoints — `/predict` para a decisão e `/agent` para explicar a decisão via LLM. Arquitetura alvo em Azure documentada com detalhes em `docs/architecture-azure.md`.

## Notebook

`notebooks/01_eda.ipynb` tem tudo que precisa pra entender o problema e ver o modelo funcionando:

- EDA da base Kaggle: distribuição do target, corte por idade e profissão.
- Comparação Baseline vs Thompson em 500 rodadas simulando um cenário com drift de conversão. O Thompson supera o baseline tanto em taxa de conversão quanto em regret acumulado.
- Cinco casos de teste tirados de `data/golden_set/evaluation_cases.jsonl` com a recomendação de cada política e o que esperávamos que fosse recomendado.

Precisa dos CSVs Kaggle baixados pra rodar a EDA. As células de bandit não dependem dos CSVs e rodam soltas.

## Rodando local

**Pré-requisitos:** Python 3.10+ e Git.

Clona e entra na pasta:

```bash
git clone https://github.com/aliceborges/datathon-7mlet-grupo-75.git
cd datathon-7mlet-grupo-75
```

Cria a venv e ativa:

```powershell
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

Instala em modo editável com as libs de dev:

```bash
pip install -e .[dev]
```

Copia o template do `.env` e preenche localmente (o `.env` real fica fora do git):

```powershell
# Windows
copy .env.example .env
```

```bash
# Linux/macOS
cp .env.example .env
```

As variáveis principais são `API_HOST`, `API_PORT`, `LOG_LEVEL`, `AUDIT_LOG_PATH`, `MLFLOW_TRACKING_URI`, `CHAMPION_MODEL_NAME` e o bloco Azure OpenAI (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`, `LLM_TEMPERATURE`).

Sem credenciais Azure, o `/predict` continua funcionando com o stub Thompson. O `/agent` responde 503 até você preencher as chaves.

### Base Kaggle

1. Baixa `train.csv` e `test.csv` de https://www.kaggle.com/datasets/aguado/telemarketing-jyb-dataset/data.
2. Joga os dois em `data/kaggle/`.
3. Roda o preprocessing:

```bash
python data/make_dataset.py
```

O output vai pra `data/processed/` (`train_clean.csv`, `test_clean.csv`). Se quiser regenerar os artefatos sintéticos (`offer_catalog.csv`, `offer_events.csv`, `delayed_rewards.csv`):

```bash
python data/synthetic_enrichment/generator.py --seed 42
```

### Abrir o notebook

```bash
jupyter lab notebooks/01_eda.ipynb
```

### Subir a API

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Swagger em `http://localhost:8000/docs`, OpenAPI schema em `http://localhost:8000/openapi.json`.

### MLflow

```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns.db
```

UI em `http://localhost:5000`. O código que loga runs vive em `src/models/train.py` e registra parâmetros, métricas e as tags padrão do grupo (`model_name`, `model_type`, `framework`, `owner`, `risk_level`, `training_data_version`, `git_sha`, `phase`, `group`). O teste de integração em `tests/test_train_integration.py` prova que a persistência funciona.

### Docker (opcional)

```bash
docker compose up -d
```

Sobe API + MLflow + Prometheus + Grafana. Precisa do `.env` preenchido pra o `/agent` responder.

### Testes

```bash
pytest
```

## Arquitetura Azure

A escolha foi rodar tudo em Azure porque encaixa bem com os componentes gerenciados que a gente conhece do curso. A API FastAPI vive num App Service (ou AKS se precisar escalar), atrás de um Application Gateway. Os segredos ficam no Key Vault e são consumidos via Managed Identity, então nada de chave hardcoded ou em env var na esteira. MLflow Tracking e Model Registry migram pra Azure Machine Learning, e os datasets ficam versionados no Data Lake Gen2. Logs de decisão vão pra Cosmos DB ou Azure SQL, dependendo se o acesso é mais orientado a documento ou a schema.

Do lado do assistente LLM, usamos Azure OpenAI Service. No piloto atual estamos no plano Azure for Students com deployment `model-router`, o que dá conta pra demo. Em produção migraria pra deployments dedicados (chat e embeddings) com quota reservada. Observabilidade end-to-end fica no Application Insights + Log Analytics, e a detecção de drift roda como job periódico via Databricks ou Azure Functions usando Evidently. O diagrama Mermaid completo e uma discussão qualitativa de FinOps estão em `docs/architecture-azure.md`.

## Mapa de pastas

- `data/`: bases Kaggle, camada processada, golden set e corpus RAG. Sintéticos em `data/synthetic_enrichment/`.
- `docs/`: documentação de API, arquitetura Azure, benchmark, MLOps, model card, system card e LGPD.
- `evaluation/`: avaliação offline do assistente LLM (golden set em linguagem natural, RAGAS, LLM-as-judge, CLI runner).
- `monitoring/`: configs de Prometheus e dashboards Grafana provisionados.
- `notebooks/`: `01_eda.ipynb` com EDA, comparação de políticas e casos de teste.
- `reports/`: relatório da geração dos dados sintéticos.
- `src/`: código-fonte (API, agente ReAct, políticas MAB, guardrail, monitoramento).
- `tests/`: suíte pytest (unitários + integração).

## Limitações

- Os dados são públicos do Kaggle mais artefatos sintéticos. Nenhum PII real passa pelo pipeline.
- A camada processada tira `id`, `duration` e outras colunas auxiliares pra fechar a porta pra vazamento temporal.
- A simulação de delayed rewards usa hipóteses fixas com sementes documentadas — não representa distribuição real de mercado.
- O `/agent` depende de Azure OpenAI. Sem as chaves, só o stub de decisão fica de pé.
- A latência atual atende bem pra demo; sistema bancário de alta frequência exigiria bem mais tuning.
