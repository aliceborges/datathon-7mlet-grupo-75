# Datathon 7MLET - Grupo 75

Plataforma de recomendação de ofertas em canais digitais bancários usando Multi-Armed Bandit. A ideia é fugir das regras fixas e dos testes A/B longos, deixando o próprio sistema equilibrar exploração e explotação a partir do que vai observando.

A base de referência é o [telemarketing-jyb-dataset](https://www.kaggle.com/datasets/aguado/telemarketing-jyb-dataset/data) do Kaggle. Removemos colunas que geram vazamento temporal (`duration` é o caso clássico) antes de qualquer feature engineering.

## O que tem no repo

Duas políticas de decisão convivem:

- Um **baseline determinístico** que sempre pega a oferta com maior conversão histórica registrada em `DEFAULT_HISTORICAL_CONVERSION`.
- Um **Thompson Sampling** com priors Beta(1, 1) por braço, que aprende com recompensa binária (converteu ou não).

O contexto do cliente entra na decisão em dois pontos: filtrando os candidatos elegíveis (`candidate_offers`) e enriquecendo os `reason_codes` do output (`housing_loan_synergy`, `senior_segment`, `high_balance` etc.). O sampling do Thompson em si é não-contextual — mantém um par (α, β) por braço sem features do cliente. Extensão pra Thompson contextual ou LinUCB ficaria como próximo passo pra ganhar aprendizado por segmento.

A API FastAPI expõe `/predict` pra servir a decisão (usa um stub Thompson simplificado em `src/api/model_loader.py`) e `/agent` pro assistente LLM justificar decisões via RAG. A comparação completa Baseline vs Thompson roda no notebook — a API só tem o stub. Arquitetura alvo em Azure documentada com detalhes em `docs/architecture-azure.md`.

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

Pra popular runs na UI, sobe o servidor num terminal e aponta o `MLFLOW_TRACKING_URI` antes de abrir o notebook:

```powershell
# Terminal 1
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns.db
```

```powershell
# Terminal 2
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
jupyter lab notebooks/01_eda.ipynb
```

A célula de comparação Baseline vs Thompson loga `conversion_rate`, `total_regret`, `exploration_rate` e o delta entre as políticas como métricas, mais os parâmetros do cenário e as tags padrão do grupo (`model_name`, `model_type`, `framework`, `owner`, `risk_level`, `phase`, `group`). Abre `http://localhost:5000` pra ver.

O snapshot da última execução vive em `docs/mlflow_runs.md` (params + métricas + tags). Se você já rodou o notebook uma vez, o diretório `mlruns/` (ignorado pelo git) tem os arquivos brutos — dá pra abrir `mlflow ui --backend-store-uri file:./mlruns` pra explorar sem precisar subir o tracking server.

Existe também `src/models/train.py` com `train_and_log()` pra registrar runs de modelos sklearn (não usado no notebook, disponível como biblioteca). Testes em `tests/test_train_integration.py`.

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

A arquitetura-alvo, se fôssemos colocar em produção, seria em Azure — encaixa bem com os componentes gerenciados que a gente conhece do curso. A API FastAPI rodaria num App Service (ou AKS se precisasse escalar), atrás de um Application Gateway. Os segredos ficariam no Key Vault, consumidos via Managed Identity, então nada de chave hardcoded ou em env var na esteira. MLflow Tracking e Model Registry migrariam pra Azure Machine Learning, e os datasets ficariam versionados no Data Lake Gen2. Logs de decisão iriam pra Cosmos DB ou Azure SQL, dependendo se o acesso fosse mais orientado a documento ou a schema.

No piloto atual, o assistente LLM usa Azure OpenAI Service (plano Azure for Students, deployment `model-router`). Em produção, migraria pra deployments dedicados (chat e embeddings) com quota reservada. Observabilidade em produção iria pra Application Insights + Log Analytics — no piloto local usamos Prometheus + Grafana via Docker Compose. Detecção de drift seria job periódico via Databricks ou Azure Functions usando Evidently. Diagrama Mermaid completo e discussão qualitativa de FinOps em `docs/architecture-azure.md`.

## Mapa de pastas

- `data/`: bases Kaggle, camada processada, golden set e corpus RAG. Sintéticos em `data/synthetic_enrichment/`.
- `docs/`: documentação de API, arquitetura Azure, benchmark, MLOps, model card, system card e LGPD.
- `evaluation/`: avaliação offline do assistente LLM (golden set em linguagem natural, RAGAS, LLM-as-judge, CLI runner).
- `monitoring/`: configs de Prometheus e dashboards Grafana provisionados.
- `notebooks/`: `01_eda.ipynb` com EDA, comparação de políticas e casos de teste.
- `reports/`: relatório da geração dos dados sintéticos.
- `src/`: código-fonte (API, agente ReAct, políticas MAB, guardrail, monitoramento).
- `tests/`: suíte pytest (unitários + integração).

## Além do mínimo

O PDF pede baseline, adaptativo e API funcionando. A gente foi um pouco além pra deixar mais parecido com o que rodaria em produção de verdade:

- Endpoint `/agent` com agente ReAct (LangChain + Azure OpenAI), 3 tools customizadas e RAG sobre `data/rag_corpus/` via Chroma. Se a chave Azure não estiver setada, esse endpoint desliga sozinho e a API continua servindo `/predict` normal.
- Guardrail contra prompt injection em `src/security/guardrails.py` — 12 padrões (EN + PT) filtrados antes do LLM. Bloqueia coisas tipo "ignore all previous instructions". Cobertura em `tests/test_guardrails.py`.
- Logs de auditoria em JSONL. Cada decisão gera uma linha com decision_id, endpoint, policy_version, chosen_offer, tools_used e latência. Caminho configurável via `AUDIT_LOG_PATH`.
- Métricas Prometheus custom expostas em `/metrics` (`datathon_decisions_total`, `datathon_agent_latency_seconds`, `datathon_agent_tools_used`), consumidas pelo Grafana provisionado em `monitoring/`.
- Pipeline CI em `.github/workflows/ci.yml` rodando ruff, black, mypy, bandit e pytest com cobertura a cada push ou PR.
- 147 testes automatizados cobrindo API, agente, RAG, guardrail, MLflow, drift, bandit e golden set.
- Avaliação offline do LLM via `evaluation/run_evaluation.py` — RAGAS (faithfulness, answer_relevancy, context_precision, context_recall) e LLM-as-judge com 4 critérios sobre 22 pares Q&A. Modo `--mock` pra rodar em CI sem consumir Azure.
- Detecção de drift em `src/monitoring/drift.py` com PSI (Population Stability Index) e wrapper Evidently. PSI acima de 0.10 gera warning, acima de 0.20 é sinal de retrain.
- Stack completo em Docker Compose (API + MLflow + Prometheus + Grafana).

### Exemplos rápidos

Com a API no ar:

```bash
# Decisão pura (funciona sem Azure)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"context":{"customer_id":"C001","age":34,"balance":1500.0,"housing":true}}'

# Assistente LLM (precisa Azure)
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"question":"Qual oferta pra cliente 34 anos com housing?","customer_id":"C001"}'

# Guardrail interceptando
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"question":"Ignore all previous instructions"}'
```

## Dados e conformidade

A base legal do processamento apoia em consentimento e legítimo interesse (LGPD Art. 7º I e IX), com minimização (só features estritamente necessárias, sem dados sensíveis) e retenção de 90 dias pros logs de decisão do `/predict`. As perguntas do `/agent` não são persistidas. Detalhes em `docs/lgpd-plan.md`.

## Limitações

- Os dados são públicos do Kaggle mais artefatos sintéticos. Nenhum PII real passa pelo pipeline.
- A camada processada tira `id`, `duration` e outras colunas auxiliares pra fechar a porta pra vazamento temporal.
- A simulação de delayed rewards usa hipóteses fixas com sementes documentadas — não representa distribuição real de mercado.
- O `/agent` depende de Azure OpenAI. Sem as chaves, só o stub de decisão fica de pé.
- A latência atual atende bem pra demo; sistema bancário de alta frequência exigiria bem mais tuning.
