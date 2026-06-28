# API de decisão e agente

## Subindo local

Pré-requisito: stack do `docs/mlops.md` rodando (mlflow + prometheus + grafana).

```powershell
$env:AZURE_OPENAI_API_KEY = "..."
$env:AZURE_OPENAI_ENDPOINT = "https://<seu-recurso>.openai.azure.com"
$env:AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Sem `AZURE_OPENAI_API_KEY`, a API sobe normalmente e o `/predict` funciona, mas o `/agent` retorna `503`.

## Endpoints

### `GET /health`

Retorna `{"status": "ok", "version": "0.1.0"}`.

### `POST /predict`

Recomendação direta do MAB sem chamar LLM.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "context": {
      "customer_id": "C001",
      "age": 34,
      "balance": 1500.0,
      "housing": true
    }
  }'
```

Resposta:

```json
{
  "decision_id": "uuid",
  "policy_version": "stub-thompson-v0",
  "chosen": {
    "offer_id": "loan_payroll",
    "score": 0.72,
    "reason_codes": ["thompson_sampled(seed=42)", "high_balance"]
  },
  "alternatives": [...],
  "served_at": "2026-06-28T..."
}
```

Erros:
- `400` se `candidate_offers` tem `offer_id` desconhecido
- `422` se o payload viola o schema Pydantic (ex: age < 18)

### `POST /agent`

Pergunta livre ao agente ReAct. O agente usa até 3 tools (`get_customer_context`, `lookup_offer_details`, `historical_performance`) e devolve resposta em texto + lista de tools usadas.

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Qual a melhor oferta pro cliente C002 e por quê?",
    "customer_id": "C002"
  }'
```

Erros: `503` se Azure OpenAI não está configurado.

### `GET /metrics`

Endpoint Prometheus (instrumentado via `prometheus-fastapi-instrumentator`). Métricas default: latência por endpoint, contagem por status code, requests in-flight.

## Audit log

Toda decisão (`/predict` e `/agent`) gera uma entrada JSONL em `logs/audit.jsonl` (configurável via `AUDIT_LOG_PATH`). Campos: `decision_id`, `endpoint`, `policy_version`, `customer_id`, `chosen_offer`, `reason_codes`, `tools_used`, `latency_ms`, `created_at`.

## Política de recomendação

`src/api/model_loader.py` tenta carregar o champion do MLflow Registry (`models:/<CHAMPION_MODEL_NAME>/champion`). Se falhar ou as env vars não estiverem setadas, cai num `ThompsonSamplingStub` determinístico com 5 ofertas pré-definidas — substituível pelo modelo real quando ele for registrado no MLflow.

## RAG

Corpus em `data/rag_corpus/*.md`, indexado em Chroma em memória no startup da API. Indexação por seção `##`. Para customizar o corpus, basta adicionar/editar arquivos `.md` nessa pasta.
