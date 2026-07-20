# Avaliação offline

## Como rodar

Modo mock (sem custo Azure, RAG fake + judge determinístico):

```powershell
python -m evaluation.run_evaluation --mock
```

Modo real (precisa de `AZURE_OPENAI_API_KEY` e `AZURE_OPENAI_ENDPOINT`):

```powershell
python -m evaluation.run_evaluation
```

Saídas em `docs/evaluation_results.json` (detalhado) e `docs/evaluation_results.md` (tabela resumo).

## Golden set

`data/golden_set/datathon_v1.jsonl` — 22 pares cobrindo 6 categorias:

| Categoria | Descrição | N |
| --- | --- | --- |
| recommendation | Recomendação personalizada para cliente específico | 5 |
| comparison | Comparação entre ofertas | 4 |
| lookup | Métricas históricas e features | 4 |
| demographic | Segmentos demográficos (idade, job) | 3 |
| conditional | Condições de elegibilidade e restrições | 4 |
| edge_case | Casos limite (cliente desconhecido, oferta inexistente) | 3 |

Schema de cada linha:

```json
{
  "id": "q-001",
  "category": "recommendation",
  "query": "...",
  "expected_answer": "...",
  "contexts": ["..."],
  "expected_tools": ["get_customer_context"]
}
```

Pra adicionar novos pares, basta acrescentar uma linha no JSONL e re-rodar.

## RAGAS

4 métricas padrão computadas via `ragas` (modo real) ou via overlap lexical (modo mock):

- **faithfulness** — a resposta se sustenta nos contexts recuperados?
- **answer_relevancy** — a resposta atende à pergunta?
- **context_precision** — quanto do contexto recuperado é relevante?
- **context_recall** — o contexto cobre o que era esperado?

## LLM-as-judge

4 critérios:

1. **factual_correctness** — alinhamento factual com `expected_answer`
2. **relevance** — resposta direta à pergunta
3. **completeness** — cobertura dos pontos-chave
4. **business_alignment** — respeita regras do catálogo (nomes de oferta corretos, restrições como `loan=True` bloqueia `loan_personal`, `housing=True` desbloqueia `insurance_basic`)

Cada critério devolve `{score: 0.0–1.0, justification: str}`. O agregado calcula média por critério e média geral.

## Como interpretar

- RAGAS > 0.7 em todas as 4 métricas = pipeline RAG sólido
- LLM-as-judge overall_mean > 0.7 = respostas confiáveis pra produção
- `business_alignment` é o critério mais crítico pra esse caso de uso — score baixo aqui significa que o agente está sugerindo ofertas incompatíveis com regras de negócio
