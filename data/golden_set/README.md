# Golden Set de Avaliacao Offline

Este diretorio versiona casos fixos usados para avaliar a politica de recomendacao antes de servir em producao.

## Arquivo principal

- `evaluation_cases.jsonl`: um caso por linha, em JSON.

## Campos esperados

- `case_id`: identificador unico do caso.
- `segment`: segmento sintetico usado na analise de exposicao.
- `context`: payload compativel com `CustomerContext`.
- `candidate_offers`: lista de ofertas candidatas.
- `expected_action`: acao esperada para o caso.
- `expected_reward`: recompensa esperada para a acao esperada.
- `justification`: porque o caso existe.
- `pass_criteria`: criterio de sucesso do caso.
- `policy_should_not_be_used`: marca casos fora de escopo.
- `sensitivity_probe`: instrucao simples para o teste de sensibilidade.

## Como rodar a avaliacao

```powershell
python -m src.evaluation.offline_golden_set
```

O comando gera um resumo reproduzivel com comparacao entre baseline deterministico e Thompson sampling, alem de analise de sensibilidade e exposicao por segmento.