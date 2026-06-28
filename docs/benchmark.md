# Benchmark do agente

## Configurações comparadas

| Config | Modelo | Temp | RAG | Max iter |
| --- | --- | --- | --- | --- |
| `baseline-cold` | gpt-4o-mini | 0.0 | sim | 8 |
| `baseline-warm` | gpt-4o-mini | 0.7 | sim | 8 |
| `no-rag-cold` | gpt-4o-mini | 0.0 | não | 8 |

Por que essas três:

- **baseline-cold** é o padrão de produção (temperatura zero, RAG ligado).
- **baseline-warm** mede o custo de variabilidade — útil pra ver se respostas mudam com `temperature=0.7`.
- **no-rag-cold** isola o efeito do RAG: mesma temperatura, mas sem retrieval no catálogo.

## Como rodar

Com Azure OpenAI real (precisa de `AZURE_OPENAI_API_KEY` e `AZURE_OPENAI_ENDPOINT` no `.env`):

```powershell
python -m evaluation.benchmark_agent
```

Sem credenciais Azure (gera template com agente mockado, útil pra CI ou pra documentar o framework):

```powershell
python -m evaluation.benchmark_agent --mock
```

Saídas:

- `docs/benchmark_results.json` — runs completos por configuração
- `docs/benchmark_results.md` — tabela resumo com latência, tools/req e tamanho de resposta

## Como interpretar

- **Latência (ms)** — tempo médio por pergunta, incluindo round-trips ao Azure e execução das tools. Esperado: 1500-5000ms por request com Azure real.
- **Tools/req** — média de tools chamadas por resposta. Mais ferramentas = raciocínio mais elaborado, mas custa latência.
- **Tam. resp.** — número de caracteres da resposta final. Indicador grosseiro de verbosidade.

Para qualidade de resposta (correção, factualidade), o framework de avaliação por LLM-as-judge fica fora deste benchmark.

## Conjunto de perguntas

Cinco perguntas fixas em `DEFAULT_QUESTIONS` cobrem três tipos de raciocínio:

1. Perfis demográficos (idosos)
2. Comparação de produtos
3. Recomendação por cliente
4. Lookup de métricas
5. Raciocínio condicional entre ofertas

Substituível via parâmetro em código se quiser benchmark com perguntas específicas do domínio.
