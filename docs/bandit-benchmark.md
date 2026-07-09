# Benchmark de Bandit

## Políticas comparadas

- **Baseline determinístico**: escolhe a oferta com maior conversão histórica observada.
- **Política adaptativa**: Thompson Sampling com priors Beta(1, 1) por oferta.

## Métricas reportadas

- **Recompensa total**: soma das conversões simuladas.
- **Regret total**: diferença acumulada entre a melhor oferta possível e a oferta escolhida.
- **Exploração**: fração de rounds em que a política adaptativa não segue a escolha determinística.
- **Conversão simulada**: reward dividido pelo número de rounds.
- **Cold-start**: fração de rounds em que a oferta escolhida ainda não tinha histórico observado.
- **Delayed reward**: fração de conversões que carregam atraso e média de dias de atraso.

## Interpretação

- O baseline dá uma referência estável, fácil de explicar e reproduzir.
- Thompson Sampling explora ofertas novas quando a incerteza ainda é alta e reduz isso à medida que recebe retorno.
- Cold-start fica explícito porque ofertas sem histórico começam com prior neutra e são contadas separadamente nos relatórios.
- Recompensas atrasadas são reportadas como atraso associado ao evento convertido, para evitar confundir conversão com disponibilidade imediata do sinal.

## Uso

O fluxo principal está em `src/evaluation/bandit_benchmark.py`.

Exemplo:

```python
from src.evaluation.bandit_benchmark import compare_bandit_policies

result = compare_bandit_policies(scenarios, seed=75)
print(result["baseline"].to_dict())
print(result["adaptive"].to_dict())
print(result["delta"])
```
