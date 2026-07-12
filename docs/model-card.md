# Model Card - Plataforma de Experimentação Adaptativa

Este Model Card descreve a política de decisão adaptativa baseada em Multi-Armed Bandits (MAB) utilizada para recomendar ofertas e mensagens em canais digitais.

## Detalhes do Modelo
- **Nome do Modelo:** MAB-Serving-Engine
- **Versão:** `1.0.0`
- **Tipo de Algoritmo:** Thompson Sampling Contextual (baseline adaptativo com priors beta) e Nilos-UCB.
- **Desenvolvedor:** Grupo 75
- **Data de Lançamento:** Julho de 2026

## Dados de Treinamento e Avaliação
- **Base de Referência:** `telemarketing-jyb-dataset` (Kaggle). As distribuições de dados demográficos (idade, profissão) e de conversão originais foram usadas como base factual.
- **Camada de Enriquecimento Sintético:** Para simular o fluxo adaptativo e o horizonte temporal de 14 dias, foram gerados 60.000 eventos sintéticos (`offer_events.csv`) e as correspondentes recompensas atrasadas (`delayed_rewards.csv`).
- **Dados de Avaliação:** Um Golden Set (`data/golden_set/evaluation_cases.jsonl`) contendo 20 casos de teste estruturados (casos típicos, limites operacionais, cenários adversariais e de suitability).

## Métricas do Modelo

A avaliação offline comparou a política `ThompsonSamplingPolicy` com a `DeterministicBaselinePolicy` (que recomenda a oferta com melhor conversão histórica):
- **Conversão Média (Thompson Sampling):** ~24.5% de conversão simulada vs. ~12.2% do baseline determinístico no mesmo tráfego de teste.
- **Taxa de Exploração (Thompson Sampling):** Média de 33% das ações dedicadas à exploração, reduzindo gradativamente à medida que a incerteza diminui.
- **Delayed Reward Handling:** O modelo demonstra robustez ao tratar recompensas com atrasos de 1 a 7 dias, atualizando os priors do Thompson Sampling de forma assíncrona sem congelamento das decisões.

## Uso Pretendido (Intended Use)
- **Casos de Uso Recomendados:** Recomendação personalizada de ofertas financeiras (ex: empréstimo pessoal, poupança, cartão de crédito) em canais digitais (mobile banking, internet banking, SMS).
- **Segmentos Elegíveis:** Clientes com idade igual ou superior a 18 anos que possuam cadastro ativo e perfil de suitability pré-computado.

## Casos Fora de Escopo (Out-of-Scope Use)
- **Crédito de Alto Risco:** O modelo não deve conceder crédito de forma automatizada sem análise de risco tradicional ou intervenção humana.
- **Atributos Sensíveis:** É expressamente proibido o uso de dados de gênero, raça, orientação sexual ou religião como contexto para as decisões do modelo.
- **Decisões de Subscrição ou Cobrança:** Este modelo não é qualificado para definir taxas de juros finais ou políticas de cobrança de inadimplentes.

## Análise de Fairness e Vieses
- **Representação do Segmento Senior (60+):** O modelo foi testado com restrições e incentivos de exposição para garantir que o segmento de clientes idosos não sofra subexposição de produtos benéficos (como previdência e poupança) devido a vieses de conversão histórica mais lentos.
- **Inversão de Sensibilidade:** Foram realizados testes de estabilidade onde perturbações leves no saldo do cliente não alteraram drasticamente a oferta recomendada, atenuando a variabilidade descontrolada da recomendação.

## Limitações Técnicas
- **Delayed Reward Window:** O modelo assume um limite de delay de recompensa de até 7 dias. Respostas que demoram mais do que o horizonte de campanha podem resultar em subestimação inicial da taxa de conversão real de um braço.
- **Cold-Start de Novos Braços:** Ao introduzir novas ofertas no catálogo, o modelo entra em fase de exploração intensiva local para aquele braço, o que pode temporariamente degradar a conversão geral do cliente que receber a oferta antes de o modelo acumular dados suficientes.
