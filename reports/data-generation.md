# Relatório de Geração de Dados Sintéticos

## Processo de Geração

O processo de enriquecimento sintético de dados foi projetado para construir uma camada de experimentação adaptativa realista sobre o dataset de marketing bancário selecionado do Kaggle (`telemarketing-jyb-dataset`). A geração ocorre de maneira totalmente reprodutível, baseada em sementes aleatórias controladas, e separa fisicamente a base original das interações e recompensas do ecossistema de Multi-Armed Bandits (MAB).

O processo é composto pelas seguintes etapas:
1. **Definição do Catálogo de Ofertas:** Criação de um conjunto fixo de braços/ofertas (`offer_catalog.csv`), cada um com atributos próprios como canal, valor da recompensa, tipo de oferta (cashback, voucher, etc.) e público-alvo sintético (idade e profissão).
2. **Simulação de Eventos de Exposição (Impressões):** Amostragem de registros contextuais da base do Kaggle limpa para simular visitas ou interações de clientes (`offer_events.csv`). Para cada evento, é selecionada uma oferta baseando-se em afinidades de contexto e probabilidade histórica.
3. **Modelagem de Recompensas Atrasadas (Delayed Rewards):** Simulação de conversões tardias com base em score de propensão. Se a oferta for aceita, uma recompensa é disparada após um atraso aleatório de $1$ a $7$ dias, respeitando o horizonte de simulação e sendo registrada em `delayed_rewards.csv`.

---

## Sementes e Parâmetros Utilizados

Para garantir a total reprodutibilidade do pipeline, os seguintes parâmetros foram configurados e fixados:
- **Semente Aleatória:** `75` (controla o gerador de números aleatórios `numpy.random.default_rng(75)`).
- **Horizonte Temporal:** `14` dias (período de duração da campanha simulada).
- **Tamanho do Catálogo:** `12` ofertas exclusivas criadas sinteticamente.
- **Timestamp Inicial:** `2024-01-01` (início cronológico dos eventos).
- **Volume de Eventos:** Gerado com base no tamanho do dataset de treino processado ($n = 2 \times len(source\_df)$).

---

## Hipóteses de Comportamento

1. **Afinidade Contextual:** Presume-se que clientes convertem com maior probabilidade se a oferta pertencer à sua faixa etária alvo, ao seu canal de preferência ou se a sua profissão coincidir com o target da campanha.
2. **Propensão Base:** Cada oferta tem uma taxa de aceitação base definida no catálogo (por exemplo, entre 8% e 20%). O score final de propensão é ajustado de forma aditiva/subtrativa com base nas variáveis contextuais e de histórico (ex: penalização por campanhas repetidas).
3. **Conversão com Atraso (Delayed Conversion):** Assume-se que a conversão real não ocorre imediatamente no instante da impressão da oferta, mas segue uma distribuição de atraso uniforme/discreta entre 1 e 7 dias, refletindo o tempo de decisão de compra do cliente em canais digitais.

---

## Limitações e Riscos

- **Simulação Estática de Contexto:** Os dados contextuais dos clientes (idade, profissão) não mudam de comportamento ou de estado ao longo dos 14 dias de simulação.
- **Ausência de Feedback Dinâmico no Gerador:** O gerador de dados sintéticos simula a interação sem atualizar ativamente a política de exibição do MAB em tempo real (o MAB consome os eventos sequencialmente offline ou online, mas o gerador apenas cria o histórico factível).
- **Viés de Origem (Kaggle Dataset):** O comportamento de conversão ainda carrega correlações do dataset original do Kaggle (como a variável `y` que indica conversão no telemarketing original).
- **Simplificação de Delayed Rewards:** O atraso é simulado como um inteiro discreto em dias, enquanto em um cenário de produção real os atrasos seguem distribuições contínuas (ex: exponencial ou Weibull) e dependem de fatores operacionais de conciliação.
