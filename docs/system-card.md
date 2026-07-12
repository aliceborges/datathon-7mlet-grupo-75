# System Card - Plataforma de Experimentação Adaptativa

Este System Card descreve a arquitetura geral, fluxo de decisão, guardrails e governança da plataforma de experimentação adaptativa.

## Escopo do Sistema
A plataforma gerencia a alocação dinâmica de ofertas financeiras digitais para os clientes elegíveis, balanceando exploração (descoberta de novas preferências) e explotação (aproveitamento das melhores ofertas históricas). Além disso, inclui um assistente baseado em LLM (Large Language Model) integrado via arquitetura ReAct para responder dúvidas de usuários e explicar decisões tomadas.

---

## Fluxo de Decisão

O fluxo de decisão do sistema segue as seguintes etapas:
1. **Requisição de Recomendação:** O canal digital chama o endpoint `/predict` passando o `CustomerContext`.
2. **Filtro de Suitability (Guardrails):** O sistema aplica regras rígidas de segurança antes de processar os algoritmos adaptativos (ex: bloqueia ofertas de empréstimos para clientes inadimplentes ou negativados).
3. **Seleção por MAB:** A política carregada (Thompson Sampling ou UCB) escolhe o melhor braço elegível e calcula a ordenação das alternativas.
4. **Log de Auditoria:** Detalhes da decisão (ID da transação, versão da política, braço selecionado, reason codes e score) são gravados em arquivo JSONL assíncrono local de forma imutável.
5. **Retorno da Resposta:** A oferta selecionada é apresentada no canal.

---

## Cenários de Risco e Contramedidas

### 1. Reward Hacking
* **Risco:** O modelo pode aprender a recomendar excessivamente uma oferta de baixo valor agregado mas de conversão instantânea (ex: brindes ou ofertas gratuitas), comprometendo o ROI da instituição.
* **Guardrail:** Limitação do volume diário de conversão por braço no catálogo e balanceamento de recompensas ponderadas pelo valor financeiro esperado, e não apenas pela taxa de clique simples.

### 2. Manipulação de Contexto
* **Risco:** Chamadas maliciosas com atributos adulterados (ex: saldo artificialmente inflado) para forçar o MAB a exibir ofertas premium.
* **Guardrail:** Validação de schemas rígida via Pydantic na entrada e sanitização de dados integrando com barramentos corporativos de dados confiáveis (Backend-of-Frontend).

### 3. Abuso do Assistente LLM
* **Risco:** Prompt Injection ou jailbreak para fazer o assistente responder tópicos fora do escopo do Datathon ou violar regras comerciais internas.
* **Guardrail:** Utilização de `InputGuardrail` na API que valida a pergunta do usuário contra políticas de segurança antes de enviá-la para o Azure OpenAI. Filtros de output barram respostas contendo termos proibidos.

### 4. Violação de Suitability
* **Risco:** Recomendar crédito para menores de 18 anos ou para perfis de alta vulnerabilidade financeira.
* **Guardrail:** Validação forçada na camada de pre-routing. Menores de 18 anos são rejeitados de imediato no schema de dados do Pydantic (`CustomerContext.age >= 18`).

---

## Plano de Monitoramento

- **Observabilidade de Negócio:** Painel com taxa de conversão diária por braço, arrependimento acumulado (regret) e receita líquida gerada.
- **Detecção de Drift de Dados:** Execução semanal do script de cálculo de PSI (Population Stability Index) comparando o perfil de entrada atual contra o baseline de treinamento. Alertas são disparados se `PSI >= 0.10` (drift moderado) ou `PSI >= 0.20` (drift severo, disparando retreino automático).
- **Monitoramento de Latência:** Telemetria via endpoints Prometheus `/metrics` medindo latência média dos endpoints `/predict` e `/agent`.
