# Plano de Conformidade LGPD - Plataforma de Experimentação Adaptativa

Este plano formaliza as diretrizes adotadas para garantir a conformidade do processamento de dados do Datathon com a Lei Geral de Proteção de Dados (LGPD).

## Bases Legais e Finalidades

Todo o ciclo de vida dos dados processados na plataforma apoia-se nas seguintes bases legais:
1. **Consentimento (Art. 7º, I, LGPD):** Coletado digitalmente quando o cliente aceita receber ofertas personalizadas em canais digitais.
2. **Execução de Contrato (Art. 7º, V, LGPD):** Quando o cliente clica para adquirir a oferta recomendada (ex: contratação de empréstimo ou abertura de conta premium).
3. **Legítimo Interesse (Art. 7º, IX, LGPD):** Para a melhoria contínua dos modelos MAB e personalização de serviços digitais, respeitando a legítima expectativa do cliente e sem violar seus direitos fundamentais.

---

## Princípios de Minimização e Retenção

- **Minimização de Dados:** Apenas atributos estritamente necessários para a personalização das ofertas são integrados ao modelo (idade, profissão, saldo de conta, e relacionamento financeiro básico). Dados sensíveis (origem racial, convicções religiosas, dados de saúde, ou dados de menores de idade) são completamente omitidos e filtrados na origem.
- **Ciclo de Retenção:**
  - **Logs de Recomendação (`/predict`):** Retidos por 90 dias para fins de auditoria interna e depuração de erros, sendo arquivados após esse período.
  - **Histórico de Conversões (Recompensas):** Armazenados de forma agregada para re-treino de novas políticas. Os dados individualizados são anonimizados após 365 dias da finalização da campanha.
  - **Perguntas do Assistente LLM (`/agent`):** O texto livre inserido no assistente é descartado imediatamente após o término da sessão do usuário e não é persistido para re-treino de LLM corporativo.

---

## Mapeamento de Atributos e Atributos Protegidos

| Atributo | Categoria | Finalidade | Tratamento de Proteção |
| :--- | :--- | :--- | :--- |
| `customer_id` | Identificador Indireto | Associação de recompensas atrasadas. | Pseudonimizado (hash UUID aleatório no serving). |
| `age` | Dado Pessoal Comum | Direcionamento etário e suitability legal (bloqueio de menores). | Agrupado em faixas etárias (`age_band`) para análise e modelagem offline. |
| `job` | Dado Pessoal Comum | Personalização por afinidade profissional. | Normalizado contra um dicionário fixo de profissões. |
| `balance` | Dado Pessoal Comum | Proteção de suitability (evitar superendividamento). | Ocultado em relatórios agregados e arredondado em faixas de saldo. |

---

## Política de Logs e Telemetria

Os logs de auditoria estruturados gerados pela plataforma seguem diretrizes rígidas de segurança:
1. **Anonimato de Respostas:** O payload completo das mensagens textuais geradas pela LLM não é gravado no log corporativo para evitar vazamento de dados inseridos inadvertidamente pelo usuário.
2. **Imutabilidade:** O arquivo de logs JSONL local é configurado para gravação "append-only".
3. **Exclusão de IP e Dispositivos:** IP do usuário e cabeçalhos sensíveis do navegador (User-Agent completo) são expurgados da telemetria de produção.

---

## Plano de Resposta a Incidentes

Em caso de suspeita de vazamento de dados ou incidentes de segurança:
1. **Identificação e Isolamento:** O time de segurança da informação isola as instâncias afetadas e rotaciona todas as chaves de acesso armazenadas no Azure Key Vault.
2. **Análise de Impacto:** Determinação se houve vazamento de dados pseudonimizados ou se chaves de criptografia foram expostas.
3. **Notificação:** Comunicação imediata à ANPD (Autoridade Nacional de Proteção de Dados) e aos titulares afetados em até 48 horas após a confirmação do incidente, contendo a descrição da natureza do incidente, medidas tomadas e recomendações de segurança.
