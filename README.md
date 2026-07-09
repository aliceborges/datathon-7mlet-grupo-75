# Datathon 7MLET - Grupo 75

## Visão do Problema
Plataforma de experimentação adaptativa baseada em Multi-Armed Bandit (MAB) para recomendação de ofertas em canais digitais de uma instituição financeira. Este projeto substitui regras estáticas e testes A/B longos por uma política que equilibra exploração e explotação, maximizando a conversão com segurança e governança.

## Escopo e Escolhas de Design
* **Algoritmo:** Implementação de política adaptativa (Thompson Sampling ou Nilos-UCB) para decisão de ofertas.
* **Arquitetura:** Microsserviços com API para servir a decisão, gerando logs auditáveis.
* **Governança:** Documentação estruturada com Model Card, System Card e conformidade LGPD.
* **IA/LLM:** Assistente integrado para resumir experimentos e explicar decisões algorítmicas.
* **Cloud:** Arquitetura-alvo desenhada exclusivamente para o ecossistema Azure.

## Instruções de Execução Local

### 1. Pré-requisitos
* Python 3.10 ou superior
* Git

### 2. Instalação
Clone o repositório e navegue até a pasta:
    git clone https://github.com/seu-usuario/datathon-7mlet-grupo-XX.git
    cd datathon-7mlet-grupo-XX

Crie e ative o ambiente virtual:
    # Windows (PowerShell)
    python -m venv .venv
    .venv\Scripts\Activate.ps1

    # Linux/macOS
    python -m venv .venv
    source .venv/bin/activate

Instale os pacotes definidos no `pyproject.toml` em modo editável:

    pip install -e .[dev]

Esse comando instala as dependências de runtime e as dependências de desenvolvimento declaradas no `pyproject.toml`.

### 3. Variáveis de Ambiente
Copie o arquivo de configuração de exemplo e preencha as credenciais necessárias (nunca comite o arquivo .env com dados reais):
    # Windows
    copy .env.example .env

    # Linux/macOS
    cp .env.example .env

As variáveis principais usadas pelo projeto são `API_HOST`, `API_PORT`, `LOG_LEVEL`, `AUDIT_LOG_PATH`, `MLFLOW_TRACKING_URI`, `CHAMPION_MODEL_NAME`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION` e `LLM_TEMPERATURE`.

### 4. Subida da API
Se quiser rodar a API localmente, exporte as variáveis necessárias e inicie o servidor:

```powershell
$env:AZURE_OPENAI_API_KEY = "..."
$env:AZURE_OPENAI_ENDPOINT = "https://<seu-recurso>.openai.azure.com"
$env:AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"
$env:AZURE_OPENAI_API_VERSION = "2024-08-01-preview"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:CHAMPION_MODEL_NAME = "<nome-do-modelo>"
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Sem credenciais Azure OpenAI, o `/predict` continua funcionando com o stub de política, mas o `/agent` responde `503`.

### 5. Ingestão de Dados (Kaggle)
O projeto requer a base de dados original para alimentar o pipeline de processamento.
1. Acesse o link: https://www.kaggle.com/datasets/aguado/telemarketing-jyb-dataset/data
2. Baixe os arquivos `train.csv` e `test.csv`.
3. Coloque ambos os arquivos no diretório `data/kaggle/` do projeto.
4. Execute o script de ingestão para padronizar os cabeçalhos e limpar identificadores sem valor preditivo:
    python data/make_dataset.py

Os arquivos limpos (`train_clean.csv` e `test_clean.csv`) serão gerados automaticamente na pasta `data/processed/`.

### 6. Validação Inicial
Execute os testes de fumaça e da API para confirmar que o ambiente está consistente:

    pytest tests/test_smoke.py
    pytest tests/test_api.py -q

Se preferir uma execução mais ampla, rode `pytest` na raiz do repositório.

## Mapa de Pastas
* data/: Bases do Kaggle originais, camada processada, golden set e corpus RAG; `data/synthetic_enrichment/` fica reservado para artefatos sintéticos das etapas seguintes.
* docs/: Documentação de API, arquitetura Azure, benchmark e MLOps.
* notebooks/: Análise Exploratória de Dados (EDA).
* src/: Código-fonte principal, incluindo API, agentes, modelos e camadas de segurança/monitoramento.
* tasks/: Plano incremental do projeto e critérios de cada etapa.
* tests/: Suíte de testes unitários e de integração (pytest).

## Limitações Conhecidas
* O projeto utiliza dados sintéticos baseados em conjuntos públicos do Kaggle e não processa dados reais de clientes (PII).
* A API do agente depende de Azure OpenAI; sem essas credenciais, apenas o stub de recomendação fica disponível.
* A camada processada do Kaggle remove `id`, `duration` e outras colunas auxiliares, e o preprocessing valida o schema final para evitar vazamento temporal.
* A simulação de delayed rewards (recompensas atrasadas) assume hipóteses e sementes fixas documentadas nas etapas de experimento.
* A plataforma foi desenvolvida para fins de experimentação em ambiente simulado e não atende, no seu estado atual, aos requisitos de latência para um sistema bancário de alta frequência em tempo real.