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

### 4. Ingestão de Dados (Kaggle)
O projeto requer a base de dados original para alimentar o pipeline de processamento.
1. Acesse o link: https://www.kaggle.com/datasets/aguado/telemarketing-jyb-dataset/data
2. Baixe os arquivos `train.csv` e `test.csv`.
3. Coloque ambos os arquivos no diretório `data/kaggle/` do projeto.
4. Execute o script de ingestão para padronizar os cabeçalhos e limpar identificadores sem valor preditivo:
    python data/make_dataset.py

Os arquivos limpos (`train_clean.csv` e `test_clean.csv`) serão gerados automaticamente na pasta `data/processed/`.

### 5. Geração de Enriquecimento Sintético
Após gerar os dados processados, crie os artefatos sintéticos executando:
    python data/synthetic_enrichment/generate_synthetic_data.py --seed 42

Isso cria:
- `data/synthetic_enrichment/arm_catalog.csv`
- `data/synthetic_enrichment/contexts.csv`
- `data/synthetic_enrichment/impressions.csv`

### 6. Validação Inicial
Execute a suíte de testes para garantir que o ambiente está configurado corretamente:
    pytest

## Mapa de Pastas
* data/: Diretório de dados contendo as bases do Kaggle originais, processadas, enriquecimento sintético e o golden set (arquivos de dados brutos são ignorados pelo repositório).
* docs/: Documentação de arquitetura Azure, governança e planos LGPD.
* notebooks/: Análise Exploratória de Dados (EDA) e protótipos de simulação.
* reports/: Relatórios técnicos de geração de dados e experimentação.
* src/: Código-fonte principal, incluindo API, modelos algorítmicos e integração com LLM.
* tests/: Suíte de testes unitários e de integração (pytest).

## Limitações Conhecidas
* O projeto utiliza dados sintéticos baseados em conjuntos públicos do Kaggle e não processa dados reais de clientes (PII).
* A simulação de delayed rewards (recompensas atrasadas) assume hipóteses e sementes fixas que estão documentadas no relatório de geração de dados.
* A plataforma foi desenvolvida para fins de experimentação em ambiente simulado e não atende, no seu estado atual, aos requisitos de latência para um sistema bancário de alta frequência em tempo real.