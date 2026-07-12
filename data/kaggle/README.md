# Base Kaggle

## Fonte
Dataset de telemarketing usado na etapa 1: https://www.kaggle.com/datasets/aguado/telemarketing-jyb-dataset/data

## Contrato da Camada Processada
O script [data/make_dataset.py](../make_dataset.py) consome `train.csv` e `test.csv` em `data/kaggle/`, padroniza os nomes das colunas (`emp.var.rate` -> `emp_var_rate`), remove identificadores e grava a camada processada em `data/processed/`.

### Colunas removidas
`id`, `duration` e `Unnamed: 0` são descartadas do fluxo de decisão. O script valida o schema final para impedir colunas fora do contrato.

### Schema esperado
`train_clean.csv` contém as 19 variáveis explicativas abaixo e o alvo `y`.

`age`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `campaign`, `pdays`, `previous`, `poutcome`, `emp_var_rate`, `cons_price_idx`, `cons_conf_idx`, `euribor3m`, `nr_employed`, `y`

`test_clean.csv` contém as mesmas variáveis explicativas sem `y`.

## Limitações
Esta pasta guarda apenas a base bruta e a referência de contrato. Qualquer nova extração Kaggle deve preservar o mesmo conjunto de colunas ou atualizar este documento junto com os testes da camada processada.