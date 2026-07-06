# Glossário de features do dataset

## age

Idade do cliente em anos. Vai de 18 a 95 no dataset original.
Faixa sênior (>= 60) tem propensão maior a `loan_payroll` e `savings_premium`.

## job

Categoria ocupacional. Valores comuns: admin, blue-collar, technician, retired, student, services, management.
Clientes em `retired` ou `management` têm balance médio mais alto.

## marital

Estado civil. Valores: married, single, divorced.
Não é forte preditor isolado mas combinado com housing/loan ajuda na segmentação familiar.

## education

Nível educacional. Valores: primary, secondary, tertiary, unknown.
Tertiary correlaciona com maior taxa de aceite de `credit_card_gold`.

## balance

Saldo médio anual em euros. Pode ser negativo (cheque especial usado).
Cliente com balance > 5000 tem mais chance de aceitar `savings_premium`.

## housing

Indica se o cliente tem financiamento habitacional ativo.
True desbloqueia recomendação de `insurance_basic` (sinergia residencial).

## loan

Indica se o cliente já tem empréstimo pessoal ativo no banco.
True bloqueia recomendação de `loan_personal` para evitar sobreendividamento.

## contact

Canal usado no último contato. Valores: cellular, telephone, unknown.
Cellular tem taxa de resposta significativamente maior que telephone.
