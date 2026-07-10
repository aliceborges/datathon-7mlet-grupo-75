# Enriquecimento Sintetico

Esta camada gera a base de experimentacao adaptativa a partir do recorte Kaggle processado, sem reutilizar dados reais de clientes. O gerador produz tres artefatos separados e reproduziveis por seed:

- `offer_catalog.csv`
- `offer_events.csv`
- `delayed_rewards.csv`

## Como gerar

```bash
python -m data.synthetic_enrichment.generator --input data/processed/train_clean.csv --output data/synthetic_enrichment/generated --seed 75
```

## Semente, horizonte e regras

- Semente padrao: `75`.
- Horizonte temporal padrao: `14` dias.
- Recompensas atrasadas: apenas eventos convertidos podem gerar recompensa, com atraso inteiro entre `1` e `7` dias, sempre dentro do horizonte de simulacao.
- Catalogo de ofertas: cada oferta recebe janela de validade, faixa etaria alvo, `target_job`, canal e retorno esperado.
- Eventos: cada linha representa uma exposicao a oferta, com contexto sintetico inferido da base processada e uma probabilidade de conversao deterministica por seed.

## Schema dos arquivos

### offer_catalog.csv

- `offer_id`: identificador unico da oferta.
- `offer_type`: tipo da oferta sintetica.
- `channel`: canal de entrega associado.
- `reward_type`: tipo de recompensa.
- `reward_value`: valor monetario ou equivalente da recompensa.
- `discount_rate`: taxa de desconto simulada.
- `target_job`: ocupacao alvo.
- `target_age_min`: idade minima do publico alvo.
- `target_age_max`: idade maxima do publico alvo.
- `valid_from`: inicio da validade da oferta.
- `valid_until`: fim da validade da oferta.
- `expected_acceptance_rate`: taxa de aceitacao esperada.
- `expected_reward_delay_days`: atraso esperado da recompensa.

### offer_events.csv

- `event_id`: identificador unico do evento.
- `source_row_id`: indice da linha de origem usada como contexto sintetico.
- `client_id`: identificador sintetico do cliente.
- `offer_id`: oferta exposta no evento.
- `offer_type`: tipo da oferta.
- `channel`: canal de entrega.
- `event_timestamp`: instante da exposicao.
- `age`: idade sintetica do cliente.
- `age_band`: faixa etaria derivada.
- `job`: profissao sintetica.
- `marital`: estado civil sintetico.
- `education`: escolaridade sintetica.
- `contact`: canal de contato sintetico.
- `source_y`: rotulo de origem quando disponivel.
- `propensity_score`: probabilidade simulada de aceitacao.
- `accepted_offer`: indica aceitacao da oferta.
- `converted_with_delay`: indica conversao com recompensa atrasada.
- `reward_delay_days`: atraso observado para a recompensa.

### delayed_rewards.csv

- `reward_id`: identificador unico da recompensa.
- `event_id`: evento de origem da recompensa.
- `client_id`: cliente associado.
- `offer_id`: oferta associada.
- `reward_timestamp`: instante em que a recompensa se materializa.
- `delay_days`: atraso em dias entre evento e recompensa.
- `reward_value`: valor da recompensa.
- `reward_type`: tipo da recompensa.
- `reward_reason`: motivo da recompensa atrasada.

## Observabilidade e auditoria

- Cada recompensa tem `event_id` correspondente em `offer_events.csv`.
- `reward_timestamp` sempre e posterior ao `event_timestamp` do evento gerador.
- O gerador nao depende da base original de clientes; ele usa apenas o recorte Kaggle processado como referencia de distribuicoes e segmentos.