AGENT PLAN
DIRECTION ALPHA DISCOVERY

Objetivo:

Descobrir qual fonte de dados realmente contém direção.

Não otimizar.

Não adicionar features aleatórias.

Não mexer em hiperparâmetros.

Somente descobrir:

Onde mora o Alpha?
EXPERIMENTO A
Direction From Microstructure

Hipótese:

Fluxo agressivo
precede direção.
Dataset

Manter Triple Barrier.

Mesmo target.

Mesmo TimeSeriesSplit.

Mesmo Walk Forward.

Features Permitidas

Somente:

cvd

cvd_delta

cvd_acceleration

oi_delta

oi_acceleration

funding_delta

funding_acceleration

liquidation_density

premium_delta
Features Proibidas
RSI

MACD

EMA

Bollinger

Sentiment

Hour

Day

VWAP
Métricas

Reportar:

ROC AUC

PR AUC

Brier

LogLoss

Calibration

SHAP
Perguntas
Microestrutura sozinha
supera AUC 0.54?
Qual feature explica
mais de 10% do ganho?
Existe uma feature dominante?
Critério

Aprovado:

AUC >= 0.56

Excelente:

AUC >= 0.58
EXPERIMENTO B
Regime Specific Models

Hipótese:

Misturar regimes
destrói Alpha.
Separar dataset

Trend

Range

High Vol

Low Vol

Treinar:

model_trend.pkl

model_range.pkl

model_high_vol.pkl

model_low_vol.pkl

Cada modelo recebe:

mesmas features

mesmo target
Comparação

Comparar:

Modelo Único
vs
Modelos Específicos
Perguntas
Qual regime possui maior edge?
Qual regime é inviável?
Existe regime
com AUC > 0.60?
Critério

Aprovado:

Se qualquer regime produzir:

AUC >= 0.60

Porque significa:

Alpha localizado.
EXPERIMENTO C
Meta Labeling

Inspirado em:

Marcos López de Prado

Hipótese:

O modelo já sabe gerar sinais.

Mas não sabe quais sinais ignorar.
Inputs

Tradeability

Direction

Novo Target

Trade realizado

↓

Foi lucrativo?

Label:

1 = trade vencedor

0 = trade perdedor
Features

Somente:

tradeability_probability

direction_probability

regime

atr_pct

cvd

oi_delta

funding_delta
Objetivo

Não prever mercado.

Prever:

Vale a pena acreditar
neste sinal?
Métricas

Reportar:

ROC AUC

Profit Factor

Expectancy

Sharpe

Drawdown
Perguntas

O Meta Model consegue:

Eliminar
20% dos piores trades?
Melhorar PF?
Melhorar Expectancy?
Critério

Aprovado:

PF sobe

e

DD cai
Regra de Ouro

Antes de qualquer deploy:

O agente deve gerar:

direction_alpha_report.md

com a tabela final:

Experimento	AUC	PF	Sharpe	Status
Microstructure	?	?	?	?
Regime Models	?	?	?	?
Meta Label	?	?	?	?