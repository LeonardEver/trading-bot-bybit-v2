AGENT PLAN — EXPERIMENTO A
Tradeability Signal Frequency Audit
Objetivo

Validar se o modelo Tradeability (AUC 0.7963) gera sinais suficientes para uso operacional.

O problema a responder:

O modelo possui edge estatístico.
Mas esse edge aparece frequentemente
o suficiente para ser monetizado?
Inputs

Utilizar:

ml/model_tradeability.pkl
ml/dataset_tradeability.csv

Mesma pipeline de inferência utilizada em produção.

Nenhuma feature nova.

Nenhum retreino.

Análises obrigatórias
1. Distribuição completa das probabilidades

Gerar:

Min
Max
Mean
Median
Std
2. Percentis

Calcular:

P50
P75
P80
P85
P90
P95
P97
P99
3. Frequência de sinais

Para thresholds:

0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
0.90

Reportar:

Total sinais
% dataset
Sinais por dia
Sinais por semana
4. Tradeability Lift

Comparar:

WR Global
vs
WR acima do threshold

para cada threshold.

Exemplo:

Threshold 0.50
WR = X

Threshold 0.60
WR = Y

Threshold 0.70
WR = Z
5. Calibration Audit

Gerar:

Reliability Curve
ECE
Brier Score

Separadamente para:

Top 20%
Top 10%
Top 5%

das previsões.

6. Score Deciles

Dividir previsões em:

Decil 1
Decil 2
...
Decil 10

Medir:

Hit Rate
Tradeability Rate
Critério de aprovação

Aprovado se:

Threshold operacional
produzir

>= 5 trades/dia

e

Lift estatístico relevante

Exemplo:

Base = 67%

Threshold = 80%

Resultado = 85%
Entregáveis
tradeability_frequency_report.md

signal_distribution.png

threshold_analysis.csv

decile_analysis.csv
AGENT PLAN — EXPERIMENTO B
Economic Validation Backtest

Executar apenas após concluir o Experimento A.

Objetivo

Descobrir se o modelo Tradeability melhora o resultado financeiro.

Não queremos saber:

AUC
Accuracy
Precision

Queremos saber:

Dinheiro
Estratégia Base

Usar exatamente a mesma estratégia atual.

Sem alterações.

Sem novas features.

Sem novos filtros.

Cenário Controle
Sem Tradeability Gate

Estratégia original.

Cenário 1
Tradeability > Threshold_Operacional

Threshold definido pelo Experimento A.

Cenário 2
Tradeability > Threshold_Agressivo

Threshold superior.

Cenário 3
Tradeability > Threshold_Extremo

Top 5% dos sinais.

Métricas obrigatórias
Profit Factor
Sharpe
Sortino
Win Rate
Expectancy
Max Drawdown
Calmar Ratio
Recovery Factor
Total Trades
Trades por mês
Breakdown por regime

Separar resultados em:

Trend
Range
High Volatility
Low Volatility
Breakdown por score

Separar:

0.50-0.60

0.60-0.70

0.70-0.80

0.80+
Perguntas obrigatórias

O Agent deve responder explicitamente:

Q1
O Tradeability melhora Profit Factor?
Q2
O Tradeability reduz Drawdown?
Q3
O Tradeability aumenta Expectancy?
Q4
Qual threshold maximiza
retorno ajustado ao risco?
Q5
O modelo gera sinais suficientes
para um bot que busca
10-15 trades/dia?