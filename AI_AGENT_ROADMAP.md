# AI Agent Roadmap: Institutional Quant Pipeline (V4 - Final)

VERSION V4 - Update

## Status Atual: Em Execucao Tecnica
**Objetivo:** Evoluir um prototipo quantitativo com vazamento de dados para um pipeline HFT / intermediario de nivel institucional, focado em integridade de dados, gestao rigorosa de risco e descoberta de alpha nao-linear.

---

### Fase 1: Data Integrity & Erradicacao de Vies
*Onde a maioria falha. Sem higiene de dados, nao ha modelo valido.*
- [x] Implementar `.shift(1)` mandatorio em TODAS as features exogenas (Sentimento, Funding, OI) para garantir defasagem de 1 periodo.
- [x] Limpeza total do `dataset.csv` e `ml/dataset.csv` (remover dados contaminados por Look-Ahead Bias).
- [x] Refatorar o pipeline de coleta via WebSockets Bybit V5 para garantir timestamps precisos ao nivel do milissegundo.

### Fase 2: Risk Survival (Sobrevivencia Basica)
*Proteger o capital enquanto o sistema nao prova seu valor estatistico em Forward Testing.*
- [x] Remocao de Lotes Fixos (Hardcoded Sizing de 100 USDT).
- [x] Posicionamento de Risco Fracionario Fixo: Arriscar estritamente 0.5% a 1% do patrimonio por trade.
- [x] Implementar Trava de Drawdown Diario (Circuit Breaker: pausa de 24h se atingir -4% PnL no dia).
- [x] Gestao de Risco baseada em ATR Dinamico para calibrar a distancia do Stop Loss.

### Fase 3: Market Regime Engine
*O Filtro Mestre. O contexto precede o sinal.*
- [x] Implementar classificador de regime de mercado (Tendencia vs Lateralizacao / Volatilidade Alta vs Baixa).
- [x] Bloquear execucoes de setups de Momentum caso o regime classificado seja de compressao/range.

### Fase 4: Microestrutura Dinamica e Contexto
*Ler as intencoes institucionais atraves do fluxo de derivativos.*
- [x] Calculo de **Open Interest Delta** (Aceleracao da alavancagem, nao o valor nominal).
- [x] Calculo de **Funding Rate Delta** (Velocidade de mudanca do premio de risco).
- [x] Mapeamento de Densidade de Liquidacoes (Liquidation Clusters).
- [x] Integrar **Anchored VWAP** e Session Volume Profiles para determinar zonas de valor institucional.

### Fase 4.5: Feature Validation Layer (O "Filtro Institucional")
*A alfandega do codigo. Nenhuma feature entra em producao sem:*
- [x] 1. Melhorar pelo menos um KPI principal de performance (Sharpe Ratio, Profit Factor ou Max Drawdown).
- [x] 2. Demonstrar *Incremental Information Gain* atraves de SHAP Values, Permutation Importance ou Estudo de Ablacao (Drop-Column).
- [x] 3. Nao apresentar colinearidade excessiva absoluta com o conjunto de features base.
- [x] 4. Manter robustez estatistica em validacao *Walk-Forward Analysis*.

### Fase 5: Glass-Box ML (Machine Learning Auditavel)
*Abrir a caixa preta do algoritmo LightGBM para entender a alocacao de risco.*
- [x] Integrar **Walk-Forward Analysis (WFA)** continuo para re-treino em janelas deslizantes temporais (mitigacao de Concept Drift).
- [x] Analisar e exportar **SHAP Values** para monitorar a evolucao temporal das features mais relevantes.
- [x] Calibracao de Probabilidade (Platt Scaling): Garantir que uma inferencia de 70% de confianca pelo LightGBM reflita historicamente ~70% de Win Rate no mercado.

### Fase 6: Alpha Discovery
*Busca continua por ineficiencias de curtissimo prazo.*
- [x] Desenvolver rastreador de **CVD Spot vs CVD Perp** (Detectar capital de investimento real vs especulacao alavancada).
- [x] Estudo quantitativo de desvios padrao extremos da VWAP e reversao a media.

### Fase 7: Alpha Portfolio Construction
*Pensamento de fundo quantitativo: Distribuicao de risco em portfolio de sinais.*
- [x] Implementar arquitetura de Ensemble de modelos (Sinais Fracos Ortogonais = Sinal Direcional Forte).
- [x] Avaliacao dinamica de peso de estrategia: Reduzir exposicao a sub-sistemas em drawdown e focar naqueles com maior Sharpe recente.

### Fase 8: Kelly Fractional + Capital Scaling
*Escalar a agressividade com matematica comprovada.*
- [x] Transicao segura para Kelly Fracionario (Ex: Half-Kelly) baseado na Expectancia e Variancia comprovadas pelo bot ao vivo.
- [x] Otimizacao ativa de latencia e transicao estrategica para Maker Execution (Limit Orders passivas) focando na captacao de Rebates (taxas negativas).
