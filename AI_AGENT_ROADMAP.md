# AI Agent Roadmap: Trading Bot Evolution (V2 to Institutional)

## Status Atual: 🟡 Em Desenvolvimento
**Objetivo:** Elevar a estratégia de "Varejo Direcional" para "Edge Estrutural" com foco em derivativos institucionais na Bybit.

---

## 🛠 Fase 1: Arquitetura de Dados e Estacionaridade
*Foco: Garantir que os dados alimentados no modelo ML sejam matematicamente válidos.*

- [x] **Estacionaridade:** Substituir preços nominais e médias móveis puras por **Retornos Logarítmicos** (`Log Returns`) no `ml/features.py`.
- [x] **Memória de Longo Prazo:** Implementar **Diferenciação Fracionária** (`Fractional Differencing`) para manter o sinal preditivo sem violar a estacionaridade.
- [x] **Infraestrutura Real-time:** Migrar a coleta de dados de REST Polling para **WebSockets** (Bybit V5) para reduzir latência de execução.

## 📊 Fase 2: Inteligência de Derivativos
*Foco: Parar de olhar apenas para o preço e começar a olhar para a mecânica do contrato.*

- [x] **Módulo de Funding:** Adicionar monitoramento de `Funding Rate` e `Predicted Funding`.
- [x] **Filtro de Custo de Carrego:** Implementar lógica no `strategy.py` para evitar Longs quando o funding está excessivamente alto (custo proibitivo).
- [x] **Arbitragem Interna:** Monitorar o `Premium Index` para identificar desvios entre o preço spot e o perpétuo.

## 🌊 Fase 3: Order Flow e Liquidez
*Foco: Identificar onde o "Smart Money" está se posicionando.*

- [x] **CVD (Cumulative Volume Delta):** Implementar o cálculo de delta de volume (agressão de compra vs venda).
- [x] **Open Interest (OI):** Integrar variações no OI para confirmar se movimentos de preço são sustentados por capital novo ou fechamento de posições.
- [x] **Liquidation Tracker:** Criar sinais baseados em cascatas de liquidação (oportunidades de reversão à média).

## ⚡ Fase 4: Microestrutura e Execução Profissional
*Foco: Reduzir custos operacionais que destroem o PnL.*

- [x] **Execução Maker:** Implementar lógica de ordens `Limit` com monitoramento de profundidade do book para capturar o spread e evitar `Taker Fees`.
- [x] **Dynamic Risk Management:** Alterar o `risk_management.py` para que o Stop Loss seja baseado em **Múltiplos de ATR** (ajuste dinâmico à volatilidade).
- [x] **Slippage Control:** Adicionar um threshold máximo de slippage aceitável antes de cancelar ou reprecificar uma ordem.

## 🔬 Fase 5: Validação e Robustez
*Foco: Garantir que o lucro no papel se traduza em lucro na conta.*

- [ ] **Backtest de Alta Fidelidade:** Incluir taxas reais (0.02% Maker / 0.05% Taker) e custos de funding acumulados no simulador.
- [ ] **Walk-Forward Analysis:** Implementar janelas de treino e teste móveis para garantir que o modelo se adapta a novos regimes de mercado.
- [ ] **Dashboard de Métricas:** Centralizar Sharpe Ratio, Sortino e Max Drawdown em tempo real no `monitoring_dashboard.py`.

---

## 📝 Notas para o Agente de IA
1. **Iteração:** Atualize este arquivo `.md` marcando os itens concluídos `[x]` após cada deploy ou correção bem-sucedida.
2. **Prioridade:** Siga a ordem das Fases. Não tente otimizar a execução (Fase 4) se os dados base (Fase 1) ainda estiverem gerando sinais falsos.
3. **Logs:** Sempre que concluir uma tarefa, anexe o impacto observado (ex: "Fase 1 concluída: Erro de estacionaridade reduzido em 40%").

---

## Logs de Execução
- **2026-05-12 - Fase 1 concluída:** `ml/features.py` passou a produzir retornos logarítmicos, distâncias logarítmicas das EMAs, ATR percentual e diferenciação fracionária; `ml.config.FEATURES` removeu `close`/EMAs nominais do input do ML. Validação: `py_compile` passou e amostra sintética gerou 31 features estacionárias.
- **2026-05-12 - Fase 1 Infra Real-time concluída:** o loop principal agora alimenta uma cache OHLCV por WebSocket Bybit V5 e usa REST apenas para aquecimento/fallback inicial.
- **2026-05-12 - Fase 2 concluída:** adicionados funding, predicted funding e premium basis ao pipeline; `strategy.py` bloqueia longs quando o custo de carrego excede os limites configurados. Validação: teste sintético bloqueou sinal `buy` com funding de 0.10%.
- **2026-05-12 - Fase 3 concluída:** adicionados CVD por tape de trades, variação de Open Interest e tracker de liquidações com sinal de reversão; `strategy.py` passou a pontuar fluxo, OI e cascatas. Validação: teste sintético calculou CVD 0.3333 e anexou sinais ao score.
- **2026-05-12 - Fase 4 concluída:** execução de entrada migrou para Limit/PostOnly com checagem de book, threshold de slippage de 0.08% e cancelamento defensivo de ordem não preenchida; `risk_management.py` centraliza TP/SL por múltiplos de ATR. Validação: teste sintético retornou TP 110.0 e SL 96.0 para Buy com ATR 5 em regime trending.
