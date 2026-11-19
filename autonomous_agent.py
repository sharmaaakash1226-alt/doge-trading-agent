"""
Autonomous Task & Conversational Trading Agent Engine.
Handles:
  1. Chat command parsing + professional trader dialogue (LLM or built-in Quant-NLP).
  2. Task-driven background worker (IDLE, RESEARCH_ONLY, TRACK_ONLY, AUTO_TRADE).
  3. Trade gap cooldown enforcement ("trade on sometimes gap").
  4. Real-time Stop-Loss & Target auto-exit tracking across all open paper positions.
"""

import asyncio
import re
import time
from typing import Any, Callable, Dict, List, Optional
import httpx

from ai_research import load_api_config, _resolve_llm_provider
from trading_memory import trading_memory


class AutonomousTradingAgent:
    def __init__(self):
        self.mode: str = "IDLE"  # IDLE | RESEARCH_ONLY | TRACK_ONLY | AUTO_TRADE
        # Strategy 1: SOL 8x Hourly Trend-Surfer with Progressive Tiered De-Leveraging
        self.symbols: List[str] = ["SOL/USDT"]
        self.timeframe: str = "1h"
        self.scan_interval_sec: int = 10
        self.trade_gap_sec: int = 0  # Re-research and enter immediately upon trade exit
        self.trade_amount_usdt: float = 100.0
        self.max_open_positions: int = 1  # Strictly 1 trade at a time
        self.min_confidence: int = 80  # Strict 80%+ high-conviction threshold

        # 8x Isolated Leverage Autonomous Engine Configuration
        self.leverage: float = 8.0  # 8x leverage
        self.capital_allocation_pct: float = 0.30  # 30% capital initial tier
        self.target_roi_pct: float = 0.60  # +60% profit target (+7.50% price move at 8x)
        self.initial_sl_roi_pct: float = 0.30  # -30% stop-loss (-3.75% price move at 8x)
        self.trailing_sl_roi_pct: float = 999.0  # Strict 2:1 RR (No trailing shakeouts)
        self.auto_loop: bool = True  # Continuous autonomous cycle until stopped
        self.order_entry_type: str = "LIMIT_MAKER"  # Post-Only Maker entry

        # Anti-Whipsaw consecutive loss cooldown
        self.consecutive_losses: int = 0
        self.whipsaw_cooldown_until: float = 0.0

        # Strategy Guardian Health Monitoring Engine
        self.peak_balance: float = 100.0
        self.strategy_consecutive_losses: int = 0
        self.recent_closed_trades: List[Dict[str, Any]] = []
        self.strategy_health: Dict[str, Any] = {
            "status": "HEALTHY",  # HEALTHY | WARNING | DANGER
            "rolling_win_rate_pct": 50.0,
            "rolling_trades_count": 0,
            "consecutive_losses": 0,
            "current_drawdown_pct": 0.0,
            "peak_balance": 100.0,
            "reasons": [],
            "expired": False,
        }

        self.last_scan_ts: float = 0.0
        self.next_scan_ts: float = 0.0
        self.last_trade_ts: float = 0.0
        self.last_research_ts: float = 0.0
        self.total_auto_entries: int = 0
        self.total_auto_exits: int = 0

        self._task: Optional[asyncio.Task] = None
        self._running: bool = False
        self.chat_history: List[Dict[str, Any]] = []

        # Callbacks wired by agent_server.py
        self._run_analysis_cb: Optional[Callable] = None
        self._place_order_cb: Optional[Callable] = None
        self._close_position_cb: Optional[Callable] = None
        self._get_portfolio_cb: Optional[Callable] = None

        self.add_chat(
            "agent",
            "Autonomous Trading Agent online. Status: IDLE. Ready for 'SOL 8x Hourly Trend-Surfer' "
            "(Progressive Tiered Margin 30% -> 5% floor, 8x Leverage, +60% TP, -30% SL, 1 trade at a time).",
            action_tag="SYSTEM_READY",
        )

    def wire_callbacks(self, run_analysis_cb, place_order_cb, close_position_cb, get_portfolio_cb):
        self._run_analysis_cb = run_analysis_cb
        self._place_order_cb = place_order_cb
        self._close_position_cb = close_position_cb
        self._get_portfolio_cb = get_portfolio_cb

    def add_chat(self, role: str, text: str, action_tag: str = "") -> Dict[str, Any]:
        entry = {
            "id": f"msg_{int(time.time() * 1000)}_{len(self.chat_history)}",
            "role": role,  # user | agent | alert
            "text": text,
            "action_tag": action_tag,
            "timestamp": int(time.time()),
        }
        self.chat_history.append(entry)
        if len(self.chat_history) > 80:
            self.chat_history = self.chat_history[-80:]
        return entry

    def evaluate_strategy_health(self, current_balance: Optional[float] = None) -> Dict[str, Any]:
        """
        Strategy Guardian Health Monitor:
        Metrics & Thresholds:
          1. Rolling Win Rate (Last 50 trades):
             - Healthy: >= 34%
             - Warning: 30% - 33.9%
             - Danger:  < 28% (evaluated when trades >= 15)
          2. Consecutive Losses:
             - Healthy: <= 10 trades
             - Warning: 11 - 12 trades
             - Danger:  >= 14 trades
          3. Max Drawdown from Peak:
             - Healthy: < 45%
             - Warning: 45% - 55%
             - Danger:  > 60%
        
        Action if DANGER:
          - Automatically HALTS trading immediately (mode = IDLE, _running = False)
          - Sets expired = True
          - Issues Critical Red Alert
        """
        if current_balance is None and self._get_portfolio_cb:
            port = self._get_portfolio_cb()
            current_balance = float(port.get("balance_usdt", 100.0))
        elif current_balance is None:
            current_balance = 100.0

        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

        # Calculate Drawdown from Peak
        if self.peak_balance > 0:
            drawdown_pct = max(0.0, round(((self.peak_balance - current_balance) / self.peak_balance) * 100.0, 2))
        else:
            drawdown_pct = 0.0

        # Calculate Rolling Win Rate over last 50 trades
        recent_50 = self.recent_closed_trades[-50:]
        trades_count = len(recent_50)
        wins = sum(1 for t in recent_50 if t.get("pnl", 0) > 0)
        rolling_wr = round((wins / trades_count) * 100.0, 1) if trades_count > 0 else 50.0

        consec_losses = self.strategy_consecutive_losses

        # Evaluate Danger Condition
        danger_reasons = []
        if drawdown_pct > 60.0:
            danger_reasons.append(f"Peak Drawdown exceeded 60% (Current DD: {drawdown_pct:.1f}%)")
        if consec_losses >= 14:
            danger_reasons.append(f"Consecutive loss streak reached Danger threshold ({consec_losses} losses >= 14)")
        if trades_count >= 15 and rolling_wr < 28.0:
            danger_reasons.append(f"Rolling Win Rate collapsed below 28% (Current WR: {rolling_wr:.1f}% across {trades_count} trades)")

        # Evaluate Warning Condition
        warning_reasons = []
        if not danger_reasons:
            if 45.0 <= drawdown_pct <= 60.0:
                warning_reasons.append(f"Elevated Drawdown ({drawdown_pct:.1f}% between 45% - 60%)")
            if 11 <= consec_losses < 14:
                warning_reasons.append(f"Elevated loss streak ({consec_losses} consecutive losses)")
            if trades_count >= 15 and 28.0 <= rolling_wr <= 33.9:
                warning_reasons.append(f"Rolling Win Rate compressed to warning zone ({rolling_wr:.1f}%)")

        prev_status = self.strategy_health.get("status", "HEALTHY")

        if danger_reasons:
            status = "DANGER"
            is_expired = True
            reasons = danger_reasons
            
            # AUTOMATIC CIRCUIT BREAKER: HALT TRADING IMMEDIATELY!
            if self._running or self.mode != "IDLE":
                self._running = False
                self.mode = "IDLE"
                self.add_chat(
                    "alert",
                    f"🛑 [CIRCUIT BREAKER TRIGGERED] STRATEGY EXPIRED & TRADING AUTOMATICALLY HALTED! "
                    f"Danger Zone breached: {'; '.join(danger_reasons)}. "
                    f"RED SIGNAL: Current market regime has invalidated the strategy edge. Change or re-calibrate strategy before resuming.",
                    action_tag="STRATEGY_EXPIRED",
                )
        elif warning_reasons:
            status = "WARNING"
            is_expired = False
            reasons = warning_reasons
            if prev_status != "WARNING":
                self.add_chat(
                    "alert",
                    f"⚠️ [STRATEGY HEALTH WARNING] Strategy entered Warning Zone: {'; '.join(warning_reasons)}. "
                    f"Turbulence elevated. Monitor closely.",
                    action_tag="STRATEGY_WARNING",
                )
        else:
            status = "HEALTHY"
            is_expired = False
            reasons = []

        self.strategy_health = {
            "status": status,
            "rolling_win_rate_pct": rolling_wr,
            "rolling_trades_count": trades_count,
            "consecutive_losses": consec_losses,
            "current_drawdown_pct": drawdown_pct,
            "peak_balance": round(self.peak_balance, 2),
            "reasons": reasons,
            "expired": is_expired,
        }
        return self.strategy_health

    def reset_strategy_guardian(self):
        """Resets the Strategy Guardian health metrics after user recalibrates or changes strategy."""
        current_balance = 100.0
        if self._get_portfolio_cb:
            port = self._get_portfolio_cb()
            current_balance = float(port.get("balance_usdt", 100.0))
        self.peak_balance = current_balance
        self.strategy_consecutive_losses = 0
        self.recent_closed_trades = []
        self.strategy_health = {
            "status": "HEALTHY",
            "rolling_win_rate_pct": 50.0,
            "rolling_trades_count": 0,
            "consecutive_losses": 0,
            "current_drawdown_pct": 0.0,
            "peak_balance": round(self.peak_balance, 2),
            "reasons": [],
            "expired": False,
        }
        self.add_chat(
            "agent",
            "Strategy Guardian metrics reset. New strategy baseline calibrated.",
            action_tag="GUARDIAN_RESET",
        )

    def get_state(self) -> Dict[str, Any]:
        now = time.time()
        gap_remaining = max(0, int(self.trade_gap_sec - (now - self.last_trade_ts))) if self.last_trade_ts > 0 else 0
        next_scan_in = max(0, int(self.next_scan_ts - now)) if self._running and self.next_scan_ts > 0 else 0
        self.evaluate_strategy_health()
        return {
            "mode": self.mode,
            "running": self._running,
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "scan_interval_sec": self.scan_interval_sec,
            "trade_gap_sec": self.trade_gap_sec,
            "gap_cooldown_remaining_sec": gap_remaining,
            "next_scan_in_sec": next_scan_in,
            "trade_amount_usdt": self.trade_amount_usdt,
            "max_open_positions": self.max_open_positions,
            "leverage": self.leverage,
            "capital_allocation_pct": self.capital_allocation_pct,
            "target_roi_pct": self.target_roi_pct,
            "initial_sl_roi_pct": self.initial_sl_roi_pct,
            "trailing_sl_roi_pct": self.trailing_sl_roi_pct,
            "auto_loop": self.auto_loop,
            "total_auto_entries": self.total_auto_entries,
            "total_auto_exits": self.total_auto_exits,
            "strategy_health": self.strategy_health,
            "chat_history": self.chat_history[-30:],
        }

    # ---------------------------------------------------------
    # TASK LIFECYCLE CONTROL
    # ---------------------------------------------------------
    async def start_task(
        self,
        mode: str,
        symbols: Optional[List[str]] = None,
        timeframe: Optional[str] = None,
        trade_gap_sec: Optional[int] = None,
        scan_interval_sec: Optional[int] = None,
        trade_amount_usdt: Optional[float] = None,
        leverage: Optional[float] = None,
        capital_allocation_pct: Optional[float] = None,
        target_roi_pct: Optional[float] = None,
        initial_sl_roi_pct: Optional[float] = None,
        trailing_sl_roi_pct: Optional[float] = None,
        max_open_positions: Optional[int] = None,
        auto_loop: Optional[bool] = None,
    ) -> Dict[str, Any]:
        mode = mode.upper()
        if mode not in ("IDLE", "RESEARCH_ONLY", "TRACK_ONLY", "AUTO_TRADE", "CONTINUOUS_100"):
            mode = "RESEARCH_ONLY"

        if mode == "CONTINUOUS_100":
            self.symbols = ["SOL/USDT"]
            self.timeframe = "1h"
            self.leverage = 8.0
            self.target_roi_pct = 0.60  # +60% Target ROI (+7.50% price move at 8x)
            self.initial_sl_roi_pct = 0.30  # -30% Stop Loss (-3.75% price move at 8x)
            self.trailing_sl_roi_pct = 999.0  # Strict 2:1 RR (No trailing shakeouts)
            self.auto_loop = True
            self.max_open_positions = 1

        if symbols:
            self.symbols = symbols
        if timeframe:
            self.timeframe = timeframe
        if trade_gap_sec is not None:
            self.trade_gap_sec = max(0, int(trade_gap_sec))
        if scan_interval_sec is not None:
            self.scan_interval_sec = max(3, int(scan_interval_sec))
        if trade_amount_usdt is not None:
            self.trade_amount_usdt = max(5.0, float(trade_amount_usdt))
        if leverage is not None:
            self.leverage = float(leverage)
        if capital_allocation_pct is not None:
            self.capital_allocation_pct = min(1.0, max(0.05, float(capital_allocation_pct)))
        if target_roi_pct is not None:
            self.target_roi_pct = float(target_roi_pct)
        if initial_sl_roi_pct is not None:
            self.initial_sl_roi_pct = float(initial_sl_roi_pct)
        if trailing_sl_roi_pct is not None:
            self.trailing_sl_roi_pct = float(trailing_sl_roi_pct)
        if max_open_positions is not None:
            self.max_open_positions = max(1, int(max_open_positions))
        if auto_loop is not None:
            self.auto_loop = bool(auto_loop)

        self.mode = mode

        if mode == "IDLE":
            await self.stop_task()
            return self.get_state()

        if not self._running or (self._task and self._task.done()):
            self._running = True
            self._task = asyncio.create_task(self._worker_loop())

        return self.get_state()

    async def stop_task(self) -> Dict[str, Any]:
        self._running = False
        self.mode = "IDLE"
        self.next_scan_ts = 0.0
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        return self.get_state()

    # ---------------------------------------------------------
    # AUTONOMOUS SL / TARGET / TRAILING-SL TRACKER & WORKER LOOP
    # ---------------------------------------------------------
    def check_sl_tp_triggers(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Monitors open positions against live prices.
        Applies:
          1. Step-Based Milestone Trailing Stop-Loss:
             - Initial SL at -70% ROI
             - When profit reaches +20% ROI -> SL moves to -50%
             - When profit reaches +40% ROI -> SL moves to -30%
             - When profit reaches +60% ROI -> SL moves to -10%
             - When profit reaches +70% ROI -> Target hit!
          2. Instant Touch / Cross Market Exit Rule:
             - Once price reaches or touches Target or Stop-Loss (either exact, above, or below),
               immediately trigger a MARKET exit at current market price without waiting.
        """
        if not self._get_portfolio_cb or not self._close_position_cb:
            return []

        prices = {s["symbol"]: float(s["current_price"]) for s in signals if "symbol" in s and "current_price" in s}
        high_prices = {s["symbol"]: float(s.get("high_price", s.get("current_price", 0))) for s in signals if "symbol" in s}
        low_prices = {s["symbol"]: float(s.get("low_price", s.get("current_price", 0))) for s in signals if "symbol" in s}
        portfolio = self._get_portfolio_cb()
        closed_events = []

        # Iterate over a copy since close modifies positions list
        for pos in list(portfolio.get("positions", [])):
            sym = pos.get("symbol")
            if sym not in prices:
                continue
            live_price = prices[sym]
            high_price = high_prices.get(sym, live_price)
            low_price = low_prices.get(sym, live_price)
            side = pos.get("side", "LONG").upper()
            entry_price = float(pos.get("entry_price", live_price))
            leverage = float(pos.get("leverage", self.leverage))
            target_roi = float(pos.get("target_roi_pct", self.target_roi_pct))
            initial_sl_roi = float(pos.get("initial_sl_roi_pct", self.initial_sl_roi_pct))
            step_size_roi = float(pos.get("trailing_sl_roi_pct", self.trailing_sl_roi_pct))

            hit_reason = None
            if side == "LONG":
                profit_roi = ((live_price - entry_price) / entry_price) * leverage
                pos["peak_profit_roi"] = max(float(pos.get("peak_profit_roi", 0.0)), profit_roi)
                pos["peak_price"] = max(float(pos.get("peak_price", entry_price)), live_price, high_price)
                pos["trough_price"] = min(float(pos.get("trough_price", entry_price)), live_price, low_price)

                # Milestone steps: ratchets only if step_size_roi is enabled (< 50.0)
                if 0 < step_size_roi < 50.0:
                    steps = int((round(pos["peak_profit_roi"], 4) + 1e-6) // step_size_roi) if round(pos["peak_profit_roi"], 4) >= step_size_roi else 0
                    new_sl_roi = -initial_sl_roi + (steps * step_size_roi)
                    current_sl_roi = round(max(float(pos.get("current_sl_roi", -initial_sl_roi)), new_sl_roi), 4)
                else:
                    steps = 0
                    current_sl_roi = -initial_sl_roi
                pos["current_sl_roi"] = current_sl_roi

                current_sl = entry_price * (1.0 + (current_sl_roi / leverage))
                target_p = entry_price * (1.0 + (target_roi / leverage))
                pos["stop_loss"] = round(current_sl, 6)
                pos["target_price"] = round(target_p, 6)

                # Announce milestone step ratcheting
                last_step = pos.get("last_reported_step", 0)
                if steps > last_step:
                    pos["last_reported_step"] = steps
                    self.add_chat(
                        "alert",
                        f"[TRAILING SL STEP] {sym} profit reached +{steps*step_size_roi*100:.0f}% ROI! "
                        f"Ratcheting Stop-Loss from -{initial_sl_roi*100:.0f}% to {current_sl_roi*100:+.0f}% ROI (SL: ${current_sl:.4f}).",
                        action_tag="TRAILING_STEP",
                    )

                # INSTANT TOUCH / CROSS TRIGGER (DO NOT WAIT FOR EXACT PRICE):
                # If price touched or surpassed target at any point, exit immediately at market price!
                if live_price >= target_p or pos["peak_price"] >= target_p:
                    hit_reason = "TARGET_HIT"
                # If price touched or fell below stop loss at any point, exit immediately at market price!
                elif live_price <= current_sl or pos["trough_price"] <= current_sl:
                    hit_reason = "TRAILING_SL_HIT" if current_sl_roi > -initial_sl_roi else "STOP_LOSS_HIT"

            else:  # SHORT
                profit_roi = ((entry_price - live_price) / entry_price) * leverage
                pos["peak_profit_roi"] = max(float(pos.get("peak_profit_roi", 0.0)), profit_roi)
                pos["peak_price"] = max(float(pos.get("peak_price", entry_price)), live_price, high_price)
                pos["trough_price"] = min(float(pos.get("trough_price", entry_price)), live_price, low_price)

                # Milestone steps for Short
                if 0 < step_size_roi < 50.0:
                    steps = int((round(pos["peak_profit_roi"], 4) + 1e-6) // step_size_roi) if round(pos["peak_profit_roi"], 4) >= step_size_roi else 0
                    new_sl_roi = -initial_sl_roi + (steps * step_size_roi)
                    current_sl_roi = round(max(float(pos.get("current_sl_roi", -initial_sl_roi)), new_sl_roi), 4)
                else:
                    steps = 0
                    current_sl_roi = -initial_sl_roi
                pos["current_sl_roi"] = current_sl_roi

                current_sl = entry_price * (1.0 - (current_sl_roi / leverage))
                target_p = entry_price * (1.0 - (target_roi / leverage))
                pos["stop_loss"] = round(current_sl, 6)
                pos["target_price"] = round(target_p, 6)

                # Announce milestone step ratcheting
                last_step = pos.get("last_reported_step", 0)
                if steps > last_step:
                    pos["last_reported_step"] = steps
                    self.add_chat(
                        "alert",
                        f"[TRAILING SL STEP] {sym} profit reached +{steps*step_size_roi*100:.0f}% ROI! "
                        f"Ratcheting Stop-Loss from -{initial_sl_roi*100:.0f}% to {current_sl_roi*100:+.0f}% ROI (SL: ${current_sl:.4f}).",
                        action_tag="TRAILING_STEP",
                    )

                # INSTANT TOUCH / CROSS TRIGGER (DO NOT WAIT FOR EXACT PRICE):
                # If short price touched or dropped below target at any point, exit immediately at market price!
                if live_price <= target_p or pos["trough_price"] <= target_p:
                    hit_reason = "TARGET_HIT"
                # If short price touched or spiked above stop loss at any point, exit immediately at market price!
                elif live_price >= current_sl or pos["peak_price"] >= current_sl:
                    hit_reason = "TRAILING_SL_HIT" if current_sl_roi > -initial_sl_roi else "STOP_LOSS_HIT"

            if hit_reason:
                res = self._close_position_cb(pos["id"], live_price)
                if res.get("status") == "success":
                    self.total_auto_exits += 1
                    pnl = res.get("pnl", 0.0)
                    pnl_pct = res.get("pnl_pct", 0.0)
                    tag = "AUTO_TP" if hit_reason == "TARGET_HIT" else ("AUTO_TRAIL" if hit_reason == "TRAILING_SL_HIT" else "AUTO_SL")
                    msg = (
                        f"[{hit_reason} • INSTANT TOUCH EXIT] Price touched/crossed level. "
                        f"Executed immediate MARKET EXIT on {side} {sym} at ${live_price:.4f} "
                        f"(Target: ${target_p:.4f}, SL: ${current_sl:.4f}, Entry: ${entry_price:.4f}) | "
                        f"PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%) | New Balance: ${res.get('new_balance', 0):.2f}"
                    )
                    self.add_chat("alert", msg, action_tag=tag)
                    closed_events.append({"pos_id": pos["id"], "reason": hit_reason, "pnl": pnl})

                    # --- SELF-LEARNING MEMORY ENGINE HOOK ---
                    pos_record = dict(pos)
                    pos_record["exit_price"] = live_price
                    pos_record["exit_reason"] = hit_reason
                    pos_record["pnl"] = pnl
                    pos_record["pnl_pct"] = pnl_pct

                    lesson = trading_memory.record_closed_trade(pos_record)
                    if lesson:
                        self.consecutive_losses += 1
                        self.add_chat(
                            "alert",
                            f"[SELF-LEARNING MISTAKE DIAGNOSED] {sym} {side} failed: {lesson['failure_cause']} | "
                            f"Rule Formed: {lesson['preventive_rule']} (Added to active veto rules to prevent repeating).",
                            action_tag="LEARNED_MISTAKE",
                        )
                        if self.consecutive_losses >= 2:
                            self.whipsaw_cooldown_until = time.time() + 3600.0  # 1 hour pause
                            self.consecutive_losses = 0
                            self.add_chat(
                                "alert",
                                "[ANTI-WHIPSAW PROTECTION] 2 consecutive losses detected. Market exhibiting sideways chop. Cooling down auto-entries for 1 hour to protect capital.",
                                action_tag="WHIPSAW_PAUSE",
                            )
                    elif pnl > 0:
                        self.consecutive_losses = 0
                        self.add_chat(
                            "agent",
                            f"[SELF-LEARNING MEMORY] Trade won! Win rate: {trading_memory.data.get('win_rate_pct', 0):.1f}% "
                            f"({trading_memory.data.get('winning_trades', 0)}W / {trading_memory.data.get('losing_trades', 0)}L). Winning pattern reinforced.",
                            action_tag="MEMORY_WIN",
                        )

                    # --- STRATEGY GUARDIAN HEALTH ENGINE ---
                    new_bal = float(res.get("new_balance", 0.0))
                    if pnl > 0:
                        self.strategy_consecutive_losses = 0
                    else:
                        self.strategy_consecutive_losses += 1

                    self.recent_closed_trades.append({
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "timestamp": time.time(),
                        "is_win": pnl > 0
                    })
                    if len(self.recent_closed_trades) > 100:
                        self.recent_closed_trades = self.recent_closed_trades[-100:]

                    health = self.evaluate_strategy_health(new_bal)

                    # IMMEDIATELY TRIGGER RE-RESEARCH & RE-ENTRY FOR CONTINUOUS CYCLE (IF NOT EXPIRED)
                    if self._running and self.mode in ("AUTO_TRADE", "CONTINUOUS_100") and self.auto_loop:
                        if health.get("status") == "DANGER" or health.get("expired"):
                            self._running = False
                            self.mode = "IDLE"
                            self.add_chat(
                                "alert",
                                f"🛑 [CIRCUIT BREAKER] Trade exit pushed strategy into DANGER ZONE: {'; '.join(health.get('reasons', []))}. "
                                f"Trading automatically HALTED. Please change or re-calibrate strategy.",
                                action_tag="STRATEGY_EXPIRED",
                            )
                        else:
                            asyncio.create_task(self._trigger_immediate_research_and_reentry())

        return closed_events

    async def _trigger_immediate_research_and_reentry(self):
        """
        Autonomous single-trade lifecycle loop:
        When a trade exits, immediately re-scans DOGE, evaluates direction,
        sizes 30% of capital, and enters the next trade at 50x leverage.
        """
        try:
            if not self._running or self.mode not in ("AUTO_TRADE", "CONTINUOUS_100"):
                return

            portfolio = self._get_portfolio_cb() if self._get_portfolio_cb else {}
            if len(portfolio.get("positions", [])) >= self.max_open_positions:
                return

            self.add_chat(
                "agent",
                "[CONTINUOUS CYCLE] Trade settled. Running instant directional analysis & placing next SOL trade...",
                action_tag="RESEARCH_LOOP",
            )

            # Brief pause to let trade settlement finalize
            await asyncio.sleep(1.0)

            if not self._running or self.mode not in ("AUTO_TRADE", "CONTINUOUS_100"):
                return

            if self._run_analysis_cb:
                snapshot = await self._run_analysis_cb(self.symbols, self.timeframe, force_refresh_research=True)
                signals = snapshot.get("signals", [])
                await self._maybe_execute_auto_entry(signals, is_immediate_loop=True, snapshot=snapshot)
        except Exception as e:
            print(f"Immediate re-entry error: {e}")

    async def _worker_loop(self):
        while self._running and self.mode != "IDLE":
            try:
                now = time.time()
                self.last_scan_ts = now
                self.next_scan_ts = now + self.scan_interval_sec

                if self._run_analysis_cb:
                    force_news = (now - self.last_research_ts) >= 120.0
                    snapshot = await self._run_analysis_cb(
                        self.symbols, self.timeframe, force_refresh_research=force_news
                    )
                    signals = snapshot.get("signals", [])

                    # 1. Always track open positions for SL/TP when in TRACK_ONLY, AUTO_TRADE, or CONTINUOUS_100
                    if self.mode in ("TRACK_ONLY", "AUTO_TRADE", "CONTINUOUS_100"):
                        self.check_sl_tp_triggers(signals)

                    # 2. Periodic Research report in RESEARCH_ONLY mode (every 120s)
                    if self.mode == "RESEARCH_ONLY" and force_news:
                        self.last_research_ts = now
                        brief = snapshot.get("ai_briefing", {}).get("briefing", "")
                        fng = snapshot.get("macro", {}).get("fear_greed_value", 50)
                        self.add_chat(
                            "agent",
                            f"[Research Scan • F&G {fng}] {brief}",
                            action_tag="RESEARCH_BRIEF",
                        )

                    # 3. Auto-Entry with Single-Trade Cap & 30% Capital Sizing
                    if self.mode in ("AUTO_TRADE", "CONTINUOUS_100"):
                        await self._maybe_execute_auto_entry(signals, snapshot=snapshot)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Autonomous worker error: {e}")

            await asyncio.sleep(self.scan_interval_sec)

    def validate_sr_feasibility(
        self,
        candidate: Dict[str, Any],
        side: str,
        entry_price: float,
        target_price: float,
        stop_loss: float,
    ) -> tuple[bool, str, float]:
        """
        Validates:
        1. Target Achievability: Ensures no major counter-structure barrier (Resistance for LONG, Support for SHORT)
           blocks the required +70% target before it can be achieved.
        2. SL Protection: Ensures the Stop-Loss is sheltered behind structural S/R (Support for LONG, Resistance for SHORT)
           and buffers against sweep wicks so it is not easily hittable.
        Returns: (is_valid: bool, reason: str, adjusted_sl: float)
        """
        near_sup = float(candidate.get("nearest_support") or (entry_price * 0.98))
        near_res = float(candidate.get("nearest_resistance") or (entry_price * 1.02))
        atr = float(candidate.get("atr") or (entry_price * 0.015))

        adjusted_sl = stop_loss

        if side == "LONG":
            # 1. Target Feasibility Check:
            target_distance = target_price - entry_price
            if near_res < target_price:
                res_room = near_res - entry_price
                if res_room < 0.65 * target_distance:
                    pct_to_res = (near_res - entry_price) / entry_price * 100
                    return False, f"Target blocked: Overhead Resistance at ${near_res:.4f} (+{pct_to_res:.2f}%) caps price before +1.40% target (${target_price:.4f}).", adjusted_sl

            # 2. SL Protection Check:
            if stop_loss < near_sup < entry_price:
                # Support cushions our SL! Buffer if too close to support line
                if (near_sup - stop_loss) < 0.20 * atr:
                    buffered_sl = round(near_sup - (0.25 * atr), 6)
                    if (entry_price - buffered_sl) / entry_price <= 0.016:
                        adjusted_sl = buffered_sl
            elif near_sup < stop_loss:
                dist_to_sup = (entry_price - near_sup) / entry_price
                if dist_to_sup > 0.020:
                    return False, f"SL vulnerable: No support structure cushions the drop (nearest support at ${near_sup:.4f}, -{dist_to_sup*100:.2f}%).", adjusted_sl

            return True, f"Target clear of resistance (room to ${near_res:.4f}); SL sheltered by support at ${near_sup:.4f}.", adjusted_sl

        else: # SHORT
            # 1. Target Feasibility Check:
            target_distance = entry_price - target_price
            if near_sup > target_price:
                sup_room = entry_price - near_sup
                if sup_room < 0.65 * target_distance:
                    pct_to_sup = (entry_price - near_sup) / entry_price * 100
                    return False, f"Target blocked: Support floor at ${near_sup:.4f} (-{pct_to_sup:.2f}%) will bounce price before -1.40% target (${target_price:.4f}).", adjusted_sl

            # 2. SL Protection Check:
            if entry_price < near_res < stop_loss:
                # Resistance cushions our SL! Buffer if too close to resistance line
                if (stop_loss - near_res) < 0.20 * atr:
                    buffered_sl = round(near_res + (0.25 * atr), 6)
                    if (buffered_sl - entry_price) / entry_price <= 0.016:
                        adjusted_sl = buffered_sl
            elif near_res > stop_loss:
                dist_to_res = (near_res - entry_price) / entry_price
                if dist_to_res > 0.020:
                    return False, f"SL vulnerable: No resistance ceiling cushions upward wicks (nearest resistance at ${near_res:.4f}, +{dist_to_res*100:.2f}%).", adjusted_sl

            return True, f"Target clear of support (floor at ${near_sup:.4f}); SL sheltered by resistance at ${near_res:.4f}.", adjusted_sl

    def get_tiered_margin_pct(self, balance: float) -> float:
        """
        Progressive Tiered De-Leveraging Formula:
        - Balance < $200: 30% margin (Fast initial doubling phase)
        - $200 <= Balance < $300: 25% margin
        - $300 <= Balance < $400: 20% margin
        - $400 <= Balance < $500: 15% margin
        - $500 <= Balance < $600: 10% margin
        - Balance >= $600: 5% margin floor (Permanent wealth-preservation)
        """
        if balance < 200.0:
            return 0.30
        elif balance < 300.0:
            return 0.25
        elif balance < 400.0:
            return 0.20
        elif balance < 500.0:
            return 0.15
        elif balance < 600.0:
            return 0.10
        else:
            return 0.05

    async def _maybe_execute_auto_entry(
        self,
        signals: List[Dict[str, Any]],
        is_immediate_loop: bool = False,
        snapshot: Optional[Dict[str, Any]] = None,
    ):
        if not self._get_portfolio_cb or not self._place_order_cb:
            return

        # Check Strategy Guardian Health status before entry
        health = self.evaluate_strategy_health()
        if health.get("status") == "DANGER" or health.get("expired"):
            return

        now = time.time()
        # Enforce anti-whipsaw cooldown if triggered by consecutive losses
        if now < self.whipsaw_cooldown_until:
            return

        # Enforce requested gap only for regular intervals, not immediate exit re-entry
        if not is_immediate_loop and self.trade_gap_sec > 0 and self.last_trade_ts > 0 and (now - self.last_trade_ts) < self.trade_gap_sec:
            return

        portfolio = self._get_portfolio_cb()
        open_positions = portfolio.get("positions", [])

        # STRICT CONSTRAINT: Only ONE trade at a time
        if len(open_positions) >= self.max_open_positions:
            return

        current_balance = float(portfolio.get("balance_usdt", 0.0))
        # 30% Capital allocation
        allocated_capital = round(current_balance * self.capital_allocation_pct, 2)
        if allocated_capital < 1.50:
            return

        # -------------------------------------------------------------
        # CONTINUOUS: "SOL 8x HOURLY TREND-SURFER" (TIERED DE-LEVERAGING)
        # -------------------------------------------------------------
        if self.mode == "CONTINUOUS_100":
            sol_sig = next((s for s in signals if "SOL" in s.get("symbol", "")), signals[0] if signals else None)
            if not sol_sig:
                return

            c_entry = float(sol_sig.get("current_price", 0.0))
            if c_entry <= 0:
                return

            # Sizing via Progressive Tiered De-Leveraging Formula (30% -> 5% floor)
            m_pct = self.get_tiered_margin_pct(current_balance)
            allocated_capital = round(current_balance * m_pct, 2)
            if allocated_capital < 1.50:
                return

            # Trend Alignment Indicators (SOL 8x Hourly Trend-Surfer)
            e9 = float(sol_sig.get("ema_9", c_entry))
            e21 = float(sol_sig.get("ema_21", c_entry))
            e72 = float(sol_sig.get("ema_72", sol_sig.get("ema_50", c_entry)))
            lev = 8.0  # 8x Isolated Leverage

            # Decision Engine: 9 EMA vs 21 EMA with 72-Hour Macro Trend Filter
            if e9 > e21 and c_entry > e72:
                c_side = "LONG"
                bias = f"Bullish Trend-Surge (EMA9 ${e9:.2f} > EMA21 ${e21:.2f} & Price > 72h EMA ${e72:.2f})"
            elif e9 < e21 and c_entry < e72:
                c_side = "SHORT"
                bias = f"Bearish Trend-Break (EMA9 ${e9:.2f} < EMA21 ${e21:.2f} & Price < 72h EMA ${e72:.2f})"
            else:
                c_side = "LONG" if c_entry > e72 else "SHORT"
                bias = f"Macro Trend Alignment (Price vs 72h EMA ${e72:.2f} -> {c_side})"

            # Target: +60% ROI (+7.50% price move at 8x)
            # Stop Loss: -30% ROI (-3.75% price move at 8x)
            # Fixed 2:1 Risk-to-Reward (No trailing shakeouts)
            t_delta = c_entry * (0.60 / lev)
            s_delta = c_entry * (0.30 / lev)

            if c_side == "LONG":
                c_target = round(c_entry + t_delta, 2)
                c_sl = round(c_entry - s_delta, 2)
            else:
                c_target = round(c_entry - t_delta, 2)
                c_sl = round(c_entry + s_delta, 2)

            res = self._place_order_cb(
                symbol="SOL/USDT",
                side=c_side,
                entry_price=c_entry,
                stop_loss=c_sl,
                target_price=c_target,
                amount_usdt=allocated_capital,
                leverage=lev,
                target_roi_pct=0.60,
                initial_sl_roi_pct=0.30,
                trailing_sl_roi_pct=999.0,
            )

            self.last_trade_ts = now
            self.add_chat(
                "agent",
                f"[SOL 8x TREND-SURFER] {bias}. "
                f"Entered 8x {c_side} @ ${c_entry:.2f} | Target: ${c_target:.2f} (+60% ROI / +7.5%) | "
                f"SL: ${c_sl:.2f} (-30% ROI / -3.75%) | Margin: ${allocated_capital:.2f} ({int(m_pct*100)}% tier)",
                action_tag="CONTINUOUS_ORDER",
            )
            return

        open_symbols = {p.get("symbol") for p in open_positions}
        # Strict high-conviction candidate selection (min_confidence >= 80)
        candidates = [
            s for s in signals
            if s.get("signal") in ("BUY (LONG)", "SELL (SHORT)")
            and s.get("confidence_pct", 0) >= self.min_confidence
            and s.get("symbol") not in open_symbols
        ]

        if not candidates:
            return

        # Rank candidates by ADX trend strength with priority for high-momentum pairs (SOL)
        candidates.sort(
            key=lambda x: float(x.get("adx", 20.0)) + (15.0 if "SOL" in x.get("symbol", "") else 0.0),
            reverse=True,
        )

        # -------------------------------------------------------------
        # S/R CONFLUENCE & SELF-LEARNING MEMORY FILTER
        # -------------------------------------------------------------
        valid_candidates = []
        for cand in candidates:
            c_side = "LONG" if "BUY" in cand["signal"] else "SHORT"
            c_entry = float(cand["current_price"])
            t_delta = c_entry * (self.target_roi_pct / self.leverage)
            s_delta = c_entry * (self.initial_sl_roi_pct / self.leverage)

            if c_side == "LONG":
                c_target = round(c_entry + t_delta, 6)
                c_sl = round(c_entry - s_delta, 6)
            else:
                c_target = round(c_entry - t_delta, 6)
                c_sl = round(c_entry + s_delta, 6)

            # Step 1: S/R Target Achievability & SL Protection Check
            is_valid_sr, sr_msg, adj_sl = self.validate_sr_feasibility(
                cand, c_side, c_entry, c_target, c_sl
            )
            if not is_valid_sr:
                self.add_chat(
                    "agent",
                    f"[S/R FILTER] Skipping {cand['symbol']} {c_side}: {sr_msg}",
                    action_tag="SR_FILTER",
                )
                continue

            # Step 2: Self-Learning Memory Veto Check (Do NOT repeat past mistakes)
            is_valid_mem, mem_msg, conf_adj = trading_memory.evaluate_candidate(
                cand, c_side, c_entry
            )
            if not is_valid_mem:
                self.add_chat(
                    "agent",
                    f"[MEMORY VETO] Skipping {cand['symbol']} {c_side}: {mem_msg}",
                    action_tag="MEMORY_VETO",
                )
                continue

            effective_conf = float(cand.get("confidence_pct", 50)) + conf_adj
            valid_candidates.append({
                "candidate": cand,
                "side": c_side,
                "entry_price": c_entry,
                "target_price": c_target,
                "stop_loss": adj_sl,
                "sr_reason": sr_msg,
                "mem_reason": mem_msg,
                "effective_conf": effective_conf,
            })

        if not valid_candidates:
            return

        # Antigravity AI setup ranking: prioritize SOL if active setup, else rank by effective confidence
        sol_candidate = next((v for v in valid_candidates if "SOL" in v["candidate"].get("symbol", "")), None)
        if sol_candidate:
            best_pick = sol_candidate
        else:
            best_pick = max(
                valid_candidates,
                key=lambda x: (x["effective_conf"], abs(float(x["candidate"].get("rsi", 50)) - 50))
            )

        best = best_pick["candidate"]
        side = best_pick["side"]
        entry_price = best_pick["entry_price"]
        target_price = best_pick["target_price"]
        stop_loss = best_pick["stop_loss"]
        sr_reason = best_pick["sr_reason"]
        mem_reason = best_pick["mem_reason"]

        res = self._place_order_cb(
            symbol=best["symbol"],
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            amount_usdt=allocated_capital,
            leverage=self.leverage,
            target_roi_pct=self.target_roi_pct,
            initial_sl_roi_pct=self.initial_sl_roi_pct,
            trailing_sl_roi_pct=self.trailing_sl_roi_pct,
            rsi=float(best.get("rsi") or 50.0),
            nearest_support=float(best.get("nearest_support") or 0.0),
            nearest_resistance=float(best.get("nearest_resistance") or 0.0),
            atr=float(best.get("atr") or 0.0),
        )
        if res.get("status") == "success":
            self.last_trade_ts = now
            self.total_auto_entries += 1
            trail_pct_delta = (self.trailing_sl_roi_pct / self.leverage) * 100
            self.add_chat(
                "alert",
                f"[50x AUTO-ENTRY] Opened {side} {best['symbol']} with ${allocated_capital:.2f} USDT (30% capital) "
                f"@ ${entry_price:.4f} | Target: ${target_price:.4f} (+{self.target_roi_pct*100:.0f}%) | "
                f"SL: ${stop_loss:.4f} (-{self.initial_sl_roi_pct*100:.0f}%) | "
                f"Trailing SL: 20% distance ({trail_pct_delta:.2f}% price). S/R Check: {sr_reason} | Memory: {mem_reason}",
                action_tag="AUTO_ENTRY",
            )

    # ---------------------------------------------------------
    # CHAT COMMAND PARSER & PROFESSIONAL TRADER RESPONSE
    # ---------------------------------------------------------
    async def process_chat_message(self, user_message: str) -> Dict[str, Any]:
        msg_clean = user_message.strip()
        if not msg_clean:
            return {"reply": "Please enter a command or question.", "state": self.get_state()}

        self.add_chat("user", msg_clean)
        lower = msg_clean.lower()

        # Extract optional parameters from natural language
        parsed_symbols = self._extract_symbols(lower)
        parsed_gap = self._extract_gap_seconds(lower)
        parsed_amount = self._extract_amount(lower)

        # 1. STOP / PAUSE COMMANDS
        if any(k in lower for k in ["stop", "pause", "halt", "cancel task", "sleep", "idle"]):
            await self.stop_task()
            reply = "Understood. Autonomous loop stopped. Agent is now IDLE and waiting for your next command."
            self.add_chat("agent", reply, action_tag="TASK_STOPPED")
            return {"reply": reply, "state": self.get_state()}

        # 2. CLOSE ALL POSITIONS COMMAND
        if "close all" in lower or "exit all" in lower:
            closed_summary = await self._close_all_open_positions()
            self.add_chat("agent", closed_summary, action_tag="CLOSE_ALL")
            return {"reply": closed_summary, "state": self.get_state()}

        # 3. RESEARCH ONLY COMMAND
        if any(k in lower for k in ["only research", "research only", "do research", "just research", "analyze news", "scan only"]):
            syms = parsed_symbols or self.symbols
            await self.start_task(
                mode="RESEARCH_ONLY",
                symbols=syms,
                scan_interval_sec=parsed_gap or 30,
            )
            snapshot = await self._run_analysis_cb(syms, self.timeframe, force_refresh_research=True)
            brief = snapshot.get("ai_briefing", {}).get("briefing", "")
            fng = snapshot.get("macro", {}).get("fear_greed_value", 50)
            fng_lbl = snapshot.get("macro", {}).get("fear_greed_label", "Neutral")
            news_sent = snapshot.get("news", {}).get("overall_news_sentiment", "NEUTRAL")
            reply = (
                f"Switched to RESEARCH_ONLY mode on {', '.join(syms)} (trading disabled).\n"
                f"Macro: Fear & Greed {fng} ({fng_lbl}) | News Flow: {news_sent}.\n"
                f"Desk Brief: {brief}"
            )
            self.add_chat("agent", reply, action_tag="RESEARCH_ONLY")
            return {"reply": reply, "state": self.get_state(), "snapshot": snapshot}

        # 4. TRACK SL & TARGET ONLY COMMAND
        if any(k in lower for k in ["track only", "track sl", "monitor sl", "watch sl", "only track", "keep tracking"]):
            await self.start_task(mode="TRACK_ONLY", scan_interval_sec=5)
            reply = (
                "Switched to TRACK_ONLY mode. Monitoring open positions every 5s "
                "with dynamic 20% trailing stop, 70% SL, and 70% Target TP (no new entries opened)."
            )
            self.add_chat("agent", reply, action_tag="TRACK_ONLY")
            return {"reply": reply, "state": self.get_state()}

        # 4B. SOL 8x HOURLY TREND-SURFER CONTINUOUS COMMAND
        if any(k in lower for k in ["continuous", "sol 8x", "trend surfer", "trend-surfer", "start continuous", "run continuous", "sol"]):
            await self.start_task(
                mode="CONTINUOUS_100",
                symbols=["SOL/USDT"],
                timeframe="1h",
                leverage=8.0,
                capital_allocation_pct=0.30,
                target_roi_pct=0.60,
                initial_sl_roi_pct=0.30,
                trailing_sl_roi_pct=999.0,
                max_open_positions=1,
                auto_loop=True,
                scan_interval_sec=5,
            )
            snapshot = await self._run_analysis_cb(["SOL/USDT"], "1h", force_refresh_research=False)
            self.check_sl_tp_triggers(snapshot.get("signals", []))
            await self._maybe_execute_auto_entry(snapshot.get("signals", []), snapshot=snapshot)
            reply = (
                "⚡ 'SOL 8x Hourly Trend-Surfer' activated!\n"
                "• Asset: SOL/USDT (1-Hour Timeframe)\n"
                "• Leverage: 8x Isolated Margin\n"
                "• Sizing: Progressive Tiered De-Leveraging (30% -> 25% -> 20% -> 15% -> 10% -> 5% floor)\n"
                "• Target: +60% ROI (+7.50% price move)\n"
                "• Stop-Loss: -30% ROI (-3.75% price move)\n"
                "• Single Trade Rule: Strictly 1 open position at a time (Continuous re-entry)\n"
                "• Entry Engine: EMA 9 vs EMA 21 crossover aligned with 72-Hour Macro EMA.\n"
                "Say 'stop' anytime to halt."
            )
            self.add_chat("agent", reply, action_tag="CONTINUOUS_START")
            return {"reply": reply, "state": self.get_state(), "snapshot": snapshot}

        # 4C. STRATEGY GUARDIAN HEALTH STATUS & RESET COMMANDS
        if any(k in lower for k in ["guardian", "health", "vital", "vitals", "status of strategy", "strategy status"]):
            health = self.evaluate_strategy_health()
            reasons_str = f"\n• Active Flags: {'; '.join(health['reasons'])}" if health['reasons'] else ""
            reply = (
                f"🛡️ Strategy Guardian Status: {health['status']}\n"
                f"• Rolling Win Rate (Last 50): {health['rolling_win_rate_pct']:.1f}% ({health['rolling_trades_count']} trades analyzed)\n"
                f"• Consecutive Loss Streak: {health['consecutive_losses']} trades\n"
                f"• Peak Drawdown: {health['current_drawdown_pct']:.1f}% (from peak ${health['peak_balance']:.2f})\n"
                f"• Strategy Expired / Halted: {'YES (RED SIGNAL)' if health['expired'] else 'NO'}"
                f"{reasons_str}"
            )
            self.add_chat("agent", reply, action_tag="GUARDIAN_STATUS")
            return {"reply": reply, "state": self.get_state()}

        if any(k in lower for k in ["reset guardian", "reset health", "recalibrate strategy", "unblock strategy"]):
            self.reset_strategy_guardian()
            reply = "🛡️ Strategy Guardian reset! Health status restored to HEALTHY with a clean calibration baseline."
            return {"reply": reply, "state": self.get_state()}

        # 5. 50x LEVERAGE AUTONOMOUS CONTINUOUS LOOP COMMAND
        if any(k in lower for k in ["50x", "half", "70%", "auto trade", "autotrade", "start trading", "run bot", "take trade", "loop"]):
            syms = parsed_symbols or self.symbols
            await self.start_task(
                mode="AUTO_TRADE",
                symbols=syms,
                leverage=50.0,
                capital_allocation_pct=0.30,
                target_roi_pct=0.70,
                initial_sl_roi_pct=0.70,
                trailing_sl_roi_pct=0.20,
                max_open_positions=1,
                auto_loop=True,
                scan_interval_sec=5,
            )
            # Run immediate first scan + check entry
            snapshot = await self._run_analysis_cb(syms, self.timeframe, force_refresh_research=False)
            self.check_sl_tp_triggers(snapshot.get("signals", []))
            await self._maybe_execute_auto_entry(snapshot.get("signals", []), snapshot=snapshot)
            reply = (
                f"Autonomous 50x Trading Loop activated!\n"
                f"• Leverage: 50x\n"
                f"• Capital Allocation: 30% of available capital per trade\n"
                f"• Single Position Rule: Strictly 1 trade at a time (Max positions = 1)\n"
                f"• Profit Target: +70% ROI on margin (1.40% price move)\n"
                f"• Initial Stop-Loss: -70% ROI on margin (1.40% price move)\n"
                f"• Trailing Stop-Loss: Step-based ratcheting (+20% profit moves SL from -70% to -50%, etc.)\n"
                f"• Re-Research Loop: Immediately upon trade exit, rescans markets, consults Antigravity AI, and opens next trade.\n"
                f"Tracked Pairs: {', '.join(syms)}. Say 'stop' anytime to halt."
            )
            self.add_chat("agent", reply, action_tag="AUTO_TRADE")
            return {"reply": reply, "state": self.get_state(), "snapshot": snapshot}

        # 6. SELF-LEARNING MEMORY & MISTAKES STATUS COMMAND
        if any(k in lower for k in ["memory", "mistake", "mistakes", "lesson", "lessons", "learned", "veto", "rules"]):
            mem_sum = trading_memory.get_summary()
            rules_lines = []
            for r in mem_sum.get("active_veto_rules", []):
                rules_lines.append(f"  • [{r.get('symbol')} {r.get('side')}] {r.get('description')} ({r.get('strikes', 1)} strike)")
            rules_text = "\n".join(rules_lines) if rules_lines else "  • No active veto rules yet. Agent is monitoring all setups."
            reply = (
                f"🧠 Self-Learning Memory Engine Status:\n"
                f"• Total Trades Analyzed: {mem_sum['total_trades']} ({mem_sum['winning_trades']}W / {mem_sum['losing_trades']}L)\n"
                f"• Win Rate: {mem_sum['win_rate_pct']}%\n"
                f"• Cumulative Realized PnL: ${mem_sum['total_pnl_usdt']:+.2f} USDT\n"
                f"• Active Mistake Veto Rules ({mem_sum['active_veto_rules_count']}):\n{rules_text}\n\n"
                f"The agent continuously verifies every candidate against these learned rules and will NOT repeat recorded mistake patterns."
            )
            self.add_chat("agent", reply, action_tag="MEMORY_STATUS")
            return {"reply": reply, "state": self.get_state(), "memory": mem_sum}

        # 7. GENERAL CONVERSATIONAL QUERY (Grounded in Live Market, News & Portfolio)
        snapshot = await self._run_analysis_cb(self.symbols, self.timeframe, force_refresh_research=False)
        reply = await self._generate_conversational_reply(msg_clean, snapshot)
        self.add_chat("agent", reply, action_tag="CHAT_REPLY")
        return {"reply": reply, "state": self.get_state(), "snapshot": snapshot}

    async def _close_all_open_positions(self) -> str:
        if not self._get_portfolio_cb or not self._close_position_cb or not self._run_analysis_cb:
            return "Portfolio engine not ready."
        portfolio = self._get_portfolio_cb()
        positions = list(portfolio.get("positions", []))
        if not positions:
            return "No open positions to close."

        snapshot = await self._run_analysis_cb(self.symbols, self.timeframe, force_refresh_research=False)
        prices = {s["symbol"]: float(s["current_price"]) for s in snapshot.get("signals", [])}
        total_pnl = 0.0
        count = 0
        for pos in positions:
            exit_p = prices.get(pos["symbol"], pos["entry_price"])
            res = self._close_position_cb(pos["id"], exit_p)
            if res.get("status") == "success":
                total_pnl += res.get("pnl", 0.0)
                count += 1
        return f"Closed {count} open position(s) at market. Net realized PnL: ${total_pnl:+.2f} USDT."

    async def _generate_conversational_reply(self, user_msg: str, snapshot: Dict[str, Any]) -> str:
        cfg = load_api_config()
        provider = _resolve_llm_provider(cfg)
        signals = snapshot.get("signals", [])
        macro = snapshot.get("macro", {})
        news = snapshot.get("news", {})
        portfolio = snapshot.get("portfolio", {})

        # Try configured LLM first for natural dialogue
        context = (
            f"You are a professional crypto quantitative trading agent chatting with your client.\n"
            f"Agent Mode: {self.mode} | Trade Gap: {self.trade_gap_sec}s | Paper Balance: ${portfolio.get('balance_usdt', 10000)}\n"
            f"Open Positions: {len(portfolio.get('positions', []))}\n"
            f"Live Market Signals: {[(s['symbol'], s['current_price'], s['signal'], s['rsi'], s['entry_price'], s['stop_loss'], s['target_1']) for s in signals]}\n"
            f"Macro: F&G={macro.get('fear_greed_value')} ({macro.get('fear_greed_label')}), News Sentiment={news.get('overall_news_sentiment')}\n"
            f"Top Headlines: {[a['title'] for a in news.get('articles', [])[:3]]}\n"
            f"User Message: {user_msg}\n"
            f"Answer concisely (2-4 sentences) with exact numbers from the live data above."
        )

        async with httpx.AsyncClient() as client:
            if provider == "gemini" and cfg.get("gemini_api_key"):
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.get('gemini_model', 'gemini-2.5-flash')}:generateContent?key={cfg['gemini_api_key']}"
                    r = await client.post(url, json={"contents": [{"parts": [{"text": context}]}]}, timeout=8.0)
                    if r.status_code == 200:
                        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                except Exception:
                    pass
            elif provider in ("groq", "openai"):
                key = cfg.get("groq_api_key") if provider == "groq" else cfg.get("openai_api_key")
                url = "https://api.groq.com/openai/v1/chat/completions" if provider == "groq" else "https://api.openai.com/api/v1/chat/completions"
                model = cfg.get("groq_model") if provider == "groq" else cfg.get("openai_model")
                if key:
                    try:
                        r = await client.post(
                            url,
                            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                            json={"model": model, "messages": [{"role": "user", "content": context}], "max_tokens": 220},
                            timeout=8.0,
                        )
                        if r.status_code == 200:
                            return r.json()["choices"][0]["message"]["content"].strip()
                    except Exception:
                        pass
            elif provider == "ollama":
                try:
                    r = await client.post(
                        cfg.get("ollama_url", "http://localhost:11434").rstrip("/") + "/api/generate",
                        json={"model": cfg.get("ollama_model", "llama3.2"), "prompt": context, "stream": False},
                        timeout=10.0,
                    )
                    if r.status_code == 200:
                        return r.json().get("response", "").strip()
                except Exception:
                    pass

        # Built-in professional Quant-NLP fallback response
        active = [s for s in signals if s.get("signal") != "NEUTRAL"]
        best = max(active, key=lambda x: x.get("confidence_pct", 0)) if active else (signals[0] if signals else {})
        sig_summary = (
            f"Best current setup: {best.get('symbol')} -> {best.get('signal')} @ ${best.get('current_price')} "
            f"(SL: ${best.get('stop_loss')}, TP1: ${best.get('target_1')}, RSI: {best.get('rsi')})."
            if best.get("signal") != "NEUTRAL"
            else f"Markets are currently consolidating (all tracked pairs NEUTRAL). Top RSI: {best.get('symbol')} at {best.get('rsi')}."
        )
        return (
            f"[Mode: {self.mode} | Balance: ${portfolio.get('balance_usdt', 0):.2f} | Open: {len(portfolio.get('positions', []))}] "
            f"{sig_summary} Macro F&G: {macro.get('fear_greed_value')} ({macro.get('fear_greed_label')}), "
            f"News: {news.get('overall_news_sentiment')}. "
            f"(Tip: say 'only do research', 'start auto trade 3m gap', 'track sl', or 'stop' anytime.)"
        )

    # ---------------------------------------------------------
    # PARAMETER EXTRACTORS
    # ---------------------------------------------------------
    def _extract_symbols(self, lower: str) -> Optional[List[str]]:
        found = []
        if "btc" in lower or "bitcoin" in lower:
            found.append("BTC/USDT")
        if "eth" in lower or "ethereum" in lower:
            found.append("ETH/USDT")
        if "sol" in lower or "solana" in lower:
            found.append("SOL/USDT")
        return found if found else None

    def _extract_gap_seconds(self, lower: str) -> Optional[int]:
        m = re.search(r"(\d+)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hour|hours)\b", lower)
        if not m:
            return None
        val = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("s"):
            return val
        if unit.startswith("m"):
            return val * 60
        if unit.startswith("h"):
            return val * 3600
        return None

    def _extract_amount(self, lower: str) -> Optional[float]:
        m = re.search(r"\$\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*usdt", lower)
        if not m:
            return None
        num = m.group(1) or m.group(2)
        return float(num) if num else None


trading_agent = AutonomousTradingAgent()
