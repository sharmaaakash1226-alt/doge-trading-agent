"""
Self-Learning Memory Engine for Crypto Trading Agent.
Analyzes trade outcomes, diagnoses mistake patterns from losses,
derives preventive veto rules, and blocks identical mistake setups from repeating.
"""

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_memory.json")


class TradingMemory:
    def __init__(self, storage_path: str = MEMORY_FILE):
        self.storage_path = storage_path
        self.data: Dict[str, Any] = self._load_memory()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "total_pnl_usdt": 0.0,
            "mistakes_recorded": [],
            "active_veto_rules": [],
            "symbol_stats": {},
            "learned_lessons": [],
            "last_updated": int(time.time()),
        }

    def _load_memory(self) -> Dict[str, Any]:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[TradingMemory] Error loading {self.storage_path}: {e}")
        return self._default_state()

    def save_memory(self):
        try:
            self.data["last_updated"] = int(time.time())
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"[TradingMemory] Error saving {self.storage_path}: {e}")

    # ------------------------------------------------------------------
    # POST-MORTEM MISTAKE DIAGNOSIS (LEARNING FROM LOSSES)
    # ------------------------------------------------------------------
    def record_closed_trade(self, position: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Processes a closed trade. If it resulted in a loss, diagnoses the root cause,
        creates an active veto rule, and adds a lesson so the mistake is never repeated.
        """
        pnl = float(position.get("pnl", 0.0))
        pnl_pct = float(position.get("pnl_pct", 0.0))
        symbol = position.get("symbol", "UNKNOWN")
        side = position.get("side", "LONG").upper()
        entry_price = float(position.get("entry_price", 0.0))
        exit_price = float(position.get("exit_price", entry_price))
        exit_reason = position.get("exit_reason", "STOP_LOSS_HIT")
        pos_id = position.get("id", f"trade_{int(time.time())}")

        self.data["total_trades"] += 1
        self.data["total_pnl_usdt"] = round(self.data["total_pnl_usdt"] + pnl, 2)

        # Update symbol stats
        if symbol not in self.data["symbol_stats"]:
            self.data["symbol_stats"][symbol] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        sym_stat = self.data["symbol_stats"][symbol]
        sym_stat["trades"] += 1
        sym_stat["pnl"] = round(sym_stat["pnl"] + pnl, 2)

        is_win = pnl > 0.0
        if is_win:
            self.data["winning_trades"] += 1
            sym_stat["wins"] += 1
        else:
            self.data["losing_trades"] += 1
            sym_stat["losses"] += 1

        total_t = self.data["total_trades"]
        self.data["win_rate_pct"] = round((self.data["winning_trades"] / total_t) * 100.0, 1) if total_t > 0 else 0.0

        lesson_learned = None

        # IF TRADE FAILED: DIAGNOSE THE MISTAKE PATTERN
        if not is_win:
            lesson_learned = self._diagnose_mistake(position)
            if lesson_learned:
                self.data["mistakes_recorded"].append(lesson_learned)
                self._update_or_add_veto_rule(lesson_learned)
                self.data["learned_lessons"].append({
                    "trade_id": pos_id,
                    "symbol": symbol,
                    "side": side,
                    "pnl": pnl,
                    "cause": lesson_learned["failure_cause"],
                    "rule": lesson_learned["preventive_rule"],
                    "timestamp": int(time.time()),
                })

        self.save_memory()
        return lesson_learned

    def _diagnose_mistake(self, pos: Dict[str, Any]) -> Dict[str, Any]:
        """Classifies the primary structural mistake that caused the trade loss."""
        side = pos.get("side", "LONG").upper()
        entry = float(pos.get("entry_price", 0.0))
        exit_p = float(pos.get("exit_price", entry))
        rsi = float(pos.get("rsi") or 50.0)
        atr = float(pos.get("atr") or (entry * 0.015))
        near_res = float(pos.get("nearest_resistance") or 0.0)
        near_sup = float(pos.get("nearest_support") or 0.0)
        sym = pos.get("symbol", "")

        dist_res_pct = (near_res - entry) / entry if (near_res > entry and entry > 0) else 0.05
        dist_sup_pct = (entry - near_sup) / entry if (near_sup < entry and entry > 0) else 0.05

        failure_cause = "STANDARD_VOLATILITY_LOSS"
        preventive_rule = "Maintain strict 50x risk discipline and wait for confirmed momentum alignment."
        pattern_signature = {
            "side": side,
            "symbol": sym,
            "rsi_band": "MID",
            "sr_risk": "NONE",
        }

        if side == "LONG":
            if rsi >= 65.0:
                failure_cause = "OVERBOUGHT_LONG_CHASE"
                preventive_rule = f"Do NOT enter LONG on {sym} when RSI >= {rsi:.1f} (overbought exhaustion into highs)."
                pattern_signature["rsi_band"] = "OVERBOUGHT"
            elif dist_res_pct < 0.010:
                failure_cause = "RESISTANCE_CEILING_REJECTION"
                preventive_rule = f"Do NOT enter LONG on {sym} if nearest resistance is within {dist_res_pct*100:.2f}% (< 1.0% runway)."
                pattern_signature["sr_risk"] = "RESISTANCE_TOO_CLOSE"
            elif near_sup == 0 or (entry - near_sup) > 0.025 * entry:
                failure_cause = "UNSUPPORTED_FREE_FALL"
                preventive_rule = f"Do NOT enter LONG on {sym} when no support cushion exists within 2.0% of entry."
                pattern_signature["sr_risk"] = "SUPPORT_TOO_FAR"
            else:
                failure_cause = "FALSE_BREAKOUT_LONG"
                preventive_rule = f"Require higher confirmation threshold before LONG on {sym}."
                pattern_signature["rsi_band"] = "CHOP_ZONE"

        else: # SHORT
            if rsi <= 35.0:
                failure_cause = "OVERSOLD_SHORT_CHASE"
                preventive_rule = f"Do NOT enter SHORT on {sym} when RSI <= {rsi:.1f} (shorting the bottom into oversold bounce)."
                pattern_signature["rsi_band"] = "OVERSOLD"
            elif dist_sup_pct < 0.010:
                failure_cause = "SUPPORT_FLOOR_BOUNCE"
                preventive_rule = f"Do NOT enter SHORT on {sym} if nearest support is within {dist_sup_pct*100:.2f}% (< 1.0% clearance)."
                pattern_signature["sr_risk"] = "SUPPORT_TOO_CLOSE"
            elif near_res == 0 or (near_res - entry) > 0.025 * entry:
                failure_cause = "UNCEILINGED_PUMP_SQUEEZE"
                preventive_rule = f"Do NOT enter SHORT on {sym} when no resistance ceiling protects the SL within 2.0%."
                pattern_signature["sr_risk"] = "RESISTANCE_TOO_FAR"
            else:
                failure_cause = "FALSE_BREAKDOWN_SHORT"
                preventive_rule = f"Require higher confirmation threshold before SHORT on {sym}."
                pattern_signature["rsi_band"] = "CHOP_ZONE"

        return {
            "id": f"mistake_{pos.get('id', int(time.time()))}",
            "symbol": sym,
            "side": side,
            "entry_price": entry,
            "exit_price": exit_p,
            "pnl": float(pos.get("pnl", 0.0)),
            "pnl_pct": float(pos.get("pnl_pct", 0.0)),
            "rsi_at_entry": round(rsi, 2),
            "failure_cause": failure_cause,
            "preventive_rule": preventive_rule,
            "pattern_signature": pattern_signature,
            "timestamp": int(time.time()),
        }

    def _update_or_add_veto_rule(self, mistake: Dict[str, Any]):
        """Creates or strengthens an active Veto Rule that prevents this mistake from repeating."""
        sig = mistake["pattern_signature"]
        rule_key = f"{sig['symbol']}_{sig['side']}_{mistake['failure_cause']}"

        # Check existing rule
        found = None
        for r in self.data["active_veto_rules"]:
            if r.get("key") == rule_key:
                found = r
                break

        if found:
            found["strikes"] += 1
            found["last_triggered"] = int(time.time())
            found["description"] = mistake["preventive_rule"]
        else:
            self.data["active_veto_rules"].append({
                "key": rule_key,
                "symbol": sig["symbol"],
                "side": sig["side"],
                "failure_cause": mistake["failure_cause"],
                "description": mistake["preventive_rule"],
                "rsi_at_mistake": mistake["rsi_at_entry"],
                "strikes": 1,
                "created_at": int(time.time()),
                "last_triggered": int(time.time()),
            })

    # ------------------------------------------------------------------
    # PRE-TRADE EVALUATION (DO NOT REPEAT MISTAKES)
    # ------------------------------------------------------------------
    def evaluate_candidate(
        self,
        candidate: Dict[str, Any],
        side: str,
        entry_price: float,
    ) -> Tuple[bool, str, float]:
        """
        Cross-examines a potential candidate against all recorded mistakes and active veto rules.
        Returns: (allowed: bool, reason: str, confidence_adjustment: float)
        """
        sym = candidate.get("symbol", "")
        side = side.upper()
        rsi = float(candidate.get("rsi") or 50.0)
        near_res = float(candidate.get("nearest_resistance") or 0.0)
        near_sup = float(candidate.get("nearest_support") or 0.0)

        dist_res_pct = (near_res - entry_price) / entry_price if (near_res > entry_price and entry_price > 0) else 0.05
        dist_sup_pct = (entry_price - near_sup) / entry_price if (near_sup < entry_price and entry_price > 0) else 0.05

        # 1. Check against active VETO rules
        for rule in self.data["active_veto_rules"]:
            if rule["symbol"] == sym and rule["side"] == side:
                cause = rule["failure_cause"]

                # Case: Overbought LONG Chase
                if cause == "OVERBOUGHT_LONG_CHASE" and rsi >= 64.0:
                    return (
                        False,
                        f"Memory VETO ({rule['strikes']} past strike): {rule['description']} Current RSI is {rsi:.1f}.",
                        -25.0
                    )

                # Case: Oversold SHORT Chase
                if cause == "OVERSOLD_SHORT_CHASE" and rsi <= 36.0:
                    return (
                        False,
                        f"Memory VETO ({rule['strikes']} past strike): {rule['description']} Current RSI is {rsi:.1f}.",
                        -25.0
                    )

                # Case: Resistance Ceiling Rejection for LONG
                if cause == "RESISTANCE_CEILING_REJECTION" and dist_res_pct < 0.012:
                    return (
                        False,
                        f"Memory VETO ({rule['strikes']} past strike): Resistance is only {dist_res_pct*100:.2f}% away. Past trade failed at identical resistance barrier.",
                        -20.0
                    )

                # Case: Support Floor Bounce for SHORT
                if cause == "SUPPORT_FLOOR_BOUNCE" and dist_sup_pct < 0.012:
                    return (
                        False,
                        f"Memory VETO ({rule['strikes']} past strike): Support is only {dist_sup_pct*100:.2f}% away. Past short was caught by this support bounce.",
                        -20.0
                    )

        # 2. Check symbol consecutive losses penalty
        sym_stat = self.data.get("symbol_stats", {}).get(sym)
        if sym_stat and sym_stat["trades"] >= 2:
            winrate = sym_stat["wins"] / sym_stat["trades"]
            if winrate < 0.34 and sym_stat["losses"] >= 2:
                # If symbol has > 66% loss rate, apply confidence penalty
                return (
                    True,
                    f"Memory Warning: {sym} historical win rate is low ({winrate*100:.0f}% over {sym_stat['trades']} trades). Applying +10% confidence hurdle.",
                    -10.0
                )

        # 3. Winning pattern reinforcement
        if sym_stat and sym_stat["trades"] >= 2 and (sym_stat["wins"] / sym_stat["trades"]) >= 0.65:
            winrate = sym_stat["wins"] / sym_stat["trades"]
            return (
                True,
                f"Memory Confirmed: {sym} has proven high-winrate pattern ({winrate*100:.0f}% win rate).",
                +10.0
            )

        return True, "No conflicting mistake patterns in memory. Setup approved.", 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Returns structured memory state for UI and chat briefing."""
        return {
            "total_trades": self.data.get("total_trades", 0),
            "winning_trades": self.data.get("winning_trades", 0),
            "losing_trades": self.data.get("losing_trades", 0),
            "win_rate_pct": self.data.get("win_rate_pct", 0.0),
            "total_pnl_usdt": self.data.get("total_pnl_usdt", 0.0),
            "active_veto_rules_count": len(self.data.get("active_veto_rules", [])),
            "active_veto_rules": self.data.get("active_veto_rules", []),
            "recent_lessons": self.data.get("learned_lessons", [])[-6:],
            "symbol_stats": self.data.get("symbol_stats", {}),
        }


# Global memory singleton
trading_memory = TradingMemory()
