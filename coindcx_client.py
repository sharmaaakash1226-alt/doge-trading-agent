"""
CoinDCX Futures & Spot API Client for the Trading Agent.
Implements HMAC SHA-256 authentication, order execution, position tracking,
and strict risk safety limits.
"""

import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional
import httpx

COINDCX_BASE_URL = "https://api.coindcx.com"

def load_env_file():
    """Loads key-value pairs from trading_agent/.env if present."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception as e:
            print(f"Error loading .env: {e}")

load_env_file()

def format_coindcx_quantity(symbol: str, raw_qty: float) -> float:
    """Rounds order quantity to exchange step size requirements."""
    sym = symbol.upper()
    if any(k in sym for k in ["DOGE", "BONK", "PEPE", "SHIB", "FLOKI"]):
        return float(int(raw_qty))
    elif "BTC" in sym:
        return round(raw_qty, 3)
    elif any(k in sym for k in ["ETH", "SOL", "BNB"]):
        return round(raw_qty, 2)
    elif any(k in sym for k in ["XRP", "ADA", "MATIC", "DOT"]):
        return round(raw_qty, 1)
    else:
        return round(raw_qty, 2)


class CoinDCXClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        live_trading: Optional[bool] = None,
    ):
        load_env_file()
        self.api_key = api_key or os.environ.get("COINDCX_API_KEY", "").strip()
        self.api_secret = api_secret or os.environ.get("COINDCX_API_SECRET", "").strip()
        self.server_time_offset: float = 0.0  # Drift adjustment in seconds
        self._client: Optional[httpx.AsyncClient] = None
        # HARD SAFETY LATCH: Defaults to False unless explicitly set to 'true' in .env
        if live_trading is not None:
            self.live_trading_enabled = live_trading
        else:
            self.live_trading_enabled = os.environ.get("COINDCX_LIVE_TRADING", "false").lower() == "true"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    async def sync_server_time(self) -> float:
        """Calculates clock drift between local machine and CoinDCX servers."""
        try:
            import email.utils
            client = self._get_client()
            resp = await client.get(
                f"{COINDCX_BASE_URL}/exchange/ticker",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            server_date = resp.headers.get("date")
            if server_date:
                server_ts = email.utils.parsedate_to_datetime(server_date).timestamp()
                self.server_time_offset = server_ts - time.time()
                return self.server_time_offset
        except Exception:
            pass
        return 0.0

    def _generate_signature(self, body_dict: Dict[str, Any]) -> tuple[str, str]:
        """Generates epoch millisecond timestamp and HMAC-SHA256 hex signature."""
        body_dict["timestamp"] = int(round((time.time() + self.server_time_offset) * 1000))
        json_payload = json.dumps(body_dict, separators=(",", ":"))
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            json_payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return json_payload, signature

    async def _post_authenticated(self, endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_configured:
            return {"status": "error", "message": "CoinDCX API Key and Secret are not configured."}

        json_payload, signature = self._generate_signature(body)
        headers = {
            "Content-Type": "application/json",
            "X-AUTH-APIKEY": self.api_key,
            "X-AUTH-SIGNATURE": signature,
        }
        url = f"{COINDCX_BASE_URL}{endpoint}"

        client = self._get_client()
        try:
            resp = await client.post(url, content=json_payload, headers=headers)
            if resp.status_code in (200, 201):
                return {"status": "success", "data": resp.json()}
            return {
                "status": "error",
                "code": resp.status_code,
                "message": resp.text,
            }
        except Exception as e:
            return {"status": "error", "message": f"Connection error: {str(e)}"}

    async def get_futures_wallets(self) -> Dict[str, Any]:
        """Fetches live CoinDCX USDT-M Futures margin wallet balance."""
        if not self.is_configured:
            return {"status": "error", "message": "CoinDCX API Key and Secret are not configured."}
        body = {"timestamp": int(round((time.time() + self.server_time_offset) * 1000))}
        json_payload, signature = self._generate_signature(body)
        headers = {
            "Content-Type": "application/json",
            "X-AUTH-APIKEY": self.api_key,
            "X-AUTH-SIGNATURE": signature,
        }
        url = f"{COINDCX_BASE_URL}/exchange/v1/derivatives/futures/wallets"
        client = self._get_client()
        try:
            resp = await client.request("GET", url, content=json_payload, headers=headers)
            if resp.status_code == 200:
                return {"status": "success", "data": resp.json()}
            # Try POST as fallback
            resp2 = await client.post(url, content=json_payload, headers=headers)
            if resp2.status_code == 200:
                return {"status": "success", "data": resp2.json()}
            return {"status": "error", "code": resp.status_code, "message": resp.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_futures_balances(self) -> Dict[str, Any]:
        """Fetches live CoinDCX USDT-M Futures margin wallet balance."""
        body = {}
        return await self._post_authenticated("/exchange/v1/derivatives/futures/positions", body)

    async def place_futures_order(
        self,
        symbol: str,         # e.g., "B-BTC_USDT"
        side: str,           # "buy" (LONG) or "sell" (SHORT)
        order_type: str,     # "market_order" or "limit_order"
        quantity: float,
        price: Optional[float] = None,
        leverage: float = 50.0,
        stop_loss_price: Optional[float] = None,
        target_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes a real order on CoinDCX Futures if live_trading_enabled is True.
        """
        if not self.live_trading_enabled:
            return {
                "status": "simulated",
                "message": "LIVE_TRADING_ENABLED is False. Safety latch prevented real execution.",
                "details": {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "leverage": leverage
                }
            }

        clean_qty = format_coindcx_quantity(symbol, quantity)
        order_dict: Dict[str, Any] = {
            "pair": symbol,
            "side": "buy" if side.lower() in ("buy", "long") else "sell",
            "order_type": order_type.lower(),
            "total_quantity": clean_qty,
            "leverage": float(leverage),
            "position_margin_type": "isolated",
        }
        if price:
            order_dict["price"] = price
        if stop_loss_price:
            order_dict["stop_price"] = stop_loss_price
        if target_price:
            order_dict["target"] = target_price

        body = {"order": order_dict}
        return await self._post_authenticated("/exchange/v1/derivatives/futures/orders/create", body)

    async def exit_position(
        self,
        symbol: str,
        side: str = "LONG",
        quantity: Optional[float] = None,
        leverage: float = 50.0
    ) -> Dict[str, Any]:
        """Closes an active futures position on CoinDCX at market."""
        if not self.live_trading_enabled:
            return {"status": "simulated", "message": f"Safety latch active. Simulated close of {symbol}."}

        close_side = "sell" if side.upper() in ("BUY", "LONG") else "buy"
        order_dict: Dict[str, Any] = {
            "pair": symbol,
            "side": close_side,
            "order_type": "market_order",
            "leverage": float(leverage),
            "position_margin_type": "isolated",
        }
        if quantity:
            order_dict["total_quantity"] = format_coindcx_quantity(symbol, quantity)

        body = {"order": order_dict}
        return await self._post_authenticated("/exchange/v1/derivatives/futures/orders/create", body)
