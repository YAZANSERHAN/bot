#!/usr/bin/env python3
"""
Advanced Crypto AI Trading Bot  – cleaned-up, single-file version
================================================================
• Async-first, compatible with `python-telegram-bot` ≥ 20
• No TensorFlow/Keras – only scikit-learn + (optional) XGBoost / LightGBM
• Modular classes: SecurityManager ▸ DatabaseManager ▸ CryptoDataManager ▸
  EnhancedAITradingEngine ▸ TelegramCryptoBot
• Uses PostgreSQL + Redis (cache) + apscheduler
• Designed to run on a small VPS (2 vCPU / 4 GB) without GPU
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import aiohttp
import numpy as np
import pandas as pd
import psycopg2
import redis
import yfinance as yf
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from binance.client import Client as BinanceClient
from cryptography.fernet import Fernet
from psycopg2.extras import RealDictCursor
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,
                          ContextTypes)

# ── ML libs (all CPU-friendly) ────────────────────────────────────────────────
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recFall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
)
logger = logging.getLogger("crypto-bot")

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────

class SecurityManager:
    """Simple encryption + rate-limiting."""

    def __init__(self) -> None:
        self.cipher_key: bytes = os.getenv("ENCRYPTION_KEY", Fernet.generate_key())  # type: ignore[arg-type]
        self.cipher = Fernet(self.cipher_key)
        self.rate_limits: Dict[int, List[float]] = {}
        self.max_rpm = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "30"))

    # Encryption helpers -------------------------------------------------------
    def encrypt(self, raw: str) -> str:
        return self.cipher.encrypt(raw.encode()).decode()

    def decrypt(self, token: str) -> str:
        return self.cipher.decrypt(token.encode()).decode()

    # Rate-limit ---------------------------------------------------------------
    def is_limited(self, user_id: int) -> bool:
        now = time.time()
        window = self.rate_limits.get(user_id, [])
        window = [t for t in window if now - t < 60]  # keep last minute
        if len(window) >= self.max_rpm:
            self.rate_limits[user_id] = window  # prune
            return True
        window.append(now)
        self.rate_limits[user_id] = window
        return False

# ───────────────────────────────────────────────────────────────────────────────
# Database + cache
# ───────────────────────────────────────────────────────────────────────────────

class DatabaseManager:
    def __init__(self) -> None:
        self._uri = os.getenv("DATABASE_URL")
        if not self._uri:
            raise RuntimeError("DATABASE_URL not set")
        self.redis: redis.Redis = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        self._init_tables()

    def _conn(self):
        return psycopg2.connect(self._uri, cursor_factory=RealDictCursor)

    # Basic cache --------------------------------------------------------------
    def cache_get(self, key: str):
        val = self.redis.get(key)
        return json.loads(val) if val else None

    def cache_set(self, key: str, data, ttl=300):
        # Handle pandas DataFrame serialization properly
        if hasattr(data, 'to_dict'):
            # Convert DataFrame to dict and handle timestamps
            df_dict = data.to_dict()
            # Convert any datetime/timestamp objects to ISO strings
            serializable_dict = self._make_json_serializable(df_dict)
            self.redis.setex(key, ttl, json.dumps(serializable_dict))
        else:
            self.redis.setex(key, ttl, json.dumps(data, default=str))

    def _make_json_serializable(self, obj):
        """Recursively convert pandas Timestamps and other non-JSON types to strings"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # datetime-like objects
            return obj.isoformat()
        elif isinstance(obj, (pd.Timestamp, np.datetime64)):
            return str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

    # Table bootstrap ----------------------------------------------------------
    def _init_tables(self):
        ddl = """
        CREATE TABLE IF NOT EXISTS users (
            user_id BIGINT PRIMARY KEY,
            username_hash TEXT,
            subscription_end TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS signals (
            id SERIAL PRIMARY KEY,
            symbol TEXT,
            signal_type TEXT,
            confidence REAL,
            price NUMERIC,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        with self._conn() as c, c.cursor() as cur:
            cur.execute(ddl)
        logger.info("DB schema ensured")

# ───────────────────────────────────────────────────────────────────────────────
# Market data
# ───────────────────────────────────────────────────────────────────────────────

class CryptoDataManager:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.binance: Optional[BinanceClient] = None

    # Optional Binance initialisation -----------------------------------------
    def init_binance(self, key: str, secret: str):
        try:
            self.binance = BinanceClient(key, secret)
            _ = self.binance.ping()
            logger.info("Binance connection OK")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Binance init failed → yfinance fallback (%s)", exc)
            self.binance = None

    # Fetch OHLCV --------------------------------------------------------------
    async def get_ohlcv(self, symbol: str, interval: str = "1h", limit: int = 500) -> Optional[pd.DataFrame]:
        cache_key = f"ohlcv:{symbol}:{interval}:{limit}"
        from_cache = self.db.cache_get(cache_key)
        if from_cache:
            df = pd.DataFrame(from_cache)
            # Convert string timestamps back to datetime index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            elif df.index.name == 'timestamp' or len(df.index) > 0:
                df.index = pd.to_datetime(df.index)
            return df

        if self.binance:
            klines = self.binance.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "qav", "trades", "tbav", "tqav", "ignore",
            ])
            df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        else:
            # yfinance expects e.g. BTC-USD
            ticker = symbol.replace("USDT", "-USD")
            df = yf.download(ticker, period="60d", interval=interval, progress=False)
            df.rename(columns=str.lower, inplace=True)
            if df.empty:
                return None

        # Cache the DataFrame - reset index to make timestamp a column for JSON serialization
        cache_df = df.reset_index()
        self.db.cache_set(cache_key, cache_df, ttl=300)
        return df

# ───────────────────────────────────────────────────────────────────────────────
# ML Engine
# ───────────────────────────────────────────────────────────────────────────────

class EnhancedAITradingEngine:
    FEATURES = ["close", "volume"]  # keep simple → extensible

    def __init__(self, db: DatabaseManager, market: CryptoDataManager):
        self.db = db
        self.market = market
        self.models: Dict[str, Dict[str, object]] = {}
        self.scalers: Dict[str, MinMaxScaler] = {}

    # Utility ------------------------------------------------------------------
    @staticmethod
    def _make_features(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["close"] = df["close"]
        out["volume"] = df["volume"]
        out["return_1"] = df["close"].pct_change().fillna(0)
        out["return_6"] = df["close"].pct_change(6).fillna(0)
        return out.fillna(0)

    async def train(self, symbol: str):
        df = await self.market.get_ohlcv(symbol, limit=1500)
        if df is None or len(df) < 200:
            logger.warning("Not enough data for %s", symbol)
            return

        feat = self._make_features(df)
        target = (feat["return_6"].shift(-6) > 0).astype(int)  # 6-step ahead up/down
        feat, target = feat.iloc[:-6], target.iloc[:-6]

        X_train, X_test, y_train, y_test = train_test_split(feat, target, test_size=0.2, random_state=42, stratify=target)

        scaler = MinMaxScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        models = {
            "rf": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
            "gb": GradientBoostingClassifier(random_state=42),
            "xgb": XGBClassifier(n_estimators=250, random_state=42, n_jobs=-1, verbosity=0),
            "lgb": LGBMClassifier(n_estimators=250, random_state=42),
        }
        perf = {}
        for name, mdl in models.items():
            mdl.fit(X_train_s, y_train)
            y_pred = mdl.predict(X_test_s)
            perf[name] = {
                "acc": accuracy_score(y_test, y_pred),
                "prec": precision_score(y_test, y_pred, zero_division=0),
                "rec": recall_score(y_test, y_pred, zero_division=0),
            }
            logger.info("%s | %s → acc=%.3f", symbol, name, perf[name]["acc"])

        self.models[symbol] = models
        self.scalers[symbol] = scaler

    async def signal(self, symbol: str) -> Optional[Dict]:
        if symbol not in self.models:
            await self.train(symbol)
            if symbol not in self.models:
                return None

        df = await self.market.get_ohlcv(symbol, limit=100)
        if df is None:
            return None
        feat = self._make_features(df).iloc[-1:]
        scaler = self.scalers[symbol]
        X = scaler.transform(feat)
        probs = {
            name: mdl.predict_proba(X)[0][1] if hasattr(mdl, "predict_proba") else mdl.predict(X)[0]
            for name, mdl in self.models[symbol].items()
        }
        # equal-weight ensemble
        ensemble = sum(probs.values()) / len(probs)
        signal_type = "BUY" if ensemble > 0.6 else "SELL" if ensemble < 0.4 else "HOLD"
        return {
            "symbol": symbol,
            "prob": ensemble,
            "signal": signal_type,
            "price": float(df["close"].iloc[-1]),
            "confidence": abs(ensemble - 0.5) * 2,
        }

# ───────────────────────────────────────────────────────────────────────────────
# Telegram bot
# ───────────────────────────────────────────────────────────────────────────────

class TelegramCryptoBot:
    popular = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
    ]

    def __init__(self, token: str):
        self.sec = SecurityManager()
        self.db = DatabaseManager()
        self.market = CryptoDataManager(self.db)
        self.ai = EnhancedAITradingEngine(self.db, self.market)
        self.scheduler = AsyncIOScheduler()
        self.app = Application.builder().token(token).build()

        # Handlers -------------------------------------------------------------
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("signals", self.signals))
        self.app.add_handler(CommandHandler("help", self.help))
        self.app.add_handler(CallbackQueryHandler(self.cb))
        self.app.add_error_handler(self.err)

    # ── Commands ────────────────────────────────────────────────────────────
    async def start(self, upd: Update, ctx: ContextTypes.DEFAULT_TYPE):
        uid = upd.effective_user.id
        if self.sec.is_limited(uid):
            await upd.effective_message.reply_text("🚫 Too many requests – cool down 1 min.")
            return
        await upd.effective_message.reply_text(
            "👋 Welcome to *Crypto AI Bot*!  Use /signals to get the latest calls.",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 Signals", callback_data="signals")],
            ]),
        )

    async def help(self, upd: Update, ctx):  # pylint: disable=unused-argument
        await upd.effective_message.reply_text("Available commands: /start /signals /help")

    async def signals(self, upd: Update, ctx):  # pylint: disable=unused-argument
        msg = await upd.effective_message.reply_text("Generating signals…")
        texts: List[str] = []
        for sym in self.popular:
            sig = await self.ai.signal(sym)
            if not sig:
                continue
            emoji = "🟢" if sig["signal"] == "BUY" else "🔴" if sig["signal"] == "SELL" else "🟡"
            texts.append(f"{emoji} *{sym}* → {sig['signal']} (conf {sig['confidence']*100:.1f}% @ {sig['price']:.2f})")
        await msg.edit_text("\n".join(texts) if texts else "No valid signals right now.", parse_mode="Markdown")

    # ── Callback buttons ────────────────────────────────────────────────────
    async def cb(self, upd: Update, ctx):  # pylint: disable=unused-argument
        data = upd.callback_query.data  # type: ignore[attr-defined]
        await upd.callback_query.answer()
        if data == "signals":
            await self.signals(upd, ctx)

    # ── Error handler ───────────────────────────────────────────────────────
    async def err(self, upd: object, ctx):  # pylint: disable=unused-argument
        logger.error("Update %s caused error %s", upd, ctx.error)  # type: ignore[attr-defined]

    # ── Scheduled broadcast to all users in DB (demo) ───────────────────────
    async def broadcast_signals(self):
        logger.info("Running scheduled broadcast …")
        with self.db._conn() as c, c.cursor() as cur:
            cur.execute("SELECT user_id FROM users")
            users = [row["user_id"] for row in cur.fetchall()]
        for uid in users:
            try:
                await self.app.bot.send_message(uid, "⏰ New signals available!  Use /signals")
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("cannot DM %s: %s", uid, exc)

    # ── Run bot ─────────────────────────────────────────────────────────────
    def run(self):
        logger.info("Bot polling …")
        
        # Initialize scheduler now that we have an event loop
        self.scheduler = AsyncIOScheduler()
        self.scheduler.add_job(self.broadcast_signals, "interval", hours=4)
        self.scheduler.start()
        
        self.app.run_polling()

# ───────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ───────────────────────────────────────────────────────────────────────────────

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN env var missing")

    bot = TelegramCryptoBot(token)

    # Optional Binance API keys (only if available in your .env)
    if os.getenv("BINANCE_API_KEY") and os.getenv("BINANCE_API_SECRET"):
        bot.market.init_binance(
            os.getenv("BINANCE_API_KEY"),
            os.getenv("BINANCE_API_SECRET")
        )

    # Async setup and run
    async def bootstrap():
        for sym in TelegramCryptoBot.popular[:3]:
            await bot.ai.train(sym)
        bot.scheduler.start()
        bot.run()

    asyncio.run(bootstrap())


if __name__ == "__main__":
    main()

