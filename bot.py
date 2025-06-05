
#!/usr/bin/env python3
"""
Advanced Crypto AI Trading Bot – Cleaned-Up Version
Compatible with:
- python-telegram-bot ≥ 20
- PostgreSQL + Redis
- scikit-learn, XGBoost, LightGBM
"""

from __future__ import annotations
import asyncio, json, logging, os, time
from datetime import datetime
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
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s")
logger = logging.getLogger("crypto-bot")

class SecurityManager:
    def __init__(self):
        self.cipher_key = os.getenv("ENCRYPTION_KEY", Fernet.generate_key())
        self.cipher = Fernet(self.cipher_key)
        self.rate_limits: Dict[int, List[float]] = {}
        self.max_rpm = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "30"))

    def encrypt(self, raw: str) -> str:
        return self.cipher.encrypt(raw.encode()).decode()

    def decrypt(self, token: str) -> str:
        return self.cipher.decrypt(token.encode()).decode()

    def is_limited(self, user_id: int) -> bool:
        now = time.time()
        window = self.rate_limits.get(user_id, [])
        window = [t for t in window if now - t < 60]
        if len(window) >= self.max_rpm:
            self.rate_limits[user_id] = window
            return True
        window.append(now)
        self.rate_limits[user_id] = window
        return False

class DatabaseManager:
    def __init__(self):
        self._uri = os.getenv("DATABASE_URL")
        if not self._uri:
            raise RuntimeError("DATABASE_URL not set")
        self.redis = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        self._init_tables()

    def _conn(self):
        return psycopg2.connect(self._uri, cursor_factory=RealDictCursor)

    def cache_get(self, key: str):
        val = self.redis.get(key)
        return json.loads(val) if val else None

    def cache_set(self, key: str, data, ttl=300):
        if hasattr(data, 'to_dict'):
            df_dict = data.to_dict()
            self.redis.setex(key, ttl, json.dumps(self._make_json_serializable(df_dict)))
        else:
            self.redis.setex(key, ttl, json.dumps(data, default=str))

    def _make_json_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(i) for i in obj]
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif isinstance(obj, (pd.Timestamp, np.datetime64)):
            return str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

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

class CryptoDataManager:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.binance: Optional[BinanceClient] = None

    def init_binance(self, key: str, secret: str):
        try:
            self.binance = BinanceClient(key, secret)
            self.binance.ping()
            logger.info("Binance connected")
        except Exception as exc:
            logger.warning("Binance failed: %s", exc)
            self.binance = None

    async def get_ohlcv(self, symbol: str, interval: str = "1h", limit: int = 500) -> Optional[pd.DataFrame]:
        cache_key = f"ohlcv:{symbol}:{interval}:{limit}"
        from_cache = self.db.cache_get(cache_key)
        if from_cache:
            df = pd.DataFrame(from_cache)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.set_index('timestamp')

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
            ticker = symbol.replace("USDT", "-USD")
            df = yf.download(ticker, period="60d", interval=interval, progress=False)
            df.rename(columns=str.lower, inplace=True)
            if df.empty: return None

        self.db.cache_set(cache_key, df.reset_index(), ttl=300)
        return df

class EnhancedAITradingEngine:
    def __init__(self, db: DatabaseManager, market: CryptoDataManager):
        self.db = db
        self.market = market
        self.models = {}
        self.scalers = {}

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
            return

        feat = self._make_features(df)
        target = (feat["return_6"].shift(-6) > 0).astype(int)
        feat, target = feat.iloc[:-6], target.iloc[:-6]

        X_train, X_test, y_train, y_test = train_test_split(feat, target, test_size=0.2, random_state=42, stratify=target)
        scaler = MinMaxScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        models = {
            "rf": RandomForestClassifier(n_estimators=200),
            "gb": GradientBoostingClassifier(),
            "xgb": XGBClassifier(n_estimators=250, verbosity=0),
            "lgb": LGBMClassifier(n_estimators=250),
        }

        for name, mdl in models.items():
            mdl.fit(X_train_s, y_train)
        self.models[symbol] = models
        self.scalers[symbol] = scaler

    async def signal(self, symbol: str) -> Optional[Dict]:
        if symbol not in self.models:
            await self.train(symbol)
        if symbol not in self.models:
            return None

        df = await self.market.get_ohlcv(symbol, limit=100)
        if df is None: return None
        feat = self._make_features(df).iloc[-1:]
        scaler = self.scalers[symbol]
        X = scaler.transform(feat)

        probs = {
            name: mdl.predict_proba(X)[0][1] if hasattr(mdl, "predict_proba") else mdl.predict(X)[0]
            for name, mdl in self.models[symbol].items()
        }
        ensemble = sum(probs.values()) / len(probs)
        signal_type = "BUY" if ensemble > 0.6 else "SELL" if ensemble < 0.4 else "HOLD"
        return {
            "symbol": symbol,
            "prob": ensemble,
            "signal": signal_type,
            "price": float(df['close'].iloc[-1]),
            "confidence": abs(ensemble - 0.5) * 2,
        }

class TelegramCryptoBot:
    popular = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]

    def __init__(self, token: str):
        self.sec = SecurityManager()
        self.db = DatabaseManager()
        self.market = CryptoDataManager(self.db)
        self.ai = EnhancedAITradingEngine(self.db, self.market)
        self.scheduler = AsyncIOScheduler()
        self.app = Application.builder().token(token).build()

        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("signals", self.signals))
        self.app.add_handler(CommandHandler("help", self.help))
        self.app.add_handler(CallbackQueryHandler(self.cb))
        self.app.add_error_handler(self.err)

    async def start(self, upd: Update, ctx: ContextTypes.DEFAULT_TYPE):
        uid = upd.effective_user.id
        if self.sec.is_limited(uid):
            await upd.effective_message.reply_text("🚫 Too many requests – cool down 1 min.")
            return
        await upd.effective_message.reply_text(
            "👋 Welcome to *Crypto AI Bot*! Use /signals to get the latest calls.",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 Signals", callback_data="signals")],
            ]),
        )

    async def help(self, upd: Update, ctx):
        await upd.effective_message.reply_text("Commands: /start /signals /help")

    async def signals(self, upd: Update, ctx):
        msg = await upd.effective_message.reply_text("Generating signals…")
        texts = []
        for sym in self.popular:
            sig = await self.ai.signal(sym)
            if not sig: continue
            emoji = "🟢" if sig["signal"] == "BUY" else "🔴" if sig["signal"] == "SELL" else "🟡"
            texts.append(f"{emoji} *{sym}* → {sig['signal']} ({sig['confidence']*100:.1f}% @ {sig['price']:.2f})")
                await msg.edit_text("\n".join(texts) if texts else "No valid signals right now.", parse_mode="Markdown")
    async def cb(self, upd: Update, ctx):
        await upd.callback_query.answer()
        if upd.callback_query.data == "signals":
            await self.signals(upd, ctx)

    async def err(self, upd: object, ctx):
        logger.error("Error: %s", ctx.error)

    async def broadcast_signals(self):
        with self.db._conn() as c, c.cursor() as cur:
            cur.execute("SELECT user_id FROM users")
            users = [r["user_id"] for r in cur.fetchall()]
        for uid in users:
            try:
                await self.app.bot.send_message(uid, "⏰ New signals available! Use /signals")
            except Exception as exc:
                logger.debug("DM failed: %s", exc)

    def run(self):
        self.scheduler.add_job(self.broadcast_signals, "interval", hours=4)
        self.scheduler.start()
        self.app.run_polling()

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing")
    bot = TelegramCryptoBot(token)

    if os.getenv("BINANCE_API_KEY") and os.getenv("BINANCE_API_SECRET"):
        bot.market.init_binance(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))

    loop = asyncio.get_event_loop()
    for sym in TelegramCryptoBot.popular[:3]:
        loop.run_until_complete(bot.ai.train(sym))

    bot.run()

if __name__ == "__main__":
    main()
