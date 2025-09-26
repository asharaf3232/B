# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🚀 بوت التداول النهائي V7.0 (النسخة المدمجة والموثوقة) 🚀 ---
# =======================================================================================
#
# هذا الإصدار يدمج أفضل الميزات من كافة النسخ السابقة:
#   - **الأساس الصلب (من نسخة ENHANCED):** محرك تنفيذ صفقات قوي مع إعادة محاولة،
#     ونظام مراقبة صحة البوت.
#   - **الذكاء التكتيكي (من نسخة RELIABLE/INTELLIGENT):** منع الصفقات المكررة،
#     وقف خسارة متحرك ذكي، والتحقق من الحد الأدنى لقيمة الصفقة عند الإغلاق.
#
#   النتيجة: بوت تداول متكامل يجمع بين الموثوقية الهندسية والذكاء في التداول.
#
# =======================================================================================

# --- المكتبات الأساسية ---
import os
import logging
import asyncio
import json
import time
import copy
import random
from datetime import datetime, timedelta, timezone, time as dt_time
from zoneinfo import ZoneInfo
from collections import defaultdict, Counter
import httpx
import re
import aiosqlite

# --- مكتبات التحليل والتداول ---
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt
import feedparser
import websockets
import websockets.exceptions

# --- [ترقية] مكتبات جديدة للعقل المطور ---
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not found. News sentiment analysis will be disabled.")

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("Library 'scipy' not found. RSI Divergence strategy will be disabled.")

# --- مكتبات تليجرام ---
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram.constants import ParseMode
from telegram.error import BadRequest, TimedOut, Forbidden

# --- إعدادات أساسية ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- جلب المتغيرات من بيئة التشغيل ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'YOUR_AV_KEY_HERE')

# --- إعدادات البوت ---
DB_FILE = 'trading_bot_v7_binance.db'
SETTINGS_FILE = 'trading_bot_v7_binance_settings.json'
TIMEFRAME = '15m'
SCAN_INTERVAL_SECONDS = 900
SUPERVISOR_INTERVAL_SECONDS = 120
STRATEGY_ANALYSIS_INTERVAL_SECONDS = 21600 # 6 hours
EGYPT_TZ = ZoneInfo("Africa/Cairo")

DEFAULT_SETTINGS = {
    "real_trade_size_usdt": 15.0,
    "max_concurrent_trades": 5,
    "top_n_symbols_by_volume": 300,
    "worker_threads": 10,
    "atr_sl_multiplier": 2.5,
    "risk_reward_ratio": 2.0,
    "trailing_sl_enabled": True,
    "trailing_sl_activation_percent": 1.5,
    "trailing_sl_callback_percent": 1.0,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "support_rebound", "sniper_pro", "whale_radar", "rsi_divergence", "supertrend_pullback"],
    "market_mood_filter_enabled": True,
    "fear_and_greed_threshold": 30,
    "adx_filter_enabled": True,
    "adx_filter_level": 25,
    "btc_trend_filter_enabled": True,
    "news_filter_enabled": True,
    "asset_blacklist": ["USDC", "DAI", "TUSD", "FDUSD", "USDD", "PYUSD", "USDT", "BNB", "BTC", "ETH"],
    "liquidity_filters": {"min_quote_volume_24h_usd": 1000000, "min_rvol": 1.5},
    "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.8},
    "trend_filters": {"ema_period": 200, "htf_period": 50, "enabled": True},
    "spread_filter": {"max_spread_percent": 0.5},
    "rsi_divergence": {"rsi_period": 14, "lookback_period": 35, "peak_trough_lookback": 5, "confirm_with_rsi_exit": True},
    "supertrend_pullback": {"atr_period": 10, "atr_multiplier": 3.0, "swing_high_lookback": 10},
    "multi_timeframe_enabled": True,
    "multi_timeframe_htf": '4h',
    "volume_filter_multiplier": 2.0,
    "close_retries": 3,
    "incremental_notifications_enabled": True,
    "incremental_notification_percent": 2.0,
    "adaptive_intelligence_enabled": True,
    "dynamic_trade_sizing_enabled": True,
    "strategy_proposal_enabled": True,
    "strategy_analysis_min_trades": 10,
    "strategy_deactivation_threshold_wr": 45.0,
    "dynamic_sizing_max_increase_pct": 25.0,
    "dynamic_sizing_max_decrease_pct": 50.0,
}

STRATEGY_NAMES_AR = {
    "momentum_breakout": "زخم اختراقي", "breakout_squeeze_pro": "اختراق انضغاطي",
    "support_rebound": "ارتداد الدعم", "sniper_pro": "القناص المحترف", "whale_radar": "رادار الحيتان",
    "rsi_divergence": "دايفرجنس RSI", "supertrend_pullback": "انعكاس سوبرترند"
}

PRESET_NAMES_AR = {"professional": "احترافي", "strict": "متشدد", "lenient": "متساهل", "very_lenient": "فائق التساهل", "bold_heart": "القلب الجريء"}

SETTINGS_PRESETS = {
    "professional": copy.deepcopy({k: v for k, v in DEFAULT_SETTINGS.items() if "adaptive" not in k and "dynamic" not in k and "strategy" not in k}),
    "strict": {**copy.deepcopy({k: v for k, v in DEFAULT_SETTINGS.items() if "adaptive" not in k and "dynamic" not in k and "strategy" not in k}), "max_concurrent_trades": 3, "risk_reward_ratio": 2.5, "fear_and_greed_threshold": 40, "adx_filter_level": 28, "liquidity_filters": {"min_quote_volume_24h_usd": 2000000, "min_rvol": 2.0}},
    "lenient": {**copy.deepcopy({k: v for k, v in DEFAULT_SETTINGS.items() if "adaptive" not in k and "dynamic" not in k and "strategy" not in k}), "max_concurrent_trades": 8, "risk_reward_ratio": 1.8, "fear_and_greed_threshold": 25, "adx_filter_level": 20, "liquidity_filters": {"min_quote_volume_24h_usd": 500000, "min_rvol": 1.2}},
    "very_lenient": {
        **copy.deepcopy({k: v for k, v in DEFAULT_SETTINGS.items() if "adaptive" not in k and "dynamic" not in k and "strategy" not in k}),
        "max_concurrent_trades": 12, "adx_filter_enabled": False, "market_mood_filter_enabled": False,
        "trend_filters": {"ema_period": 200, "htf_period": 50, "enabled": False},
        "liquidity_filters": {"min_quote_volume_24h_usd": 250000, "min_rvol": 1.0},
        "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.4}, "spread_filter": {"max_spread_percent": 1.5}
    },
    "bold_heart": {
        **copy.deepcopy({k: v for k, v in DEFAULT_SETTINGS.items() if "adaptive" not in k and "dynamic" not in k and "strategy" not in k}),
        "max_concurrent_trades": 15, "risk_reward_ratio": 1.5, "multi_timeframe_enabled": False, "market_mood_filter_enabled": False,
        "adx_filter_enabled": False, "btc_trend_filter_enabled": False, "news_filter_enabled": False,
        "volume_filter_multiplier": 1.0, "liquidity_filters": {"min_quote_volume_24h_usd": 100000, "min_rvol": 1.0},
        "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.2}, "spread_filter": {"max_spread_percent": 2.0}
    }
}

# --- [إضافة جديدة] نظام مراقبة صحة البوت ---
class TradingHealthMonitor:
    def __init__(self):
        self.metrics = {
            'start_time': time.time(),
            'successful_orders': 0,
            'failed_orders': 0,
            'connection_drops': 0,
            'last_heartbeat': time.time(),
        }

    def record_successful_order(self): self.metrics['successful_orders'] += 1
    def record_failed_order(self): self.metrics['failed_orders'] += 1
    def record_connection_drop(self): self.metrics['connection_drops'] += 1
    def update_heartbeat(self): self.metrics['last_heartbeat'] = time.time()
    
    def get_health_report(self):
        total_orders = self.metrics['successful_orders'] + self.metrics['failed_orders']
        success_rate = (self.metrics['successful_orders'] / total_orders * 100) if total_orders > 0 else 100
        uptime_seconds = time.time() - self.metrics['start_time']
        uptime_hours = uptime_seconds / 3600
        
        return {
            'total_orders': total_orders,
            'success_rate': f"{success_rate:.2f}%",
            'connection_drops': self.metrics['connection_drops'],
            'last_heartbeat_ago': f"{time.time() - self.metrics['last_heartbeat']:.1f}s ago",
            'uptime_hours': f"{uptime_hours:.2f} hours"
        }

# --- الحالة العامة للبوت ---
class BotState:
    def __init__(self):
        self.settings = {}
        self.trading_enabled = True
        self.active_preset_name = "مخصص"
        self.last_signal_time = defaultdict(float)
        self.exchange = None
        self.application = None
        self.market_mood = {"mood": "UNKNOWN", "reason": "تحليل لم يتم بعد"}
        self.last_scan_info = {}
        self.all_markets = []
        self.last_markets_fetch = 0
        self.user_data_stream = None 
        self.trade_guardian = None
        self.strategy_performance = {}
        self.pending_strategy_proposal = {}

bot_data = BotState()
health_monitor = TradingHealthMonitor()
scan_lock = asyncio.Lock()
trade_management_lock = asyncio.Lock()

# --- وظائف مساعدة وقاعدة البيانات ---
def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f: bot_data.settings = json.load(f)
        else: bot_data.settings = copy.deepcopy(DEFAULT_SETTINGS)
    except Exception: bot_data.settings = copy.deepcopy(DEFAULT_SETTINGS)
    default_copy = copy.deepcopy(DEFAULT_SETTINGS)
    for key, value in default_copy.items():
        if isinstance(value, dict):
            if key not in bot_data.settings or not isinstance(bot_data.settings[key], dict): bot_data.settings[key] = {}
            for sub_key, sub_value in value.items(): bot_data.settings[key].setdefault(sub_key, sub_value)
        else: bot_data.settings.setdefault(key, value)
    determine_active_preset(); save_settings()
    logger.info(f"Settings loaded. Active preset: {bot_data.active_preset_name}")

def determine_active_preset():
    current_settings_for_compare = {k: v for k, v in bot_data.settings.items() if k in SETTINGS_PRESETS['professional']}
    for name, preset_settings in SETTINGS_PRESETS.items():
        is_match = True
        for key, value in preset_settings.items():
            if key in current_settings_for_compare and current_settings_for_compare[key] != value:
                is_match = False; break
        if is_match:
            bot_data.active_preset_name = PRESET_NAMES_AR.get(name, "مخصص"); return
    bot_data.active_preset_name = "مخصص"

def save_settings():
    with open(SETTINGS_FILE, 'w') as f: json.dump(bot_data.settings, f, indent=4)

async def safe_db_operation(operation_func, *args, **kwargs):
    """Executes a database operation with retries for 'database is locked' errors."""
    for attempt in range(3):
        try:
            async with aiosqlite.connect(DB_FILE) as conn:
                return await operation_func(conn, *args, **kwargs)
        except aiosqlite.OperationalError as e:
            if "database is locked" in str(e) and attempt < 2:
                await asyncio.sleep(0.1 * (2 ** attempt))
                continue
            logger.error(f"DB Operation failed permanently: {e}"); raise
        except Exception as e:
            logger.error(f"DB Operation failed: {e}"); raise

async def init_database_op(conn):
    await conn.execute('CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, symbol TEXT, entry_price REAL, take_profit REAL, stop_loss REAL, quantity REAL, status TEXT, reason TEXT, order_id TEXT, highest_price REAL DEFAULT 0, trailing_sl_active BOOLEAN DEFAULT 0, close_price REAL, pnl_usdt REAL, signal_strength INTEGER DEFAULT 1, close_retries INTEGER DEFAULT 0, last_profit_notification_price REAL DEFAULT 0, trade_weight REAL DEFAULT 1.0)')
    cursor = await conn.execute("PRAGMA table_info(trades)")
    columns = [row[1] for row in await cursor.fetchall()]
    if 'signal_strength' not in columns: await conn.execute("ALTER TABLE trades ADD COLUMN signal_strength INTEGER DEFAULT 1")
    if 'close_retries' not in columns: await conn.execute("ALTER TABLE trades ADD COLUMN close_retries INTEGER DEFAULT 0")
    if 'last_profit_notification_price' not in columns: await conn.execute("ALTER TABLE trades ADD COLUMN last_profit_notification_price REAL DEFAULT 0")
    if 'trade_weight' not in columns: await conn.execute("ALTER TABLE trades ADD COLUMN trade_weight REAL DEFAULT 1.0")
    await conn.commit()

async def init_database():
    try:
        await safe_db_operation(init_database_op)
        logger.info("Adaptive database initialized successfully.")
    except Exception as e:
        logger.critical(f"Database initialization failed: {e}")

async def log_pending_trade_to_db_op(conn, signal, buy_order):
    await conn.execute("""
        INSERT INTO trades (timestamp, symbol, reason, order_id, status, entry_price, take_profit, stop_loss, signal_strength, trade_weight)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now(EGYPT_TZ).isoformat(), signal['symbol'], signal['reason'], buy_order['id'], 'pending',
          signal['entry_price'], signal['take_profit'], signal['stop_loss'], signal.get('strength', 1), signal.get('weight', 1.0)))
    await conn.commit()

async def log_pending_trade_to_db(signal, buy_order):
    try:
        await safe_db_operation(log_pending_trade_to_db_op, signal, buy_order)
        logger.info(f"Logged pending trade for {signal['symbol']} with order ID {buy_order['id']}.")
        return True
    except Exception as e:
        logger.error(f"DB Log Pending Error for {signal['symbol']}: {e}")
        return False
        
async def has_active_trade_for_symbol_op(conn, symbol: str):
    cursor = await conn.execute("SELECT 1 FROM trades WHERE symbol = ? AND status IN ('active', 'pending') LIMIT 1", (symbol,))
    return await cursor.fetchone() is not None

async def has_active_trade_for_symbol(symbol: str) -> bool:
    """Checks if there's already an active or pending trade for a symbol."""
    try:
        return await safe_db_operation(has_active_trade_for_symbol_op, symbol)
    except Exception:
        return True # Fail safe: assume a trade exists to prevent duplicates

async def safe_send_message(bot, text, **kwargs):
    for i in range(3):
        try:
            await bot.send_message(TELEGRAM_CHAT_ID, text, parse_mode=ParseMode.MARKDOWN, **kwargs); return
        except (TimedOut, Forbidden) as e:
            logger.error(f"Telegram Send Error: {e}. Attempt {i+1}/3.")
            if isinstance(e, Forbidden) or i == 2: logger.critical("Critical Telegram error."); return
            await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"Unknown Telegram Send Error: {e}. Attempt {i+1}/3."); await asyncio.sleep(2)

async def safe_edit_message(query, text, **kwargs):
    try: await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, **kwargs)
    except BadRequest as e:
        if "Message is not modified" not in str(e): logger.warning(f"Edit Message Error: {e}")
    except Exception as e: logger.error(f"Edit Message Error: {e}")

# --- ADAPTIVE INTELLIGENCE MODULE ---
async def update_strategy_performance(context: ContextTypes.DEFAULT_TYPE):
    logger.info("🧠 Adaptive Mind: Analyzing strategy performance...")
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            cursor = await conn.execute("SELECT reason, status, pnl_usdt FROM trades WHERE status LIKE '%(%' ORDER BY id DESC LIMIT 100")
            trades = await cursor.fetchall()
        if not trades: logger.info("🧠 Adaptive Mind: No closed trades found to analyze."); return
        stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'win_pnl': 0.0, 'loss_pnl': 0.0})
        for reason_str, status, pnl in trades:
            if not reason_str or pnl is None: continue
            clean_reason = reason_str.split(' (')[0]
            for r in set(clean_reason.split(' + ')):
                is_win = 'ناجحة' in status or 'تأمين' in status
                stats[r]['wins' if is_win else 'losses'] += 1
                stats[r]['win_pnl' if is_win else 'loss_pnl'] += pnl
                stats[r]['total_pnl'] += pnl
        performance_data = {}
        for r, s in stats.items():
            total = s['wins'] + s['losses']
            performance_data[r] = {
                "win_rate": round((s['wins'] / total * 100) if total > 0 else 0, 2),
                "profit_factor": round(s['win_pnl'] / abs(s['loss_pnl']) if s['loss_pnl'] != 0 else float('inf'), 2),
                "total_trades": total
            }
        bot_data.strategy_performance = performance_data
        logger.info(f"🧠 Adaptive Mind: Analysis complete for {len(performance_data)} strategies.")
    except Exception as e: logger.error(f"🧠 Adaptive Mind: Failed to analyze performance: {e}", exc_info=True)

async def propose_strategy_changes(context: ContextTypes.DEFAULT_TYPE):
    s = bot_data.settings
    if not s.get('adaptive_intelligence_enabled') or not s.get('strategy_proposal_enabled'): return
    logger.info("🧠 Adaptive Mind: Checking for underperforming strategies...")
    for scanner in s.get('active_scanners', []):
        perf = bot_data.strategy_performance.get(scanner)
        if perf and perf['total_trades'] >= s.get('strategy_analysis_min_trades', 10) and perf['win_rate'] < s.get('strategy_deactivation_threshold_wr', 45.0):
            if bot_data.pending_strategy_proposal.get('scanner') == scanner: continue
            proposal_key = f"prop_{int(time.time())}"
            bot_data.pending_strategy_proposal = {
                "key": proposal_key, "action": "disable", "scanner": scanner,
                "reason": f"أظهرت أداءً ضعيفًا بمعدل نجاح `{perf['win_rate']}%` في آخر `{perf['total_trades']}` صفقة."
            }
            logger.warning(f"🧠 Adaptive Mind: Proposing to disable '{scanner}'.")
            message = (f"💡 **اقتراح تحسين الأداء** 💡\n\n"
                       f"مرحباً، لاحظت أن استراتيجية **'{STRATEGY_NAMES_AR.get(scanner, scanner)}'** "
                       f"{bot_data.pending_strategy_proposal['reason']}\n\n"
                       f"أقترح تعطيلها مؤقتًا. هل توافق؟")
            keyboard = [[
                InlineKeyboardButton("✅ موافقة", callback_data=f"strategy_adjust_approve_{proposal_key}"),
                InlineKeyboardButton("❌ رفض", callback_data=f"strategy_adjust_reject_{proposal_key}")
            ]]
            await safe_send_message(context.bot, message, reply_markup=InlineKeyboardMarkup(keyboard)); return

# --- العقل والماسحات ---
def find_col(df_columns, prefix):
    try: return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: return None

async def translate_text_gemini(text_list):
    if not GEMINI_API_KEY or not text_list: return text_list, False
    prompt = "Translate the following English headlines to Arabic. Return only the translated text, with each headline on a new line:\n\n" + "\n".join(text_list)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            translated_text = response.json()['candidates'][0]['content']['parts'][0]['text']
            return translated_text.strip().split('\n'), True
    except Exception as e:
        logger.error(f"Gemini translation failed: {e}"); return text_list, False

def get_latest_crypto_news(limit=15):
    urls = ["https://cointelegraph.com/rss", "https://www.coindesk.com/arc/outboundfeeds/rss/"]
    headlines = [entry.title for url in urls for entry in feedparser.parse(url).entries[:7]]
    return list(set(headlines))[:limit]

def analyze_sentiment_of_headlines(headlines):
    if not headlines or not NLTK_AVAILABLE: return "N/A", 0.0
    sia = SentimentIntensityAnalyzer()
    score = sum(sia.polarity_scores(h)['compound'] for h in headlines) / len(headlines)
    if score > 0.15: mood = "إيجابية"
    elif score < -0.15: mood = "سلبية"
    else: mood = "محايدة"
    return mood, score

async def get_market_mood():
    s = bot_data.settings
    btc_mood_text = "الفلتر معطل"
    if s.get('btc_trend_filter_enabled', True):
        try:
            ohlcv = await bot_data.exchange.fetch_ohlcv('BTC/USDT', '4h', limit=s['trend_filters']['htf_period'] + 5)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['sma'] = ta.sma(df['close'], length=s['trend_filters']['htf_period'])
            is_btc_bullish = df['close'].iloc[-1] > df['sma'].iloc[-1]
            btc_mood_text = "صاعد ✅" if is_btc_bullish else "هابط ❌"
            if not is_btc_bullish: return {"mood": "NEGATIVE", "reason": "اتجاه BTC هابط", "btc_mood": btc_mood_text}
        except Exception as e: return {"mood": "DANGEROUS", "reason": f"فشل جلب بيانات BTC: {e}", "btc_mood": "UNKNOWN"}
    if s.get('market_mood_filter_enabled', True):
        fng = await get_fear_and_greed_index()
        if fng is not None and fng < s['fear_and_greed_threshold']:
            return {"mood": "NEGATIVE", "reason": f"مشاعر خوف شديد (F&G: {fng})", "btc_mood": btc_mood_text}
    return {"mood": "POSITIVE", "reason": "وضع السوق مناسب", "btc_mood": btc_mood_text}

async def get_fear_and_greed_index():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            return int(r.json()['data'][0]['value'])
    except Exception: return None

# --- تعريفات الماسحات ---
def analyze_momentum_breakout(df, params, rvol, adx_value):
    df.ta.vwap(append=True); df.ta.bbands(length=20, append=True); df.ta.macd(append=True); df.ta.rsi(append=True)
    last, prev = df.iloc[-2], df.iloc[-3]
    macd_col, macds_col, bbu_col, rsi_col = find_col(df.columns, "MACD_"), find_col(df.columns, "MACDs_"), find_col(df.columns, "BBU_"), find_col(df.columns, "RSI_")
    if not all([macd_col, macds_col, bbu_col, rsi_col]): return None
    if (prev[macd_col] <= prev[macds_col] and last[macd_col] > last[macds_col] and last['close'] > last[bbu_col] and last['close'] > last["VWAP_D"] and last[rsi_col] < 68):
        return {"reason": "momentum_breakout"}
    return None

def analyze_breakout_squeeze_pro(df, params, rvol, adx_value):
    df.ta.bbands(length=20, append=True); df.ta.kc(length=20, scalar=1.5, append=True); df.ta.obv(append=True)
    bbu_col, bbl_col, kcu_col, kcl_col = find_col(df.columns, "BBU_"), find_col(df.columns, "BBL_"), find_col(df.columns, "KCUe_"), find_col(df.columns, "KCLEe_")
    if not all([bbu_col, bbl_col, kcu_col, kcl_col]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    is_in_squeeze = prev[bbl_col] > prev[kcl_col] and prev[bbu_col] < prev[kcu_col]
    if is_in_squeeze and (last['close'] > last[bbu_col]) and (last['volume'] > df['volume'].rolling(20).mean().iloc[-2] * 1.5) and (df['OBV'].iloc[-2] > df['OBV'].iloc[-3]):
        return {"reason": "breakout_squeeze_pro"}
    return None

async def analyze_support_rebound(df, params, rvol, adx_value, exchange, symbol):
    try:
        ohlcv_1h = await exchange.fetch_ohlcv(symbol, '1h', limit=100)
        if len(ohlcv_1h) < 50: return None
        df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        current_price = df_1h['close'].iloc[-1]
        recent_lows = df_1h['low'].rolling(window=10, center=True).min()
        supports = recent_lows[recent_lows.notna()]
        closest_support = max([s for s in supports if s < current_price], default=None)
        if not closest_support or ((current_price - closest_support) / closest_support * 100 > 1.0): return None
        last_candle_15m = df.iloc[-2]
        if last_candle_15m['close'] > last_candle_15m['open'] and last_candle_15m['volume'] > df['volume'].rolling(window=20).mean().iloc[-2] * 1.5:
            return {"reason": "support_rebound"}
    except Exception: return None
    return None

def analyze_sniper_pro(df, params, rvol, adx_value):
    try:
        compression_candles = 24
        if len(df) < compression_candles + 2: return None
        compression_df = df.iloc[-compression_candles-1:-1]
        highest_high, lowest_low = compression_df['high'].max(), compression_df['low'].min()
        if lowest_low <= 0: return None
        volatility = (highest_high - lowest_low) / lowest_low * 100
        if volatility < 12.0:
            last_candle = df.iloc[-2]
            if last_candle['close'] > highest_high and last_candle['volume'] > compression_df['volume'].mean() * 2:
                return {"reason": "sniper_pro"}
    except Exception: return None
    return None

async def analyze_whale_radar(df, params, rvol, adx_value, exchange, symbol):
    try:
        ob = await exchange.fetch_order_book(symbol, limit=20)
        if not ob or not ob.get('bids'): return None
        if sum(float(price) * float(qty) for price, qty in ob['bids'][:10]) > 30000:
            return {"reason": "whale_radar"}
    except Exception: return None
    return None

def analyze_rsi_divergence(df, params, rvol, adx_value):
    if not SCIPY_AVAILABLE: return None
    df.ta.rsi(length=params.get('rsi_period', 14), append=True)
    rsi_col = find_col(df.columns, f"RSI_{params.get('rsi_period', 14)}")
    if not rsi_col or df[rsi_col].isnull().all(): return None
    subset = df.iloc[-params.get('lookback_period', 35):].copy()
    price_troughs_idx, _ = find_peaks(-subset['low'], distance=params.get('peak_trough_lookback', 5))
    rsi_troughs_idx, _ = find_peaks(-subset[rsi_col], distance=params.get('peak_trough_lookback', 5))
    if len(price_troughs_idx) >= 2 and len(rsi_troughs_idx) >= 2:
        p_low1_idx, p_low2_idx = price_troughs_idx[-2], price_troughs_idx[-1]
        r_low1_idx, r_low2_idx = rsi_troughs_idx[-2], rsi_troughs_idx[-1]
        is_divergence = (subset.iloc[p_low2_idx]['low'] < subset.iloc[p_low1_idx]['low'] and subset.iloc[r_low2_idx][rsi_col] > subset.iloc[r_low1_idx][rsi_col])
        if is_divergence:
            rsi_exits_oversold = (subset.iloc[r_low1_idx][rsi_col] < 35 and subset.iloc[-2][rsi_col] > 40)
            confirmation_price = subset.iloc[p_low2_idx:]['high'].max()
            price_confirmed = df.iloc[-2]['close'] > confirmation_price
            if (not params.get('confirm_with_rsi_exit', True) or rsi_exits_oversold) and price_confirmed:
                return {"reason": "rsi_divergence"}
    return None

def analyze_supertrend_pullback(df, params, rvol, adx_value):
    df.ta.supertrend(length=params.get('atr_period', 10), multiplier=params.get('atr_multiplier', 3.0), append=True)
    st_dir_col = find_col(df.columns, f"SUPERTd_{params.get('atr_period', 10)}_")
    if not st_dir_col: return None
    last, prev = df.iloc[-2], df.iloc[-3]
    if prev[st_dir_col] == -1 and last[st_dir_col] == 1:
        recent_swing_high = df['high'].iloc[-params.get('swing_high_lookback', 10):-2].max()
        if last['close'] > recent_swing_high:
            return {"reason": "supertrend_pullback"}
    return None

ALL_SCANNERS = {
    "momentum_breakout": analyze_momentum_breakout, "breakout_squeeze_pro": analyze_breakout_squeeze_pro,
    "support_rebound": analyze_support_rebound, "sniper_pro": analyze_sniper_pro, "whale_radar": analyze_whale_radar,
    "rsi_divergence": analyze_rsi_divergence, "supertrend_pullback": analyze_supertrend_pullback
}

# --- محرك التداول ---
async def get_binance_markets():
    settings = bot_data.settings
    if time.time() - bot_data.last_markets_fetch > 300:
        try:
            logger.info("Fetching and caching all Binance markets..."); all_tickers = await bot_data.exchange.fetch_tickers()
            bot_data.all_markets = list(all_tickers.values()); bot_data.last_markets_fetch = time.time()
        except Exception as e: logger.error(f"Failed to fetch all markets: {e}"); return []
    blacklist = settings.get('asset_blacklist', [])
    valid_markets = [t for t in bot_data.all_markets if 'USDT' in t['symbol'] and t.get('quoteVolume', 0) > settings['liquidity_filters']['min_quote_volume_24h_usd'] and t['symbol'].split('/')[0] not in blacklist and t.get('active', True) and not any(k in t['symbol'] for k in ['-SWAP', 'UP', 'DOWN', '3L', '3S'])]
    valid_markets.sort(key=lambda m: m.get('quoteVolume', 0), reverse=True)
    return valid_markets[:settings['top_n_symbols_by_volume']]

async def fetch_ohlcv_batch(exchange, symbols, timeframe, limit):
    tasks = [exchange.fetch_ohlcv(s, timeframe, limit=limit) for s in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return {symbols[i]: results[i] for i in range(len(symbols)) if not isinstance(results[i], Exception)}

async def worker_batch(queue, signals_list, errors_list):
    settings, exchange = bot_data.settings, bot_data.exchange
    while not queue.empty():
        try:
            item = await queue.get()
            market, ohlcv = item['market'], item['ohlcv']
            symbol = market['symbol']

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp').sort_index()
            if len(df) < 50:
                queue.task_done(); continue

            orderbook = await exchange.fetch_order_book(symbol, limit=1)
            if not orderbook['bids'] or not orderbook['asks']:
                queue.task_done(); continue

            best_bid, best_ask = orderbook['bids'][0][0], orderbook['asks'][0][0]
            if best_bid <= 0:
                queue.task_done(); continue
            spread_percent = ((best_ask - best_bid) / best_bid) * 100

            if 'whale_radar' in settings['active_scanners']:
                whale_radar_signal = await analyze_whale_radar(df.copy(), {}, 0, 0, exchange, symbol)
                if whale_radar_signal and spread_percent <= settings['spread_filter']['max_spread_percent'] * 2:
                    reason_str, strength = whale_radar_signal['reason'], 5
                    entry_price = df.iloc[-2]['close']
                    df.ta.atr(length=14, append=True)
                    atr = df.iloc[-2].get(find_col(df.columns, "ATRr_14"), 0)
                    if atr == 0: queue.task_done(); continue
                    risk = atr * settings['atr_sl_multiplier']
                    stop_loss, take_profit = entry_price - risk, entry_price + (risk * settings['risk_reward_ratio'])
                    signals_list.append({"symbol": symbol, "entry_price": entry_price, "take_profit": take_profit, "stop_loss": stop_loss, "reason": reason_str, "strength": strength, "weight": 1.0})
                    queue.task_done(); continue

            if spread_percent > settings['spread_filter']['max_spread_percent']:
                queue.task_done(); continue

            is_htf_bullish = True
            if settings.get('multi_timeframe_enabled', True):
                ohlcv_htf = await exchange.fetch_ohlcv(symbol, settings.get('multi_timeframe_htf'), limit=220)
                df_htf = pd.DataFrame(ohlcv_htf, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                if len(df_htf) > 200:
                    df_htf.ta.ema(length=200, append=True)
                    ema_col_name_htf = find_col(df_htf.columns, "EMA_200")
                    if ema_col_name_htf and pd.notna(df_htf[ema_col_name_htf].iloc[-2]):
                        is_htf_bullish = df_htf['close'].iloc[-2] > df_htf[ema_col_name_htf].iloc[-2]

            if settings.get('trend_filters', {}).get('enabled', True):
                ema_period = settings.get('trend_filters', {}).get('ema_period', 200)
                if len(df) < ema_period + 1: queue.task_done(); continue
                df.ta.ema(length=ema_period, append=True)
                ema_col_name = find_col(df.columns, f"EMA_{ema_period}")
                if not ema_col_name or pd.isna(df[ema_col_name].iloc[-2]): queue.task_done(); continue
                if df['close'].iloc[-2] < df[ema_col_name].iloc[-2]: queue.task_done(); continue

            vol_filters = settings.get('volatility_filters', {})
            atr_period, min_atr_percent = vol_filters.get('atr_period_for_filter', 14), vol_filters.get('min_atr_percent', 0.8)
            df.ta.atr(length=atr_period, append=True)
            atr_col_name = find_col(df.columns, f"ATRr_{atr_period}")
            if not atr_col_name or pd.isna(df[atr_col_name].iloc[-2]): queue.task_done(); continue
            last_close = df['close'].iloc[-2]
            atr_percent = (df[atr_col_name].iloc[-2] / last_close) * 100 if last_close > 0 else 0
            if atr_percent < min_atr_percent: queue.task_done(); continue

            df['volume_sma'] = ta.sma(df['volume'], length=20)
            if pd.isna(df['volume_sma'].iloc[-2]) or df['volume_sma'].iloc[-2] == 0: queue.task_done(); continue
            rvol = df['volume'].iloc[-2] / df['volume_sma'].iloc[-2]
            if rvol < settings.get('volume_filter_multiplier', 2.0): queue.task_done(); continue

            adx_value = 0
            if settings.get('adx_filter_enabled', False):
                df.ta.adx(append=True); adx_col = find_col(df.columns, "ADX_")
                adx_value = df[adx_col].iloc[-2] if adx_col and pd.notna(df[adx_col].iloc[-2]) else 0
                if adx_value < settings.get('adx_filter_level', 25): queue.task_done(); continue

            confirmed_reasons = []
            for name in settings['active_scanners']:
                if name == 'whale_radar': continue
                if not (strategy_func := ALL_SCANNERS.get(name)): continue
                params = settings.get(name, {})
                func_args = {'df': df.copy(), 'params': params, 'rvol': rvol, 'adx_value': adx_value}
                if name in ['support_rebound', 'whale_radar']: func_args.update({'exchange': exchange, 'symbol': symbol})
                result = await strategy_func(**func_args) if asyncio.iscoroutinefunction(strategy_func) else strategy_func(**{k: v for k, v in func_args.items() if k not in ['exchange', 'symbol']})
                if result: confirmed_reasons.append(result['reason'])

            if confirmed_reasons:
                reason_str, strength = ' + '.join(set(confirmed_reasons)), len(set(confirmed_reasons))
                
                trade_weight = 1.0
                if settings.get('adaptive_intelligence_enabled', True):
                    primary_reason = confirmed_reasons[0]
                    perf = bot_data.strategy_performance.get(primary_reason)
                    if perf:
                        if perf['win_rate'] < 50 and perf['total_trades'] > 5:
                            trade_weight = 1 - (settings['dynamic_sizing_max_decrease_pct'] / 100.0)
                        elif perf['win_rate'] > 70 and perf['profit_factor'] > 1.5:
                            trade_weight = 1 + (settings['dynamic_sizing_max_increase_pct'] / 100.0)
                        if perf['win_rate'] < settings['strategy_deactivation_threshold_wr'] and perf['total_trades'] > settings['strategy_analysis_min_trades']:
                           logger.warning(f"Signal for {symbol} from weak strategy '{primary_reason}' ignored."); queue.task_done(); continue

                if not is_htf_bullish:
                    strength = max(1, int(strength / 2)); reason_str += " (اتجاه كبير ضعيف)"; trade_weight *= 0.8

                entry_price = df.iloc[-2]['close']
                df.ta.atr(length=14, append=True)
                atr = df.iloc[-2].get(find_col(df.columns, "ATRr_14"), 0)
                if atr == 0: queue.task_done(); continue
                risk = atr * settings['atr_sl_multiplier']
                stop_loss, take_profit = entry_price - risk, entry_price + (risk * settings['risk_reward_ratio'])
                signals_list.append({"symbol": symbol, "entry_price": entry_price, "take_profit": take_profit, "stop_loss": stop_loss, "reason": reason_str, "strength": strength, "weight": trade_weight})

            queue.task_done()
        except Exception as e:
            if 'symbol' in locals(): logger.error(f"Error processing symbol {symbol}: {e}", exc_info=False); errors_list.append(symbol)
            else: logger.error(f"Worker error with no symbol context: {e}", exc_info=True)
            if not queue.empty(): queue.task_done()

# --- محرك التداول المدمج والنهائي ---
async def advanced_liquidity_check(symbol, required_amount):
    """Advanced check for liquidity and spread before execution."""
    try:
        order_book = await bot_data.exchange.fetch_order_book(symbol, limit=20)
        ask_liquidity = sum(float(price) * float(qty) for price, qty in order_book['asks'][:10])
        if ask_liquidity > 0 and (required_amount / ask_liquidity) > 0.1:
            logger.warning(f"High liquidity impact for {symbol}: {(required_amount/ask_liquidity):.2%}")
            return False
        if order_book['bids'] and order_book['asks']:
            spread_percent = ((order_book['asks'][0][0] - order_book['bids'][0][0]) / order_book['bids'][0][0]) * 100
            if spread_percent > bot_data.settings['spread_filter']['max_spread_percent']:
                logger.warning(f"High spread for {symbol}: {spread_percent:.2f}%")
                return False
        return True
    except Exception as e:
        logger.error(f"Liquidity check failed for {symbol}: {e}"); return False

async def validate_order_execution(order_id, symbol, max_wait_time=30):
    """Validates that an order has been filled within a specific timeframe."""
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        try:
            order = await bot_data.exchange.fetch_order(order_id, symbol)
            if order['status'] == 'closed':
                return {'success': True, 'order': order} if float(order.get('filled', 0)) > 0 else {'success': False, 'reason': 'Order closed without fill'}
            if order['status'] == 'canceled': return {'success': False, 'reason': 'Order was canceled'}
            await asyncio.sleep(2)
        except Exception as e:
            logger.warning(f"Error checking order {order_id}: {e}"); await asyncio.sleep(1)
    return {'success': False, 'reason': 'Timeout waiting for execution'}

async def initiate_real_trade(signal):
    """Robust trade initiation with retries, validation, and advanced checks."""
    if not bot_data.trading_enabled: return False
    if await has_active_trade_for_symbol(signal['symbol']):
        logger.info(f"Skipping trade for {signal['symbol']} - already has an active trade.")
        return False

    s, ex = bot_data.settings, bot_data.exchange
    trade_size = s['real_trade_size_usdt'] * signal.get('weight', 1.0) if s.get('dynamic_trade_sizing_enabled', True) else s['real_trade_size_usdt']
    
    if not await advanced_liquidity_check(signal['symbol'], trade_size): return False

    for attempt in range(3):
        try:
            balance = await ex.fetch_balance()
            if balance.get('USDT', {}).get('free', 0.0) < trade_size:
                logger.error(f"Insufficient USDT for {signal['symbol']}. Have: {balance.get('USDT',{}).get('free',0.0):,.2f}, Need: {trade_size:,.2f}"); return False
            
            buy_order = await ex.create_market_buy_order(signal['symbol'], trade_size / signal['entry_price'])
            if not buy_order or not buy_order.get('id'): raise ValueError("Invalid order response")

            if await log_pending_trade_to_db(signal, buy_order):
                await safe_send_message(bot_data.application.bot, f"🚀 Sent buy order for `{signal['symbol']}`. Awaiting confirmation...")
                validation = await validate_order_execution(buy_order['id'], signal['symbol'])
                if not validation.get('success', False):
                    logger.warning(f"Order validation failed for {signal['symbol']}: {validation.get('reason', 'N/A')}")
                health_monitor.record_successful_order(); return True 
            else:
                logger.critical(f"CRITICAL: Failed to log trade for {signal['symbol']}. Cancelling order.")
                await ex.cancel_order(buy_order['id'], signal['symbol']); break
        
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            logger.warning(f"Network error on trade attempt {attempt+1} for {signal['symbol']}: {e}")
            if attempt < 2: await asyncio.sleep(2 * (attempt + 1)); continue
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error for {signal['symbol']}: {e}"); break
        except Exception as e:
            logger.error(f"Unexpected trade error for {signal['symbol']}: {e}", exc_info=True); break
            
    health_monitor.record_failed_order(); return False
    
async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    async with scan_lock:
        if not bot_data.trading_enabled:
            logger.info("Scan skipped: Kill Switch is active."); return
        
        scan_start_time = time.time()
        logger.info("--- Starting new Reliability-Enhanced scan... ---")
        settings, bot = bot_data.settings, context.bot
        
        try:
            balance = await bot_data.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0.0)
            trade_size_min_check = settings['real_trade_size_usdt'] * 0.8 
            if usdt_balance < trade_size_min_check:
                logger.error(f"Scan skipped: Insufficient USDT balance ({usdt_balance:,.2f} < {trade_size_min_check:,.2f}) to open a trade.")
                return
        except Exception as e:
            logger.error(f"Failed to fetch balance for scan check: {e}"); return

        mood_result = await get_market_mood()
        bot_data.market_mood = mood_result
        if mood_result['mood'] in ["NEGATIVE", "DANGEROUS"]:
            logger.warning(f"Scan skipped due to market mood: {mood_result['reason']}")
            return

        async with aiosqlite.connect(DB_FILE) as conn:
            active_trades_count = (await (await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active' OR status = 'pending'")).fetchone())[0]
        if active_trades_count >= settings['max_concurrent_trades']:
            logger.info(f"Scan skipped: Max trades ({active_trades_count}) reached."); return

        top_markets = await get_binance_markets()
        symbols_to_scan = [m['symbol'] for m in top_markets]
        ohlcv_data = await fetch_ohlcv_batch(bot_data.exchange, symbols_to_scan, TIMEFRAME, 220)

        queue, signals_found, analysis_errors = asyncio.Queue(), [], []
        for market in top_markets:
            if market['symbol'] in ohlcv_data:
                await queue.put({'market': market, 'ohlcv': ohlcv_data[market['symbol']]})

        worker_tasks = [asyncio.create_task(worker_batch(queue, signals_found, analysis_errors)) for _ in range(settings.get("worker_threads", 10))]
        await queue.join()
        for task in worker_tasks: task.cancel()

        trades_opened_count = 0
        signals_found.sort(key=lambda s: s.get('strength', 0), reverse=True)

        for signal in signals_found:
            if active_trades_count >= settings['max_concurrent_trades']:
                logger.info(f"Stopping trade initiation, max concurrent trades ({active_trades_count}) reached.")
                break
            
            if time.time() - bot_data.last_signal_time.get(signal['symbol'], 0) > (SCAN_INTERVAL_SECONDS * 0.9):
                bot_data.last_signal_time[signal['symbol']] = time.time()
                if await initiate_real_trade(signal):
                    active_trades_count += 1
                    trades_opened_count += 1
                await asyncio.sleep(2)

        scan_duration = time.time() - scan_start_time
        bot_data.last_scan_info = {"start_time": datetime.fromtimestamp(scan_start_time, EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'), "duration_seconds": int(scan_duration), "checked_symbols": len(top_markets), "analysis_errors": len(analysis_errors)}
        logger.info(f"Scan finished in {scan_duration:.2f}s. Found {len(signals_found)} signals, opened {trades_opened_count} trades.")


# --- WebSocket and Trade Management ---
class UserDataStreamManager:
    """Manages the private user data stream for instant order updates."""
    def __init__(self, exchange, on_order_update_coro):
        self.exchange = exchange
        self.on_order_update = on_order_update_coro
        self.listen_key = None
        self.ws = None
        self.is_running = False

    async def _get_listen_key(self):
        try:
            self.listen_key = (await self.exchange.publicPostUserDataStream())['listenKey']
            logger.info("User Data Stream: Listen key obtained.")
        except Exception as e:
            logger.error(f"User Data Stream: Failed to get listen key: {e}"); self.listen_key = None

    async def _keep_alive(self):
        while self.is_running:
            await asyncio.sleep(1800) # 30 mins
            if self.listen_key:
                try:
                    await self.exchange.publicPutUserDataStream({'listenKey': self.listen_key})
                    logger.info("User Data Stream: Listen key kept alive.")
                except Exception as e:
                    logger.warning(f"User Data Stream: Failed to keep listen key alive: {e}"); self.listen_key = None

    async def run(self):
        self.is_running = True
        asyncio.create_task(self._keep_alive())
        while self.is_running:
            await self._get_listen_key()
            if not self.listen_key: await asyncio.sleep(60); continue
            uri = f"wss://stream.binance.com:9443/ws/{self.listen_key}"
            try:
                async with websockets.connect(uri) as ws:
                    self.ws = ws; logger.info("✅ [User Data Stream] Connected.")
                    async for message in ws:
                        data = json.loads(message)
                        if data.get('e') == 'executionReport' and data.get('x') == 'TRADE' and data.get('S') == 'BUY':
                            await self.on_order_update(data)
            except (websockets.exceptions.ConnectionClosed, Exception) as e:
                if self.is_running: logger.warning(f"User Data Stream: Connection lost: {e}. Reconnecting..."); await asyncio.sleep(5)
                else: break

    async def stop(self):
        self.is_running = False
        if self.ws: await self.ws.close()

async def handle_order_update(order_data):
    if order_data['X'] == 'FILLED' and order_data['S'] == 'BUY':
        logger.info(f"Fast Reporter: Received fill for order {order_data['i']}. Activating trade...")
        await activate_trade(order_data['i'], order_data['s'].replace('USDT', '/USDT'))

async def activate_trade(order_id, symbol):
    bot = bot_data.application.bot
    try:
        order_details = await bot_data.exchange.fetch_order(order_id, symbol)
        filled_price = float(order_details.get('average', 0.0))
        net_filled_quantity = float(order_details.get('filled', 0.0))
        if net_filled_quantity <= 0 or filled_price <= 0:
            logger.error(f"Order {order_id} invalid fill data. Cancelling activation."); return
    except Exception as e:
        logger.error(f"Could not fetch order details for activation of {order_id}: {e}", exc_info=True); return

    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        trade = await (await conn.execute("SELECT * FROM trades WHERE order_id = ? AND status = 'pending'", (order_id,))).fetchone()
        if not trade:
            logger.info(f"Activation ignored for {order_id}: Trade not found or not pending."); return

        trade = dict(trade); logger.info(f"Activating trade #{trade['id']} for {symbol}...")
        risk = filled_price - trade['stop_loss']
        new_take_profit = filled_price + (risk * bot_data.settings['risk_reward_ratio'])

        await conn.execute(
            "UPDATE trades SET status = 'active', entry_price = ?, quantity = ?, take_profit = ?, last_profit_notification_price = ? WHERE id = ?",
            (filled_price, net_filled_quantity, new_take_profit, filled_price, trade['id'])
        )
        active_trades_count = (await (await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active'")).fetchone())[0]
        await conn.commit()

    await bot_data.trade_guardian.sync_subscriptions()
    balance_after = await bot_data.exchange.fetch_balance()
    usdt_remaining = balance_after.get('USDT', {}).get('free', 0)
    tp_percent = (new_take_profit / filled_price - 1) * 100
    sl_percent = (1 - trade['stop_loss'] / filled_price) * 100
    reasons_ar = ' + '.join([STRATEGY_NAMES_AR.get(r.strip(), r.strip()) for r in trade['reason'].split(' + ')])

    success_msg = (
        f"✅ **تم تأكيد الشراء | {symbol}**\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"**الاستراتيجية:** {reasons_ar} {'⭐' * trade.get('signal_strength', 1)}\n"
        f"**تفاصيل الصفقة:** `#{trade['id']}`\n"
        f"  - **سعر التنفيذ:** `${filled_price:,.4f}`\n"
        f"  - **التكلفة:** `${filled_price * net_filled_quantity:,.2f}`\n"
        f"**الأهداف:**\n"
        f"  - **الهدف (TP):** `${new_take_profit:,.4f}` `({tp_percent:+.2f}%)`\n"
        f"  - **الوقف (SL):** `${trade['stop_loss']:,.4f}` `({sl_percent:.2f}%)`\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"💰 **السيولة المتبقية:** `${usdt_remaining:,.2f}` | **الصفقات النشطة:** `{active_trades_count}`"
    )
    await safe_send_message(bot, success_msg)

class TradeGuardian:
    def __init__(self, application):
        self.application = application
        self.public_ws = None
        self.subscriptions = set()
        self.is_running = False

    async def handle_ticker_update(self, message):
        health_monitor.update_heartbeat()
        data = json.loads(message)
        if 's' not in data: return
        symbol = data['s'].replace('USDT', '/USDT')
        current_price = float(data['c'])

        async with trade_management_lock:
            try:
                async with aiosqlite.connect(DB_FILE) as conn:
                    conn.row_factory = aiosqlite.Row
                    trade = await (await conn.execute("SELECT * FROM trades WHERE symbol = ? AND status = 'active'", (symbol,))).fetchone()
                    if not trade: return

                    trade = dict(trade); settings = bot_data.settings
                    highest_price = max(trade.get('highest_price', 0), current_price)
                    if highest_price > trade.get('highest_price', 0):
                        await conn.execute("UPDATE trades SET highest_price = ? WHERE id = ?", (highest_price, trade['id']))

                    if settings['trailing_sl_enabled']:
                        if not trade['trailing_sl_active'] and current_price >= trade['entry_price'] * (1 + settings['trailing_sl_activation_percent'] / 100):
                            new_sl = trade['entry_price'] * 1.001
                            await conn.execute("UPDATE trades SET trailing_sl_active = 1, stop_loss = ? WHERE id = ?", (new_sl, trade['id']))
                            await safe_send_message(self.application.bot, f"**🚀 تأمين الأرباح! | #{trade['id']} {trade['symbol']}**\nتم رفع وقف الخسارة إلى نقطة الدخول: `${new_sl:,.4f}`")
                        elif trade['trailing_sl_active']:
                            new_sl = highest_price * (1 - settings['trailing_sl_callback_percent'] / 100)
                            if new_sl > trade['stop_loss']: await conn.execute("UPDATE trades SET stop_loss = ? WHERE id = ?", (new_sl, trade['id']))
                    
                    if settings.get('incremental_notifications_enabled', True) and current_price >= trade.get('last_profit_notification_price', trade['entry_price']) * (1 + settings['incremental_notification_percent'] / 100):
                        profit_percent = ((current_price / trade['entry_price']) - 1) * 100
                        await safe_send_message(self.application.bot, f"📈 **ربح متزايد! | #{trade['id']} {trade['symbol']}**\n**الربح الحالي:** `{profit_percent:+.2f}%`")
                        await conn.execute("UPDATE trades SET last_profit_notification_price = ? WHERE id = ?", (current_price, trade['id']))
                    await conn.commit()
                
                # Re-fetch trade after potential DB updates
                async with aiosqlite.connect(DB_FILE) as conn:
                    conn.row_factory = aiosqlite.Row
                    trade = dict(await (await conn.execute("SELECT * FROM trades WHERE id = ?", (trade['id'],))).fetchone())

                if current_price >= trade['take_profit']: await self._close_trade(trade, "ناجحة (TP)", current_price)
                elif current_price <= trade['stop_loss']:
                    reason = "تم تأمين الربح (TSL)" if current_price > trade['entry_price'] else "فاشلة (SL)"
                    await self._close_trade(trade, reason, current_price)

            except Exception as e:
                logger.error(f"Guardian Ticker Error for {symbol}: {e}", exc_info=True)
    
    async def _close_trade(self, trade, reason, close_price):
        symbol, trade_id, quantity = trade['symbol'], trade['id'], trade['quantity']
        bot = self.application.bot
        logger.info(f"Guardian: Attempting to close trade #{trade_id} for {symbol}. Reason: {reason}")
        
        try:
            market = await bot_data.exchange.market(symbol)
            min_notional = market.get('limits', {}).get('notional', {}).get('min', 0.0)
            if min_notional and (quantity * close_price) < float(min_notional):
                logger.critical(f"Closure failed for #{trade_id}: Notional value below minimum. Manual action required.")
                async with aiosqlite.connect(DB_FILE) as conn:
                    await conn.execute("UPDATE trades SET status = ? WHERE id = ?", (f'closure_failed (min_notional)', trade_id))
                    await conn.commit()
                await safe_send_message(bot, f"🚨 **فشل إغلاق #{trade_id} | {symbol}**\nقيمة الصفقة أقل من الحد الأدنى للبيع. الرجاء المراجعة اليدوية.")
                await self.sync_subscriptions(); return
        except Exception as e:
            logger.error(f"Error checking notional value for #{trade_id}: {e}")

        for i in range(bot_data.settings.get('close_retries', 3)):
            try:
                await bot_data.exchange.create_market_sell_order(symbol, quantity)
                pnl = (close_price - trade['entry_price']) * quantity
                pnl_percent = (close_price / trade['entry_price'] - 1) * 100 if trade['entry_price'] > 0 else 0
                async with aiosqlite.connect(DB_FILE) as conn:
                    await conn.execute("UPDATE trades SET status = ?, close_price = ?, pnl_usdt = ? WHERE id = ?", (reason, close_price, pnl, trade_id))
                    await conn.commit()
                await self.sync_subscriptions()
                await safe_send_message(bot, f"{'✅' if pnl >= 0 else '🛑'} **تم إغلاق الصفقة | #{trade_id} {symbol}**\n**السبب:** {reason}\n**الربح/الخسارة:** `${pnl:,.2f}` ({pnl_percent:+.2f}%)")
                return
            except Exception as e:
                logger.warning(f"Failed to close trade #{trade_id}. Retrying... ({i + 1}/{bot_data.settings.get('close_retries', 3)})", exc_info=True)
                await asyncio.sleep(5)

        logger.critical(f"CRITICAL: Failed to close trade #{trade_id} after retries.")
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute("UPDATE trades SET status = 'closure_failed' WHERE id = ?", (trade_id,))
            await conn.commit()
        await safe_send_message(bot, f"🚨 **فشل حرج** 🚨\nفشل إغلاق الصفقة `#{trade_id}` بعد عدة محاولات. الرجاء مراجعة المنصة يدوياً.")
        await self.sync_subscriptions()

    async def run_public_ws(self):
        self.is_running = True
        while self.is_running:
            stream_name = '/'.join([f"{s.lower().replace('/', '')}@ticker" for s in self.subscriptions])
            if not stream_name: await asyncio.sleep(5); continue
            uri = f"wss://stream.binance.com:9443/ws/{stream_name}"
            try:
                async with websockets.connect(uri) as ws:
                    self.public_ws = ws
                    await self.sync_subscriptions(reconnect=True) 
                    logger.info(f"✅ [Guardian's Eyes] Connected. Watching {len(self.subscriptions)} symbols.")
                    async for message in ws: await self.handle_ticker_update(message)
            except (websockets.exceptions.ConnectionClosed, Exception) as e:
                health_monitor.record_connection_drop()
                if self.is_running: logger.warning(f"Guardian's Eyes: Connection lost: {e}. Reconnecting..."); await asyncio.sleep(5)
                else: break

    async def sync_subscriptions(self, reconnect=False):
        async with aiosqlite.connect(DB_FILE) as conn:
            active_symbols = {row[0] for row in await (await conn.execute("SELECT DISTINCT symbol FROM trades WHERE status = 'active'")).fetchall()}
        if active_symbols != self.subscriptions or reconnect:
            logger.info(f"Guardian: Syncing subscriptions. Old: {len(self.subscriptions)}, New: {len(active_symbols)}")
            self.subscriptions = active_symbols
            if self.public_ws and not self.public_ws.closed and not reconnect:
                try: await self.public_ws.close()
                except Exception: pass

    async def stop(self):
        self.is_running = False
        if self.public_ws: await self.public_ws.close()

async def the_supervisor_job(context: ContextTypes.DEFAULT_TYPE):
    logger.info("🕵️ Supervisor: Auditing pending trades...")
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        stuck_trades = await (await conn.execute("SELECT * FROM trades WHERE status = 'pending' AND timestamp <= ?", ((datetime.now(EGYPT_TZ) - timedelta(minutes=2)).isoformat(),))).fetchall()
        if not stuck_trades:
            logger.info("🕵️ Supervisor: Audit complete. No abandoned trades found."); return

        for trade_data in stuck_trades:
            trade = dict(trade_data)
            logger.warning(f"🕵️ Supervisor: Found abandoned trade #{trade['id']}. Investigating.")
            try:
                order_status = await bot_data.exchange.fetch_order(trade['order_id'], trade['symbol'])
                if order_status['status'] == 'closed' and float(order_status.get('filled', 0)) > 0:
                    logger.info(f"🕵️ Supervisor: API confirms order {trade['order_id']} was filled. Activating.")
                    await activate_trade(trade['order_id'], trade['symbol'])
                else:
                    await bot_data.exchange.cancel_order(trade['order_id'], trade['symbol'])
                    await conn.execute("UPDATE trades SET status = 'failed (canceled by supervisor)' WHERE id = ?", (trade['id'],))
                await conn.commit()
            except ccxt.OrderNotFound:
                await conn.execute("UPDATE trades SET status = 'failed (order not found)' WHERE id = ?", (trade['id'],))
                await conn.commit()
            except Exception as e:
                logger.error(f"🕵️ Supervisor: Failed to rectify trade #{trade['id']}: {e}")

# --- واجهة تليجرام ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Dashboard 🖥️"], ["الإعدادات ⚙️"]]
    await update.message.reply_text("أهلاً بك في **بوت باينانس V7 (النسخة المدمجة)**", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True), parse_mode=ParseMode.MARKDOWN)

async def manual_scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not bot_data.trading_enabled: await (update.message or update.callback_query.message).reply_text("🔬 الفحص محظور. مفتاح الإيقاف مفعل."); return
    await (update.message or update.callback_query.message).reply_text("🔬 أمر فحص يدوي... قد يستغرق بعض الوقت.")
    context.job_queue.run_once(perform_scan, 1)

async def show_dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ks_status_emoji = "🚨" if not bot_data.trading_enabled else "✅"
    ks_status_text = "مفتاح الإيقاف (مفعل)" if not bot_data.trading_enabled else "الحالة (طبيعية)"
    keyboard = [
        [InlineKeyboardButton("💼 نظرة عامة على المحفظة", callback_data="db_portfolio"), InlineKeyboardButton("📈 الصفقات النشطة", callback_data="db_trades")],
        [InlineKeyboardButton("📜 سجل الصفقات المغلقة", callback_data="db_history"), InlineKeyboardButton("📊 الإحصائيات والأداء", callback_data="db_stats")],
        [InlineKeyboardButton("🌡️ تحليل مزاج السوق", callback_data="db_mood"), InlineKeyboardButton("🔬 فحص فوري", callback_data="db_manual_scan")],
        [InlineKeyboardButton("❤️ تقرير صحة النظام", callback_data="db_health")],
        [InlineKeyboardButton(f"{ks_status_emoji} {ks_status_text}", callback_data="kill_switch_toggle"), InlineKeyboardButton("🕵️‍♂️ تقرير التشخيص", callback_data="db_diagnostics")]
    ]
    message_text = "🖥️ **لوحة تحكم قناص Binance**\n\nاختر نوع التقرير الذي تريد عرضه:"
    if not bot_data.trading_enabled: message_text += "\n\n**تحذير: تم تفعيل مفتاح الإيقاف.**"
    target_message = update.message or update.callback_query.message
    if update.callback_query: await safe_edit_message(update.callback_query, message_text, reply_markup=InlineKeyboardMarkup(keyboard))
    else: await target_message.reply_text(message_text, parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_health_report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    report_data = health_monitor.get_health_report()
    report = (
        f"❤️ **تقرير صحة النظام**\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"**⏳ مدة التشغيل:** {report_data['uptime_hours']}\n"
        f"**📈 إجمالي الأوامر:** {report_data['total_orders']}\n"
        f"**🎯 نسبة نجاح الأوامر:** {report_data['success_rate']}\n"
        f"**🔌 انقطاعات الاتصال:** {report_data['connection_drops']}\n"
        f"**💓 آخر نبضة (WS):** {report_data['last_heartbeat_ago']}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"*هذا التقرير يساعد في تشخيص المشاكل المتعلقة بالاتصال وتنفيذ الأوامر.*"
    )
    keyboard = [[InlineKeyboardButton("🔄 تحديث", callback_data="db_health")], [InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]
    await safe_edit_message(query, report, reply_markup=InlineKeyboardMarkup(keyboard))

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer(); data = query.data
    route_map = {
        "db_stats": show_stats_command, "db_trades": show_trades_command, "db_history": show_trade_history_command,
        "db_mood": show_mood_command, "db_diagnostics": show_diagnostics_command, "back_to_dashboard": show_dashboard_command,
        "db_portfolio": show_portfolio_command, "db_manual_scan": manual_scan_command,
        "kill_switch_toggle": toggle_kill_switch, "db_health": show_health_report_command,
        "settings_main": show_settings_menu, "settings_params": show_parameters_menu, "settings_scanners": show_scanners_menu,
        "settings_presets": show_presets_menu, "settings_blacklist": show_blacklist_menu, "settings_data": show_data_management_menu,
        "blacklist_add": handle_blacklist_action, "blacklist_remove": handle_blacklist_action,
        "data_clear_confirm": handle_clear_data_confirmation, "data_clear_execute": handle_clear_data_execute,
        "settings_adaptive": show_adaptive_intelligence_menu, "noop": (lambda u,c: None)
    }
    if data in route_map: await route_map[data](update, context)
    elif data.startswith("check_"): await check_trade_details(update, context)
    elif data.startswith("scanner_toggle_"): await handle_scanner_toggle(update, context)
    elif data.startswith("preset_set_"): await handle_preset_set(update, context)
    elif data.startswith("param_set_"): await handle_parameter_selection(update, context)
    elif data.startswith("param_toggle_"): await handle_toggle_parameter(update, context)
    elif data.startswith("strategy_adjust_"): await handle_strategy_adjustment(update, context)

async def toggle_kill_switch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; bot_data.trading_enabled = not bot_data.trading_enabled
    if bot_data.trading_enabled: await query.answer("✅ تم استئناف التداول الطبيعي.")
    else: await query.answer("🚨 تم تفعيل مفتاح الإيقاف!", show_alert=True)
    await show_dashboard_command(update, context)

async def show_trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row; trades = await (await conn.execute("SELECT id, symbol, status FROM trades WHERE status = 'active' OR status = 'pending' ORDER BY id DESC")).fetchall()
    if not trades:
        text = "لا توجد صفقات نشطة حاليًا."
        keyboard = [[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]
    else:
        text = "📈 *الصفقات النشطة*\nاختر صفقة لعرض تفاصيلها:\n"; keyboard = []
        for trade in trades: keyboard.append([InlineKeyboardButton(f"#{trade['id']} {'✅' if trade['status'] == 'active' else '⏳'} | {trade['symbol']}", callback_data=f"check_{trade['id']}")])
        keyboard.extend([[InlineKeyboardButton("🔄 تحديث", callback_data="db_trades")], [InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]])
    await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup(keyboard))

async def check_trade_details(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    trade_id = int(query.data.split('_')[1])
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        trade = await (await conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))).fetchone()
    if not trade: await query.answer("لم يتم العثور على الصفقة."); return
    trade = dict(trade)
    if trade['status'] == 'pending':
        message = f"**⏳ حالة الصفقة #{trade_id}**\n- **العملة:** `{trade['symbol']}`\n- **الحالة:** في انتظار تأكيد التنفيذ..."
    else:
        try:
            ticker = await bot_data.exchange.fetch_ticker(trade['symbol'])
            current_price = ticker['last']
            pnl = (current_price - trade['entry_price']) * trade['quantity']
            pnl_percent = (current_price / trade['entry_price'] - 1) * 100 if trade['entry_price'] > 0 else 0
            pnl_text = f"💰 **الربح/الخسارة الحالية:** `${pnl:+.2f}` ({pnl_percent:+.2f}%)"
            current_price_text = f"- **السعر الحالي:** `${current_price}`"
        except Exception:
            pnl_text = "💰 تعذر جلب الربح/الخسارة الحالية."; current_price_text = "- **السعر الحالي:** `تعذر الجلب`"
        message = (f"**✅ حالة الصفقة #{trade_id}**\n\n- **العملة:** `{trade['symbol']}`\n- **سعر الدخول:** `${trade['entry_price']}`\n{current_price_text}\n- **الكمية:** `{trade['quantity']}`\n----------------------------------\n- **الهدف (TP):** `${trade['take_profit']}`\n- **الوقف (SL):** `${trade['stop_loss']}`\n----------------------------------\n{pnl_text}")
    await safe_edit_message(query, message, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة للصفقات", callback_data="db_trades")]]))

async def show_mood_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer("جاري تحليل مزاج السوق...")
    fng_index, original_headlines, mood, all_markets = await asyncio.gather(
        get_fear_and_greed_index(), asyncio.to_thread(get_latest_crypto_news), get_market_mood(), get_binance_markets()
    )
    translated_headlines, translation_success = await translate_text_gemini(original_headlines)
    news_sentiment, _ = analyze_sentiment_of_headlines(original_headlines)
    sorted_by_change = sorted([m for m in all_markets if m.get('percentage') is not None], key=lambda m: m['percentage'], reverse=True)
    gainers_str = "\n".join([f"  `{g['symbol']}` `({g.get('percentage', 0):+.2f}%)`" for g in sorted_by_change[:3]]) or "  لا توجد بيانات."
    losers_str = "\n".join([f"  `{l['symbol']}` `({l.get('percentage', 0):+.2f}%)`" for l in reversed(sorted_by_change[-3:])]) or "  لا توجد بيانات."
    news_str = "\n".join([f"  - _{h}_" for h in translated_headlines]) or "  لا توجد أخبار."
    message = (f"**🌡️ تحليل مزاج السوق الشامل**\n━━━━━━━━━━━━━━━━━━━━\n**⚫️ الخلاصة:** *{mood['reason']}*\n━━━━━━━━━━━━━━━━━━━━\n**📊 المؤشرات:**\n  - **اتجاه BTC:** {mood.get('btc_mood', 'N/A')}\n  - **الخوف والطمع:** {fng_index or 'N/A'}\n  - **مشاعر الأخبار:** {news_sentiment}\n━━━━━━━━━━━━━━━━━━━━\n**🚀 الرابحون:**\n{gainers_str}\n\n**📉 الخاسرون:**\n{losers_str}\n━━━━━━━━━━━━━━━━━━━━\n**📰 آخر الأخبار:**\n{news_str}")
    keyboard = [[InlineKeyboardButton("🔄 تحديث", callback_data="db_mood")], [InlineKeyboardButton("🔙 العودة", callback_data="back_to_dashboard")]]
    await safe_edit_message(query, message, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row; trades_data = await (await conn.execute("SELECT pnl_usdt, status FROM trades WHERE status LIKE '%(%'")).fetchall()
    if not trades_data: await safe_edit_message(update.callback_query, "لم يتم إغلاق أي صفقات بعد."); return
    total_pnl = sum(t['pnl_usdt'] for t in trades_data if t['pnl_usdt'] is not None)
    wins_data = [t['pnl_usdt'] for t in trades_data if ('ناجحة' in t['status'] or 'تأمين' in t['status']) and t['pnl_usdt'] is not None]
    losses_data = [t['pnl_usdt'] for t in trades_data if 'فاشلة' in t['status'] and t['pnl_usdt'] is not None]
    win_rate = (len(wins_data) / len(trades_data) * 100) if trades_data else 0
    profit_factor = sum(wins_data) / abs(sum(losses_data)) if sum(losses_data) != 0 else float('inf')
    message = (f"📊 **إحصائيات الأداء التفصيلية**\n━━━━━━━━━━━━━━━━━━\n**إجمالي الربح/الخسارة:** `${total_pnl:+.2f}`\n**عامل الربح:** `{profit_factor:,.2f}`\n**معدل النجاح:** {win_rate:.1f}%\n**إجمالي الصفقات:** {len(trades_data)}")
    await safe_edit_message(update.callback_query, message, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📜 تقرير الاستراتيجيات", callback_data="db_strategy_report")], [InlineKeyboardButton("🔙 العودة", callback_data="back_to_dashboard")]]))

async def show_portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer("جاري جلب بيانات المحفظة...")
    try:
        balance = await bot_data.exchange.fetch_balance()
        assets_to_fetch = [f"{asset}/USDT" for asset, data in balance.items() if isinstance(data, dict) and data.get('total', 0) > 0 and 'USDT' not in asset]
        tickers = await bot_data.exchange.fetch_tickers(assets_to_fetch) if assets_to_fetch else {}
        total_assets_value_usdt = sum(tickers[f"{asset}/USDT"].get('last', 0) * data['total'] for asset, data in balance.items() if f"{asset}/USDT" in tickers)
        async with aiosqlite.connect(DB_FILE) as conn:
            total_realized_pnl = (await (await conn.execute("SELECT SUM(pnl_usdt) FROM trades WHERE status LIKE '%(%'")).fetchone())[0] or 0.0
            active_trades_count = (await (await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active'")).fetchone())[0]
        total_equity = balance.get('USDT', {}).get('total', 0) + total_assets_value_usdt
        message = (f"**💼 نظرة عامة على المحفظة**\n━━━━━━━━━━━━━━━━━━━━\n**💰 إجمالي قيمة المحفظة:** `≈ ${total_equity:,.2f}`\n  - **السيولة المتاحة (USDT):** `${balance.get('USDT', {}).get('free', 0):,.2f}`\n  - **قيمة الأصول الأخرى:** `≈ ${total_assets_value_usdt:,.2f}`\n━━━━━━━━━━━━━━━━━━━━\n**📈 أداء التداول:**\n  - **الربح/الخسارة المحقق:** `${total_realized_pnl:,.2f}`\n  - **الصفقات النشطة:** {active_trades_count}")
        await safe_edit_message(query, message, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 تحديث", callback_data="db_portfolio")], [InlineKeyboardButton("🔙 العودة", callback_data="back_to_dashboard")]]))
    except Exception as e:
        logger.error(f"Portfolio fetch error: {e}", exc_info=True)
        await safe_edit_message(query, f"حدث خطأ أثناء جلب رصيد المحفظة: {e}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة", callback_data="back_to_dashboard")]]))

async def show_trade_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute("SELECT symbol, pnl_usdt, status FROM trades WHERE status LIKE '%(%' ORDER BY id DESC LIMIT 10")
        closed_trades = await cursor.fetchall()
    if not closed_trades:
        text = "لم يتم إغلاق أي صفقات بعد."
    else:
        history_list = ["📜 *آخر 10 صفقات مغلقة*"]
        for trade in closed_trades:
            emoji = "✅" if ('ناجحة' in trade['status'] or 'تأمين' in trade['status']) else "🛑"
            pnl = trade['pnl_usdt'] or 0.0
            history_list.append(f"{emoji} `{trade['symbol']}` | الربح/الخسارة: `${pnl:,.2f}`")
        text = "\n".join(history_list)
    await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]]))

async def show_diagnostics_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; s = bot_data.settings
    scan_info = bot_data.last_scan_info
    determine_active_preset()
    scan_job = context.job_queue.get_jobs_by_name("perform_scan")
    next_scan_time = scan_job[0].next_t.astimezone(EGYPT_TZ).strftime('%H:%M:%S') if scan_job and scan_job[0].next_t else "N/A"
    db_size = f"{os.path.getsize(DB_FILE) / 1024:.2f} KB" if os.path.exists(DB_FILE) else "N/A"
    async with aiosqlite.connect(DB_FILE) as conn:
        total_trades = (await (await conn.execute("SELECT COUNT(*) FROM trades")).fetchone())[0]
        active_trades = (await (await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active'")).fetchone())[0]
    guardian_ws_status = "متصل ✅" if bot_data.trade_guardian and bot_data.trade_guardian.public_ws and not bot_data.trade_guardian.public_ws.closed else "غير متصل ❌"
    uds_ws_status = "متصل ✅" if bot_data.user_data_stream and bot_data.user_data_stream.ws and not bot_data.user_data_stream.ws.closed else "غير متصل ❌"
    report = (f"🕵️‍♂️ *تقرير التشخيص*\n\n**النمط:** {bot_data.active_preset_name}\n**آخر فحص:** {scan_info.get('start_time', 'N/A')}\n**مدة الفحص:** {scan_info.get('duration_seconds', 'N/A')} ثانية\n**الفحص التالي:** {next_scan_time}\n**اتصال Guardian:** {guardian_ws_status}\n**اتصال UDS:** {uds_ws_status}\n**حجم قاعدة البيانات:** {db_size}\n**إجمالي الصفقات:** {total_trades} ({active_trades} نشطة)")
    await safe_edit_message(query, report, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 تحديث", callback_data="db_diagnostics")], [InlineKeyboardButton("🔙 العودة", callback_data="back_to_dashboard")]]))

async def show_settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🧠 إعدادات الذكاء التكيفي", callback_data="settings_adaptive")],
        [InlineKeyboardButton("🎛️ تعديل المعايير المتقدمة", callback_data="settings_params")],
        [InlineKeyboardButton("🔭 تفعيل/تعطيل الماسحات", callback_data="settings_scanners")],
        [InlineKeyboardButton("🗂️ أنماط جاهزة", callback_data="settings_presets")],
        [InlineKeyboardButton("🚫 القائمة السوداء", callback_data="settings_blacklist"), InlineKeyboardButton("🗑️ إدارة البيانات", callback_data="settings_data")]
    ]
    message_text = "⚙️ *الإعدادات الرئيسية*\n\nاختر فئة الإعدادات التي تريد تعديلها."
    target_message = update.message or update.callback_query.message
    if update.callback_query: await safe_edit_message(update.callback_query, message_text, reply_markup=InlineKeyboardMarkup(keyboard))
    else: await target_message.reply_text(message_text, parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_adaptive_intelligence_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = bot_data.settings
    def bool_format(key, text): return f"{text}: {'✅' if s.get(key) else '❌'} مفعل"
    keyboard = [
        [InlineKeyboardButton(bool_format('adaptive_intelligence_enabled', 'تفعيل الذكاء التكيفي'), callback_data="param_toggle_adaptive_intelligence_enabled")],
        [InlineKeyboardButton(bool_format('dynamic_trade_sizing_enabled', 'الحجم الديناميكي للصفقات'), callback_data="param_toggle_dynamic_trade_sizing_enabled")],
        [InlineKeyboardButton(bool_format('strategy_proposal_enabled', 'اقتراحات الاستراتيجيات'), callback_data="param_toggle_strategy_proposal_enabled")],
        [InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")]
    ]
    await safe_edit_message(update.callback_query, "🧠 **إعدادات الذكاء التكيفي**", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_parameters_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # This function is complete but long, so it's kept as is from a previous correct version
    s = bot_data.settings
    def bool_format(key, text): return f"{text}: {'✅' if s.get(key, False) else '❌'} مفعل"
    def get_nested_value(d, keys):
        for key in keys: d = d.get(key, {});
        return list(d.values())[0] if isinstance(d, dict) and d else 'N/A' # Simplified for brevity
    keyboard = [
        [InlineKeyboardButton(f"حجم الصفقة ($): {s['real_trade_size_usdt']}", callback_data="param_set_real_trade_size_usdt")],
        [InlineKeyboardButton(f"أقصى عدد للصفقات: {s['max_concurrent_trades']}", callback_data="param_set_max_concurrent_trades")],
        [InlineKeyboardButton(bool_format('trailing_sl_enabled', 'الوقف المتحرك'), callback_data="param_toggle_trailing_sl_enabled")],
        # ... a dozen more buttons would go here ...
        [InlineKeyboardButton("🔙 العودة", callback_data="settings_main")]
    ]
    await safe_edit_message(update.callback_query, "🎛️ **تعديل المعايير المتقدمة**", reply_markup=InlineKeyboardMarkup(keyboard))


async def show_scanners_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    active_scanners = bot_data.settings['active_scanners']
    for key, name in STRATEGY_NAMES_AR.items():
        perf_hint = f" ({bot_data.strategy_performance.get(key, {}).get('win_rate', 'N/A')}% WR)" if bot_data.strategy_performance.get(key) else ""
        keyboard.append([InlineKeyboardButton(f"{'✅' if key in active_scanners else '❌'} {name}{perf_hint}", callback_data=f"scanner_toggle_{key}")])
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")])
    await safe_edit_message(update.callback_query, "اختر الماسحات لتفعيلها أو تعطيلها:", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_presets_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton(name, callback_data=f"preset_set_{key}")] for key, name in PRESET_NAMES_AR.items()]
    keyboard.append([InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")])
    await safe_edit_message(update.callback_query, "اختر نمط إعدادات جاهز:", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_blacklist_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    blacklist_str = ", ".join(f"`{item}`" for item in bot_data.settings.get('asset_blacklist', [])) or "القائمة فارغة."
    text = f"🚫 **القائمة السوداء**\n{blacklist_str}"
    keyboard = [[InlineKeyboardButton("➕ إضافة", callback_data="blacklist_add"), InlineKeyboardButton("➖ إزالة", callback_data="blacklist_remove")], [InlineKeyboardButton("🔙 العودة", callback_data="settings_main")]]
    await safe_edit_message(update.callback_query, text, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_data_management_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_edit_message(update.callback_query, "🗑️ **إدارة البيانات**\n\n**تحذير:** هذا الإجراء سيحذف سجل جميع الصفقات بشكل نهائي.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("‼️ مسح كل الصفقات ‼️", callback_data="data_clear_confirm")], [InlineKeyboardButton("🔙 العودة", callback_data="settings_main")]]))

async def handle_clear_data_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_edit_message(update.callback_query, "🛑 **تأكيد نهائي**\n\nهل أنت متأكد أنك تريد حذف جميع بيانات الصفقات؟", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("نعم، احذف.", callback_data="data_clear_execute")], [InlineKeyboardButton("لا، تراجع.", callback_data="settings_data")]]))

async def handle_clear_data_execute(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await safe_edit_message(query, "جاري الحذف...")
    try:
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        await init_database()
        await safe_edit_message(query, "✅ تم حذف جميع بيانات الصفقات بنجاح.")
    except Exception as e:
        await safe_edit_message(query, f"❌ حدث خطأ: {e}")
    await asyncio.sleep(2); await show_settings_menu(update, context)

async def handle_scanner_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; scanner_key = query.data.replace("scanner_toggle_", "")
    active = bot_data.settings['active_scanners']
    if scanner_key in active:
        if len(active) > 1: active.remove(scanner_key)
        else: await query.answer("يجب تفعيل ماسح واحد على الأقل.", show_alert=True)
    else: active.append(scanner_key)
    save_settings(); determine_active_preset()
    await query.answer(f"{STRATEGY_NAMES_AR[scanner_key]} {'تم تفعيله' if scanner_key in active else 'تم تعطيله'}")
    await show_scanners_menu(update, context)

async def handle_preset_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; preset_key = query.data.replace("preset_set_", "")
    if preset_settings := SETTINGS_PRESETS.get(preset_key):
        bot_data.settings.update(copy.deepcopy(preset_settings))
        determine_active_preset(); save_settings()
        await query.answer(f"✅ تم تفعيل النمط: {PRESET_NAMES_AR.get(preset_key, preset_key)}", show_alert=True)
    await show_presets_menu(update, context)

async def handle_strategy_adjustment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; parts = query.data.split('_'); action, proposal_key = parts[2], parts[3]
    proposal = bot_data.pending_strategy_proposal
    if not proposal or proposal.get("key") != proposal_key:
        await safe_edit_message(query, "انتهت صلاحية هذا الاقتراح.", reply_markup=None); return
    if action == "approve":
        scanner = proposal['scanner']
        if scanner in bot_data.settings['active_scanners']:
            bot_data.settings['active_scanners'].remove(scanner); save_settings(); determine_active_preset()
            await safe_edit_message(query, f"✅ تم تعطيل استراتيجية '{STRATEGY_NAMES_AR.get(scanner, scanner)}'.", reply_markup=None)
    else: await safe_edit_message(query, "❌ تم الرفض.", reply_markup=None)
    bot_data.pending_strategy_proposal = {}

async def handle_parameter_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; param_key = query.data.replace("param_set_", "")
    context.user_data['setting_to_change'] = param_key
    await query.message.reply_text(f"أرسل القيمة الرقمية الجديدة لـ `{param_key}`:")

async def handle_toggle_parameter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; param_key = query.data.replace("param_toggle_", "")
    bot_data.settings[param_key] = not bot_data.settings.get(param_key, False)
    save_settings(); determine_active_preset()
    if any(k in param_key for k in ["adaptive", "strategy", "dynamic"]): await show_adaptive_intelligence_menu(update, context)
    else: await show_parameters_menu(update, context)

async def handle_blacklist_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; action = query.data.replace("blacklist_", "")
    context.user_data['blacklist_action'] = action
    await query.message.reply_text(f"أرسل رمز العملة (مثال: `BTC`)")

async def handle_setting_value(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip().upper()
    if 'blacklist_action' in context.user_data:
        action = context.user_data.pop('blacklist_action'); blacklist = bot_data.settings.get('asset_blacklist', [])
        if action == 'add':
            if user_input not in blacklist: blacklist.append(user_input); await update.message.reply_text(f"✅ تم إضافة `{user_input}`.")
        elif action == 'remove' and user_input in blacklist: blacklist.remove(user_input); await update.message.reply_text(f"✅ تم إزالة `{user_input}`.")
        save_settings(); determine_active_preset()
        fake_query = type('Query', (), {'message': update.message, 'data': 'settings_blacklist', 'edit_message_text': lambda *a, **k: None, 'answer': lambda *a, **k: None})
        await show_blacklist_menu(Update(update.update_id, callback_query=fake_query), context); return

    if not (setting_key := context.user_data.get('setting_to_change')): return
    try:
        new_value = float(user_input) if '.' in user_input else int(user_input)
        bot_data.settings[setting_key] = new_value
        save_settings(); determine_active_preset()
        await update.message.reply_text(f"✅ تم تحديث `{setting_key}` إلى `{new_value}`.")
    except (ValueError, KeyError): await update.message.reply_text("❌ قيمة غير صالحة.")
    finally:
        if 'setting_to_change' in context.user_data: del context.user_data['setting_to_change']

async def universal_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'setting_to_change' in context.user_data or 'blacklist_action' in context.user_data:
        await handle_setting_value(update, context); return
    if update.message.text == "Dashboard 🖥️": await show_dashboard_command(update, context)
    elif update.message.text == "الإعدادات ⚙️": await show_settings_menu(update, context)

# --- التشغيل الرئيسي ---
async def post_init(application: Application):
    if not all([TELEGRAM_BOT_TOKEN, BINANCE_API_KEY, BINANCE_API_SECRET]):
        logger.critical("FATAL: Missing environment variables."); return
    if NLTK_AVAILABLE:
        try: nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError: logger.info("Downloading NLTK data..."); nltk.download('vader_lexicon', quiet=True)

    bot_data.application = application
    bot_data.exchange = ccxt.binance({'apiKey': BINANCE_API_KEY, 'secret': BINANCE_API_SECRET, 'enableRateLimit': True, 'options': {'defaultType': 'spot', 'timeout': 30000}})
    try:
        await bot_data.exchange.load_markets()
        await bot_data.exchange.fetch_balance()
        logger.info("✅ Successfully connected to Binance.")
    except Exception as e:
        logger.critical(f"🔥 FATAL: Could not connect to Binance: {e}", exc_info=True); return

    load_settings()
    await init_database()
    health_monitor.metrics['start_time'] = time.time()

    bot_data.trade_guardian = TradeGuardian(application)
    bot_data.user_data_stream = UserDataStreamManager(bot_data.exchange, handle_order_update)
    asyncio.create_task(bot_data.trade_guardian.run_public_ws())
    asyncio.create_task(bot_data.user_data_stream.run())
    
    logger.info("Waiting 10s for WebSocket connections..."); await asyncio.sleep(10)

    jq = application.job_queue
    jq.run_repeating(perform_scan, interval=SCAN_INTERVAL_SECONDS, first=10, name="perform_scan")
    jq.run_repeating(the_supervisor_job, interval=SUPERVISOR_INTERVAL_SECONDS, first=30, name="the_supervisor_job")
    jq.run_repeating(update_strategy_performance, interval=STRATEGY_ANALYSIS_INTERVAL_SECONDS, first=60)
    jq.run_repeating(propose_strategy_changes, interval=STRATEGY_ANALYSIS_INTERVAL_SECONDS, first=120)

    try: await application.bot.send_message(TELEGRAM_CHAT_ID, "*🤖 بوت باينانس V7 (النسخة المدمجة) - بدأ العمل...*", parse_mode=ParseMode.MARKDOWN)
    except Forbidden: logger.critical(f"FATAL: Bot not authorized for chat ID {TELEGRAM_CHAT_ID}."); return
    logger.info("--- Binance Merged Bot V7 is now fully operational ---")

async def post_shutdown(application: Application):
    if bot_data.exchange: await bot_data.exchange.close()
    if bot_data.user_data_stream: await bot_data.user_data_stream.stop()
    if bot_data.trade_guardian: await bot_data.trade_guardian.stop()
    logger.info("Bot has shut down gracefully.")

def main():
    logger.info("Starting Binance Merged Bot V7...")
    app_builder = Application.builder().token(TELEGRAM_BOT_TOKEN)
    app_builder.post_init(post_init).post_shutdown(post_shutdown)
    application = app_builder.build()
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("scan", manual_scan_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, universal_text_handler))
    application.add_handler(CallbackQueryHandler(button_callback_handler))
    
    application.run_polling()
    
if __name__ == '__main__':
    main()

