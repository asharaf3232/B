# -*- coding: utf-8 -*-
# =======================================================================================
# --- 🚀 بوت التداول V9 (النسخة المايسترو متعدد الأوضاع) 🚀 ---
# =======================================================================================
#
# هذا الإصدار هو الترقية النهائية التي تدمج كل الميزات من جميع النسخ السابقة،
# مع إضافة نظام ذكي متعدد الأوضاع مشابه لـ OKX Sniper Bot v33.0.
#
# --- سجل التغييرات للنسخة المايسترو V9 ---
#   ✅ [إضافة] المراجع الذكي (Intelligent Reviewer) لإدارة المخاطر.
#   ✅ [إضافة] وضع اقتناص الزخم (Momentum Scalp Mode) للخروج السريع.
#   ✅ [إضافة] فلتر التوافق الزمني (Multi-Timeframe Confluence Filter).
#   ✅ [إضافة] استراتيجية الانعكاس (Bollinger Reversal Strategy).
#   ✅ [إضافة] المايسترو (Maestro) كعقل استراتيجي يدير الأدوات تلقائيًا.
#   ✅ [إضافة] لوحة التحكم الاستراتيجي في تليجرام.
#   ✅ [النتيجة] النسخة المنافسة الكاملة والأكثر ذكاءً.
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
import aiosqlite

# --- مكتبات التحليل والتداول ---
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt
import feedparser
import websockets
import websockets.exceptions

# --- مكتبات الذكاء الاصطناعي (اختياري) ---
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
DB_FILE = 'trading_bot_v9.db'
SETTINGS_FILE = 'trading_bot_v9_settings.json'
DECISION_MATRIX_FILE = 'decision_matrix.json'  # New: For Maestro
TIMEFRAME = '15m'
SCAN_INTERVAL_SECONDS = 900
SUPERVISOR_INTERVAL_SECONDS = 120
INTELLIGENT_REVIEWER_INTERVAL_MINUTES = 30  # New
MAESTRO_INTERVAL_HOURS = 1  # New
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
    "sent_insufficient_funds_warning": False,
    "adaptive_intelligence_enabled": True,
    "dynamic_trade_sizing_enabled": True,
    "strategy_proposal_enabled": True,
    "strategy_analysis_min_trades": 10,
    "strategy_deactivation_threshold_wr": 45.0,
    "dynamic_sizing_max_increase_pct": 25.0,
    "dynamic_sizing_max_decrease_pct": 50.0,
    # New Settings for Multi-Mode Maestro
    "intelligent_reviewer_enabled": True,
    "intelligent_reviewer_interval_minutes": 30,
    "momentum_scalp_mode_enabled": False,
    "momentum_scalp_target_percent": 0.5,
    "multi_timeframe_confluence_enabled": True,
    "maestro_mode_enabled": True,
}

STRATEGY_NAMES_AR = {
    "momentum_breakout": "زخم اختراقي", "breakout_squeeze_pro": "اختراق انضغاطي",
    "support_rebound": "ارتداد الدعم", "sniper_pro": "القناص المحترف", "whale_radar": "رادار الحيتان",
    "rsi_divergence": "دايفرجنس RSI", "supertrend_pullback": "انعكاس سوبرترند",
    # New Strategy
    "bollinger_reversal": "انعكاس بولينجر"
}
PRESET_NAMES_AR = {"professional": "احترافي", "strict": "متشدد", "lenient": "متساهل", "very_lenient": "فائق التساهل", "bold_heart": "القلب الجريء"}
SETTINGS_PRESETS = {
    "professional": copy.deepcopy(DEFAULT_SETTINGS),
    "strict": {**copy.deepcopy(DEFAULT_SETTINGS), "max_concurrent_trades": 3, "risk_reward_ratio": 2.5, "fear_and_greed_threshold": 40, "adx_filter_level": 28, "liquidity_filters": {"min_quote_volume_24h_usd": 2000000, "min_rvol": 2.0}},
    "lenient": {**copy.deepcopy(DEFAULT_SETTINGS), "max_concurrent_trades": 8, "risk_reward_ratio": 1.8, "fear_and_greed_threshold": 25, "adx_filter_level": 20, "liquidity_filters": {"min_quote_volume_24h_usd": 500000, "min_rvol": 1.2}},
    "very_lenient": {
        **copy.deepcopy(DEFAULT_SETTINGS), "max_concurrent_trades": 12, "adx_filter_enabled": False,
        "market_mood_filter_enabled": False, "trend_filters": {"ema_period": 200, "htf_period": 50, "enabled": False},
        "liquidity_filters": {"min_quote_volume_24h_usd": 250000, "min_rvol": 1.0},
        "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.4}, "spread_filter": {"max_spread_percent": 1.5}
    },
    "bold_heart": {
        **copy.deepcopy(DEFAULT_SETTINGS), "max_concurrent_trades": 15, "risk_reward_ratio": 1.5, "multi_timeframe_enabled": False,
        "market_mood_filter_enabled": False, "adx_filter_enabled": False, "btc_trend_filter_enabled": False, "news_filter_enabled": False,
        "volume_filter_multiplier": 1.0, "liquidity_filters": {"min_quote_volume_24h_usd": 100000, "min_rvol": 1.0},
        "volatility_filters": {"atr_period_for_filter": 14, "min_atr_percent": 0.2}, "spread_filter": {"max_spread_percent": 2.0}
    }
}

# New: Decision Matrix for Maestro (JSON-like dict)
DECISION_MATRIX = {
    "TRENDING_HIGH_VOLATILITY": {
        "intelligent_reviewer_enabled": True,
        "momentum_scalp_mode_enabled": True,
        "multi_timeframe_confluence_enabled": True,
        "active_scanners": ["momentum_breakout", "breakout_squeeze_pro", "sniper_pro", "whale_radar"],
        "risk_reward_ratio": 1.5,
        "volume_filter_multiplier": 2.5
    },
    "TRENDING_LOW_VOLATILITY": {
        "intelligent_reviewer_enabled": True,
        "momentum_scalp_mode_enabled": False,
        "multi_timeframe_confluence_enabled": True,
        "active_scanners": ["support_rebound", "supertrend_pullback", "rsi_divergence"],
        "risk_reward_ratio": 2.5,
        "volume_filter_multiplier": 1.5
    },
    "SIDEWAYS_HIGH_VOLATILITY": {
        "intelligent_reviewer_enabled": True,
        "momentum_scalp_mode_enabled": True,
        "multi_timeframe_confluence_enabled": False,
        "active_scanners": ["bollinger_reversal", "rsi_divergence", "breakout_squeeze_pro"],
        "risk_reward_ratio": 2.0,
        "volume_filter_multiplier": 2.0
    },
    "SIDEWAYS_LOW_VOLATILITY": {
        "intelligent_reviewer_enabled": False,
        "momentum_scalp_mode_enabled": False,
        "multi_timeframe_confluence_enabled": True,
        "active_scanners": ["bollinger_reversal", "support_rebound"],
        "risk_reward_ratio": 3.0,
        "volume_filter_multiplier": 1.0
    }
}

# Save Decision Matrix to file if not exists
if not os.path.exists(DECISION_MATRIX_FILE):
    with open(DECISION_MATRIX_FILE, 'w', encoding='utf-8') as f:
        json.dump(DECISION_MATRIX, f, ensure_ascii=False, indent=4)

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
        self.public_ws = None
        self.strategy_performance = {}
        self.pending_strategy_proposal = {}
        self.current_market_regime = "UNKNOWN"  # New: For Maestro

bot_data = BotState()
scan_lock = asyncio.Lock()
trade_management_lock = asyncio.Lock()

# --- وظائف مساعدة وقاعدة البيانات ---
def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f: bot_data.settings = json.load(f)
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
    current_settings_for_compare = {k: v for k, v in bot_data.settings.items() if k in DEFAULT_SETTINGS}
    for name, preset_settings in SETTINGS_PRESETS.items():
        is_match = True
        for key, value in preset_settings.items():
            if key in current_settings_for_compare and current_settings_for_compare[key] != value:
                is_match = False
                break
        if is_match:
            bot_data.active_preset_name = PRESET_NAMES_AR.get(name, "مخصص")
            return
    bot_data.active_preset_name = "مخصص"

def save_settings():
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(bot_data.settings, f, ensure_ascii=False, indent=4)

async def init_database():
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute('CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, symbol TEXT, entry_price REAL, take_profit REAL, stop_loss REAL, quantity REAL, status TEXT, reason TEXT, order_id TEXT, highest_price REAL DEFAULT 0, trailing_sl_active BOOLEAN DEFAULT 0, close_price REAL, pnl_usdt REAL, signal_strength INTEGER DEFAULT 1, close_retries INTEGER DEFAULT 0, last_profit_notification_price REAL DEFAULT 0, trade_weight REAL DEFAULT 1.0)')
            cursor = await conn.execute("PRAGMA table_info(trades)")
            columns = [row[1] for row in await cursor.fetchall()]
            if 'signal_strength' not in columns: await conn.execute("ALTER TABLE trades ADD COLUMN signal_strength INTEGER DEFAULT 1")
            if 'close_retries' not in columns: await conn.execute("ALTER TABLE trades ADD COLUMN close_retries INTEGER DEFAULT 0")
            if 'last_profit_notification_price' not in columns: await conn.execute("ALTER TABLE trades ADD COLUMN last_profit_notification_price REAL DEFAULT 0")
            if 'trade_weight' not in columns: await conn.execute("ALTER TABLE trades ADD COLUMN trade_weight REAL DEFAULT 1.0")
            await conn.commit()
        logger.info("Maestro Edition Database initialized successfully.")
    except Exception as e: logger.critical(f"Database initialization failed: {e}")

async def safe_send_message(bot, text, **kwargs):
    try: await bot.send_message(TELEGRAM_CHAT_ID, text, parse_mode=ParseMode.MARKDOWN, **kwargs)
    except Exception as e: logger.error(f"Telegram Send Error: {e}")

async def safe_edit_message(query, text, **kwargs):
    try: await query.edit_message_text(text, parse_mode=ParseMode.MARKDOWN, **kwargs)
    except BadRequest as e:
        if "Message is not modified" not in str(e): logger.warning(f"Edit Message Error: {e}")
    except Exception as e: logger.error(f"Edit Message Error: {e}")

# --- [وحدة الذكاء التكيفي] ---
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
            reasons = clean_reason.split(' + ')
            for r in set(reasons):
                is_win = 'ناجحة' in status or 'تأمين' in status
                if is_win: stats[r]['wins'] += 1; stats[r]['win_pnl'] += pnl
                else: stats[r]['losses'] += 1; stats[r]['loss_pnl'] += pnl
                stats[r]['total_pnl'] += pnl
        performance_data = {}
        for r, s in stats.items():
            total = s['wins'] + s['losses']
            win_rate = (s['wins'] / total * 100) if total > 0 else 0
            profit_factor = s['win_pnl'] / abs(s['loss_pnl']) if s['loss_pnl'] != 0 else float('inf')
            performance_data[r] = {"win_rate": round(win_rate, 2), "profit_factor": round(profit_factor, 2), "total_trades": total}
        bot_data.strategy_performance = performance_data
        logger.info(f"🧠 Adaptive Mind: Analysis complete. Performance data for {len(performance_data)} strategies updated.")
    except Exception as e: logger.error(f"🧠 Adaptive Mind: Failed to analyze strategy performance: {e}", exc_info=True)


async def propose_strategy_changes(context: ContextTypes.DEFAULT_TYPE):
    s = bot_data.settings
    if not s.get('adaptive_intelligence_enabled') or not s.get('strategy_proposal_enabled'): return
    logger.info("🧠 Adaptive Mind: Checking for underperforming strategies to propose changes...")
    active_scanners = s.get('active_scanners', [])
    min_trades = s.get('strategy_analysis_min_trades', 10)
    deactivation_wr = s.get('strategy_deactivation_threshold_wr', 45.0)

    for scanner in active_scanners:
        perf = bot_data.strategy_performance.get(scanner)
        if perf and perf['total_trades'] >= min_trades and perf['win_rate'] < deactivation_wr:
            if bot_data.pending_strategy_proposal.get('scanner') == scanner: continue
            proposal_key = f"prop_{int(time.time())}"
            bot_data.pending_strategy_proposal = {
                "key": proposal_key, "action": "disable", "scanner": scanner,
                "reason": f"أظهرت أداءً ضعيفًا بمعدل نجاح `{perf['win_rate']}%` في آخر `{perf['total_trades']}` صفقة."
            }
            logger.warning(f"🧠 Adaptive Mind: Proposing to disable '{scanner}' due to low performance.")
            message = (f"💡 **اقتراح تحسين الأداء** 💡\n\n"
                       f"مرحباً، بناءً على التحليل المستمر، لاحظت أن استراتيجية **'{STRATEGY_NAMES_AR.get(scanner, scanner)}'** "
                       f"{bot_data.pending_strategy_proposal['reason']}\n\n"
                       f"أقترح تعطيلها مؤقتًا للتركيز على الاستراتيجيات الأكثر ربحية. هل توافق على هذا التعديل؟")
            keyboard = [[InlineKeyboardButton("✅ موافقة", callback_data=f"strategy_adjust_approve_{proposal_key}"),
                         InlineKeyboardButton("❌ رفض", callback_data=f"strategy_adjust_reject_{proposal_key}")]]
            await safe_send_message(context.bot, message, reply_markup=InlineKeyboardMarkup(keyboard))
            return

async def translate_text_gemini(text_list):
    if not GEMINI_API_KEY: logger.warning("GEMINI_API_KEY not found. Skipping translation."); return text_list, False
    if not text_list: return [], True
    prompt = "Translate the following English headlines to Arabic. Return only the translated text, with each headline on a new line:\n\n" + "\n".join(text_list)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            result = response.json()
            translated_text = result['candidates'][0]['content']['parts'][0]['text']
            return translated_text.strip().split('\n'), True
    except Exception as e: logger.error(f"Gemini translation failed: {e}"); return text_list, False

def get_alpha_vantage_economic_events():
    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == 'YOUR_AV_KEY_HERE': return []
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    params = {'function': 'ECONOMIC_CALENDAR', 'horizon': '3month', 'apikey': ALPHA_VANTAGE_API_KEY}
    try:
        response = httpx.get('https://www.alphavantage.co/query', params=params, timeout=20)
        response.raise_for_status(); data_str = response.text
        if "premium" in data_str.lower(): return []
        lines = data_str.strip().split('\r\n')
        if len(lines) < 2: return []
        header = [h.strip() for h in lines[0].split(',')]
        events = [dict(zip(header, [v.strip() for v in line.split(',')])) for line in lines[1:]]
        high_impact_events = [e.get('event', 'Unknown Event') for e in events if e.get('releaseDate', '') == today_str and e.get('impact', '').lower() == 'high' and e.get('country', '') in ['USD', 'EUR']]
        if high_impact_events: logger.warning(f"High-impact events today: {high_impact_events}")
        return high_impact_events
    except httpx.RequestError as e: logger.error(f"Failed to fetch economic calendar: {e}"); return None

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

async def get_fundamental_market_mood():
    s = bot_data.settings
    if not s.get('news_filter_enabled', True): return {"mood": "POSITIVE", "reason": "فلتر الأخبار معطل"}
    high_impact_events = await asyncio.to_thread(get_alpha_vantage_economic_events)
    if high_impact_events is None: return {"mood": "DANGEROUS", "reason": "فشل جلب البيانات الاقتصادية"}
    if high_impact_events: return {"mood": "DANGEROUS", "reason": f"أحداث هامة اليوم: {', '.join(high_impact_events)}"}
    latest_headlines = await asyncio.to_thread(get_latest_crypto_news)
    sentiment, score = analyze_sentiment_of_headlines(latest_headlines)
    logger.info(f"Market sentiment score: {score:.2f} ({sentiment})")
    if score > 0.25: return {"mood": "POSITIVE", "reason": f"مشاعر إيجابية (الدرجة: {score:.2f})"}
    elif score < -0.25: return {"mood": "NEGATIVE", "reason": f"مشاعر سلبية (الدرجة: {score:.2f})"}
    else: return {"mood": "NEUTRAL", "reason": f"مشاعر محايدة (الدرجة: {score:.2f})"}

def find_col(df_columns, prefix):
    try: return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration: return None

async def get_fear_and_greed_index():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            return int(r.json()['data'][0]['value'])
    except Exception: return None

async def get_market_mood():
    s = bot_data.settings
    if s.get('btc_trend_filter_enabled', True):
        try:
            htf_period = s['trend_filters']['htf_period']
            ohlcv = await bot_data.exchange.fetch_ohlcv('BTC/USDT', '4h', limit=htf_period + 5)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['sma'] = ta.sma(df['close'], length=htf_period)
            is_btc_bullish = df['close'].iloc[-1] > df['sma'].iloc[-1]
            btc_mood_text = "صاعد ✅" if is_btc_bullish else "هابط ❌"
            if not is_btc_bullish: return {"mood": "NEGATIVE", "reason": "اتجاه BTC هابط", "btc_mood": btc_mood_text}
        except Exception as e: return {"mood": "DANGEROUS", "reason": f"فشل جلب بيانات BTC: {e}", "btc_mood": "UNKNOWN"}
    else: btc_mood_text = "الفلتر معطل"
    if s.get('market_mood_filter_enabled', True):
        fng = await get_fear_and_greed_index()
        if fng is not None and fng < s['fear_and_greed_threshold']:
            return {"mood": "NEGATIVE", "reason": f"مشاعر خوف شديد (F&G: {fng})", "btc_mood": btc_mood_text}
    return {"mood": "POSITIVE", "reason": "وضع السوق مناسب", "btc_mood": btc_mood_text}

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

# New: Bollinger Reversal Strategy
def analyze_bollinger_reversal(df, params, rvol, adx_value):
    df.ta.bbands(length=20, append=True)
    df.ta.rsi(append=True)
    bbl_col, bbm_col, bbu_col = find_col(df.columns, "BBL_20_2.0"), find_col(df.columns, "BBM_20_2.0"), find_col(df.columns, "BBU_20_2.0")
    rsi_col = find_col(df.columns, "RSI_14")
    if not all([bbl_col, bbm_col, bbu_col, rsi_col]): return None
    last, prev = df.iloc[-2], df.iloc[-3]
    # Entry: Candle closes below lower BB, followed by candle closing inside
    if prev['close'] < prev[bbl_col] and last['close'] > last[bbl_col] and last['close'] < last[bbm_col] and last['rsi'] < 35:
        entry_price = last['close']
        stop_loss = prev['low']
        take_profit = last[bbm_col]
        return {"reason": "bollinger_reversal", "entry_price": entry_price, "take_profit": take_profit, "stop_loss": stop_loss}
    return None

SCANNERS = {
    "momentum_breakout": analyze_momentum_breakout, "breakout_squeeze_pro": analyze_breakout_squeeze_pro,
    "support_rebound": analyze_support_rebound, "sniper_pro": analyze_sniper_pro, "whale_radar": analyze_whale_radar,
    "rsi_divergence": analyze_rsi_divergence, "supertrend_pullback": analyze_supertrend_pullback,
    # New Strategy
    "bollinger_reversal": analyze_bollinger_reversal
}

# --- محرك التداول المحصن ---
async def get_binance_markets():
    settings = bot_data.settings
    if time.time() - bot_data.last_markets_fetch > 300:
        try:
            logger.info("Fetching and caching all Binance markets...")
            all_tickers = await bot_data.exchange.fetch_tickers()
            bot_data.all_markets = list(all_tickers.values()); bot_data.last_markets_fetch = time.time()
        except Exception as e: logger.error(f"Failed to fetch all markets: {e}"); return []
    blacklist = settings.get('asset_blacklist', [])
    valid_markets = [t for t in bot_data.all_markets if t.get('symbol') and 'USDT' in t['symbol'] and t['symbol'].split('/')[0] not in blacklist and t.get('quoteVolume', 0) > settings['liquidity_filters']['min_quote_volume_24h_usd'] and t.get('active', True) and not any(k in t['symbol'] for k in ['-SWAP', 'UP', 'DOWN', '3L', '3S'])]
    valid_markets.sort(key=lambda m: m.get('quoteVolume', 0), reverse=True)
    return valid_markets[:settings['top_n_symbols_by_volume']]

async def fetch_ohlcv_batch(exchange, symbols, timeframe, limit):
    tasks = [exchange.fetch_ohlcv(s, timeframe, limit=limit) for s in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return {symbols[i]: results[i] for i in range(len(symbols)) if not isinstance(results[i], Exception)}

# Modified: Multi-Timeframe Confluence Filter in Worker Batch
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
            if len(df) < 50: queue.task_done(); continue
            orderbook = await exchange.fetch_order_book(symbol, limit=1)
            best_bid, best_ask = orderbook['bids'][0][0], orderbook['asks'][0][0]
            if best_bid <= 0: queue.task_done(); continue
            spread_percent = ((best_ask - best_bid) / best_bid) * 100
            if 'whale_radar' in settings['active_scanners']:
                whale_radar_signal = await analyze_whale_radar(df.copy(), {}, 0, 0, exchange, symbol)
                if whale_radar_signal and spread_percent <= settings['spread_filter']['max_spread_percent'] * 2:
                    reason_str, strength = whale_radar_signal['reason'], 5
                    entry_price = df.iloc[-2]['close']
                    df.ta.atr(length=14, append=True)
                    atr = df.iloc[-2].get(find_col(df.columns, "ATRr_14"), 0)
                    risk = atr * settings['atr_sl_multiplier']
                    stop_loss, take_profit = entry_price - risk, entry_price + (risk * settings['risk_reward_ratio'])
                    signals_list.append({"symbol": symbol, "entry_price": entry_price, "take_profit": take_profit, "stop_loss": stop_loss, "reason": reason_str, "strength": strength, "weight": 1.0})
                    queue.task_done(); continue
            if spread_percent > settings['spread_filter']['max_spread_percent']: queue.task_done(); continue
            # New: Multi-Timeframe Confluence Filter
            is_confluence_valid = True
            if settings.get('multi_timeframe_confluence_enabled', True):
                # Fetch 1h and 4h data
                ohlcv_1h = await exchange.fetch_ohlcv(symbol, '1h', limit=100)
                ohlcv_4h = await exchange.fetch_ohlcv(symbol, '4h', limit=100)
                df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms')
                df_1h = df_1h.set_index('timestamp').sort_index()
                df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms')
                df_4h = df_4h.set_index('timestamp').sort_index()
                # MACD and SMA(50) on 1h
                df_1h.ta.macd(append=True)
                df_1h.ta.sma(length=50, append=True)
                macd_col = find_col(df_1h.columns, "MACD_12_26_9")
                sma_col = find_col(df_1h.columns, "SMA_50")
                macd_positive = df_1h[macd_col].iloc[-1] > 0 if macd_col else False
                price_above_sma = df_1h['close'].iloc[-1] > df_1h[sma_col].iloc[-1] if sma_col else False
                # EMA(200) on 4h
                df_4h.ta.ema(length=200, append=True)
                ema_col = find_col(df_4h.columns, "EMA_200")
                price_above_ema = df_4h['close'].iloc[-1] > df_4h[ema_col].iloc[-1] if ema_col else False
                is_confluence_valid = macd_positive and price_above_sma and price_above_ema
                if not is_confluence_valid:
                    logger.info(f"Confluence filter blocked {symbol}")
                    queue.task_done()
                    continue
            is_htf_bullish = True
            if settings.get('multi_timeframe_enabled', True):
                ohlcv_htf = await exchange.fetch_ohlcv(symbol, settings.get('multi_timeframe_htf'), limit=220)
                df_htf = pd.DataFrame(ohlcv_htf, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                if len(df_htf) > 200:
                    df_htf['timestamp'] = pd.to_datetime(df_htf['timestamp'], unit='ms')
                    df_htf = df_htf.set_index('timestamp').sort_index()
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
                if not (strategy_func := SCANNERS.get(name)): continue
                params = settings.get(name, {})
                func_args = {'df': df.copy(), 'params': params, 'rvol': rvol, 'adx_value': adx_value}
                if name in ['support_rebound']: func_args.update({'exchange': exchange, 'symbol': symbol})
                result = await strategy_func(**func_args) if asyncio.iscoroutinefunction(strategy_func) else strategy_func(**{k: v for k, v in func_args.items() if k not in ['exchange', 'symbol']})
                if result: confirmed_reasons.append(result['reason'])
            if confirmed_reasons:
                reason_str, strength = ' + '.join(set(confirmed_reasons)), len(set(confirmed_reasons))
                trade_weight = 1.0
                if settings.get('adaptive_intelligence_enabled', True):
                    primary_reason = confirmed_reasons[0]
                    perf = bot_data.strategy_performance.get(primary_reason)
                    if perf:
                        if perf['win_rate'] < 50 and perf['total_trades'] > 5: trade_weight = 1 - (settings['dynamic_sizing_max_decrease_pct'] / 100.0)
                        elif perf['win_rate'] > 70 and perf['profit_factor'] > 1.5: trade_weight = 1 + (settings['dynamic_sizing_max_increase_pct'] / 100.0)
                        logger.info(f"🧠 Adaptive Mind: Strategy '{primary_reason}' WR: {perf['win_rate']}%. Applying trade weight: {trade_weight:.2f}")
                        if perf['win_rate'] < settings['strategy_deactivation_threshold_wr'] and perf['total_trades'] > settings['strategy_analysis_min_trades']:
                           logger.warning(f"Signal for {symbol} from weak strategy '{primary_reason}' ignored."); queue.task_done(); continue
                if not is_htf_bullish:
                    strength = max(1, int(strength / 2)); reason_str += " (اتجاه كبير ضعيف)"; trade_weight *= 0.8
                entry_price = df.iloc[-2]['close']
                df.ta.atr(length=14, append=True)
                atr = df.iloc[-2].get(find_col(df.columns, "ATRr_14"), 0)
                risk = atr * settings['atr_sl_multiplier']
                stop_loss, take_profit = entry_price - risk, entry_price + (risk * settings['risk_reward_ratio'])
                signals_list.append({"symbol": symbol, "entry_price": entry_price, "take_profit": take_profit, "stop_loss": stop_loss, "reason": reason_str, "strength": strength, "weight": trade_weight})
            queue.task_done()
        except Exception as e:
            if 'symbol' in locals(): logger.error(f"Error processing symbol {symbol}: {e}", exc_info=True); errors_list.append(symbol)
            else: logger.error(f"Worker error with no symbol context: {e}", exc_info=True)
            if not queue.empty(): queue.task_done()

async def initiate_real_trade(signal):
    if not bot_data.trading_enabled:
        logger.warning(f"Trade for {signal['symbol']} blocked: Kill Switch active."); return False
    try:
        settings, exchange = bot_data.settings, bot_data.exchange; await exchange.load_markets()
        base_trade_size = settings['real_trade_size_usdt']; trade_weight = signal.get('weight', 1.0)
        if settings.get('dynamic_trade_sizing_enabled', True): trade_size = base_trade_size * trade_weight
        else: trade_size = base_trade_size
        balance = await exchange.fetch_balance(); usdt_balance = balance.get('USDT', {}).get('free', 0.0)
        if usdt_balance < trade_size:
             logger.error(f"Insufficient USDT for {signal['symbol']}. Have: {usdt_balance}, Need: {trade_size}")
             if not settings.get('sent_insufficient_funds_warning'):
                 await safe_send_message(bot_data.application.bot, "🚨 **فشل الشراء: رصيد غير كافٍ**\n"
                                                                  f"لا يمكن فتح صفقة جديدة لأن رصيدك من USDT أقل من حجم الصفقة المحدَّد.")
                 settings['sent_insufficient_funds_warning'] = True; save_settings()
             return False
        if settings.get('sent_insufficient_funds_warning'):
            settings['sent_insufficient_funds_warning'] = False; save_settings()
        base_amount = trade_size / signal['entry_price']
        formatted_amount = exchange.amount_to_precision(signal['symbol'], base_amount)
        buy_order = await exchange.create_market_buy_order(signal['symbol'], formatted_amount)
        if await log_pending_trade_to_db(signal, buy_order):
            await safe_send_message(bot_data.application.bot, f"🚀 تم إرسال أمر شراء لـ `{signal['symbol']}`."); return True
        else:
            await exchange.cancel_order(buy_order['id'], signal['symbol']); return False
    except ccxt.InsufficientFunds as e: logger.error(f"REAL TRADE FAILED {signal['symbol']}: {e}"); await safe_send_message(bot_data.application.bot, f"⚠️ **رصيد غير كافٍ!**"); return False
    except Exception as e: logger.error(f"REAL TRADE FAILED {signal['symbol']}: {e}", exc_info=True); return False

async def log_pending_trade_to_db(signal, buy_order):
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute("INSERT INTO trades (timestamp, symbol, reason, order_id, status, entry_price, take_profit, stop_loss, signal_strength, last_profit_notification_price, trade_weight) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (datetime.now(EGYPT_TZ).isoformat(), signal['symbol'], signal['reason'], buy_order['id'], 'pending', signal['entry_price'], signal['take_profit'], signal['stop_loss'], signal.get('strength', 1), signal['entry_price'], signal.get('weight', 1.0)))
            await conn.commit()
            logger.info(f"Logged pending trade for {signal['symbol']} with order ID {buy_order['id']}.")
            return True
    except Exception as e: logger.error(f"DB Log Pending Error: {e}"); return False

async def activate_trade_binance(order_details, original_signal_data):
    bot = bot_data.application.bot; log_ctx = {'trade_id': 'N/A'}
    try:
        filled_price, gross_filled_quantity = order_details.get('average', 0.0), order_details.get('filled', 0.0)
        if gross_filled_quantity <= 0 or filled_price <= 0:
            logger.error(f"Order {order_details['id']} invalid fill data. Price: {filled_price}, Qty: {gross_filled_quantity}."); return
        net_filled_quantity = gross_filled_quantity
        base_currency = order_details['symbol'].split('/')[0]
        if 'fee' in order_details and order_details['fee'] and 'cost' in order_details['fee']:
            fee_cost, fee_currency = order_details['fee']['cost'], order_details['fee']['currency']
            if fee_currency == base_currency:
                net_filled_quantity -= fee_cost
                logger.info(f"Fee of {fee_cost} {fee_currency} deducted. Net quantity for {order_details['symbol']} is {net_filled_quantity}.")
        if net_filled_quantity <= 0: logger.error(f"Net quantity for {order_details['id']} is zero or less. Aborting."); return
        balance_after = await bot_data.exchange.fetch_balance()
        usdt_remaining = balance_after.get('USDT', {}).get('free', 0)
    except Exception as e:
        logger.error(f"Could not fetch data for trade activation: {e}", exc_info=True)
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute("UPDATE trades SET status = 'failed', reason = 'Activation Fetch Error' WHERE order_id = ?", (order_details['id'],)); await conn.commit()
        return
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        trade = await (await conn.execute("SELECT * FROM trades WHERE order_id = ? AND status = 'pending'", (order_details['id'],))).fetchone()
        if not trade: logger.info(f"Activation ignored for {order_details['id']}: Trade not pending."); return
        trade = dict(trade); log_ctx['trade_id'] = trade['id']
        logger.info(f"Activating trade #{trade['id']} for {order_details['symbol']}...", extra=log_ctx)
        risk = filled_price - trade['stop_loss']
        new_take_profit = filled_price + (risk * bot_data.settings['risk_reward_ratio'])
        await conn.execute("UPDATE trades SET status = 'active', entry_price = ?, quantity = ?, take_profit = ? WHERE id = ?", (filled_price, net_filled_quantity, new_take_profit, trade['id']))
        active_trades_count = (await (await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active'")).fetchone())[0]
        await conn.commit()

    await bot_data.public_ws.subscribe([order_details['symbol']])
    trade_cost, tp_percent, sl_percent = filled_price * net_filled_quantity, (new_take_profit / filled_price - 1) * 100, (1 - trade['stop_loss'] / filled_price) * 100
    reasons_en = trade['reason'].split(' + ')
    reasons_ar = [STRATEGY_NAMES_AR.get(r.strip(), r.strip()) for r in reasons_en]
    reason_display_str = ' + '.join(reasons_ar)
    strength_stars = '⭐' * trade.get('signal_strength', 1)
    trade_weight = trade.get('trade_weight', 1.0)
    confidence_level_str = f"**🧠 مستوى الثقة:** `{trade_weight:.0%}` (تم تعديل الحجم)\n" if trade_weight != 1.0 else ""

    success_msg = (f"✅ **تم تأكيد الشراء | {order_details['symbol']}**\n"
                   f"**الاستراتيجية:** {reason_display_str}\n"
                   f"**قوة الإشارة:** {strength_stars}\n"
                   f"{confidence_level_str}"
                   f"🔸 **الصفقة رقم:** #{trade['id']}\n"
                   f"🔸 **سعر التنفيذ:** `${filled_price:,.4f}`\n"
                   f"🔸 **الكمية (صافي):** {net_filled_quantity:,.4f} {order_details['symbol'].split('/')[0]}\n"
                   f"🔸 **التكلفة:** `${trade_cost:,.2f}`\n"
                   f"🎯 **الهدف (TP):** `${new_take_profit:,.4f} (ربح متوقع: {tp_percent:+.2f}%)`\n"
                   f"🛡️ **الوقف (SL):** `${trade['stop_loss']:,.4f} (خسارة مقبولة: {sl_percent:.2f}%)`\n"
                   f"💰 **السيولة المتبقية (USDT):** `${usdt_remaining:,.2f}`\n"
                   f"🔄 **إجمالي الصفقات النشطة:** `{active_trades_count}`\n"
                   f"الحارس الأمين يراقب الصفقة الآن.")
    await safe_send_message(bot, success_msg)

async def the_supervisor_job(context: ContextTypes.DEFAULT_TYPE):
    logger.info("🕵️ Supervisor: Auditing pending trades...")
    async with aiosqlite.connect(DB_FILE) as conn:
        conn.row_factory = aiosqlite.Row
        two_mins_ago = (datetime.now(EGYPT_TZ) - timedelta(minutes=2)).isoformat()
        stuck_trades = await (await conn.execute("SELECT * FROM trades WHERE status = 'pending' AND timestamp <= ?", (two_mins_ago,))).fetchall()
        if not stuck_trades: logger.info("🕵️ Supervisor: Audit complete. No abandoned trades found."); return
        for trade_data in stuck_trades:
            trade = dict(trade_data); order_id, symbol = trade['order_id'], trade['symbol']
            logger.warning(f"🕵️ Supervisor: Found abandoned trade #{trade['id']}. Investigating.", extra={'trade_id': trade['id']})
            try:
                order_status = await bot_data.exchange.fetch_order(order_id, symbol)
                if order_status['status'] == 'closed' and order_status.get('filled', 0) > 0:
                    logger.info(f"🕵️ Supervisor: API confirms {order_id} was filled. Activating.", extra={'trade_id': trade['id']})
                    await activate_trade_binance(order_status, trade)
                elif order_status['status'] == 'canceled': await conn.execute("UPDATE trades SET status = 'failed' WHERE id = ?", (trade['id'],))
                else: await bot_data.exchange.cancel_order(order_id, symbol); await conn.execute("UPDATE trades SET status = 'failed' WHERE id = ?", (trade['id'],))
                await conn.commit()
            except Exception as e: logger.error(f"🕵️ Supervisor: Failed to rectify trade #{trade['id']}: {e}", extra={'trade_id': trade['id']})

# New: Task 1 - Intelligent Reviewer Job
async def intelligent_reviewer_job(context: ContextTypes.DEFAULT_TYPE):
    if not bot_data.settings.get('intelligent_reviewer_enabled', True):
        return
    logger.info("🧠 Intelligent Reviewer: Reviewing active trades for signal validity...")
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            conn.row_factory = aiosqlite.Row
            active_trades = await (await conn.execute("SELECT * FROM trades WHERE status = 'active'")).fetchall()
        for trade in active_trades:
            trade_dict = dict(trade)
            symbol = trade_dict['symbol']
            reason = trade_dict['reason'].split(' + ')[0]  # Primary reason
            if reason not in SCANNERS:
                continue
            # Fetch fresh OHLCV data
            ohlcv = await bot_data.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=220)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp').sort_index()
            if len(df) < 50:
                continue
            # Re-run the original analyzer
            analyzer_func = SCANNERS[reason]
            result = analyzer_func(df, bot_data.settings.get(reason, {}), 0, 0)
            if not result:
                # Signal invalidated, close trade
                current_price = df['close'].iloc[-1]
                await close_trade(trade_dict, "Signal Invalidated (Reviewer)", current_price, context)
                logger.info(f"🧠 Intelligent Reviewer: Closed trade #{trade['id']} for {symbol} - Signal invalidated.")
    except Exception as e:
        logger.error(f"🧠 Intelligent Reviewer Job failed: {e}", exc_info=True)

# Modified: Task 2 - Momentum Scalp Mode in check_and_close_trade (renamed to handle_ticker_update)
async def handle_ticker_update(ticker_data, context):
    symbol = ticker_data['symbol']; current_price = ticker_data['price']
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            conn.row_factory = aiosqlite.Row
            trade = await (await conn.execute("SELECT * FROM trades WHERE symbol = ? AND status = 'active'", (symbol,))).fetchone()
            if not trade: return
            trade = dict(trade); settings = bot_data.settings
            # Priority 1: SL
            if current_price <= trade['stop_loss']:
                await close_trade(trade, "فاشلة (SL)", current_price, context)
                return
            # Priority 2: Momentum Scalp Mode
            if settings.get('momentum_scalp_mode_enabled', False):
                scalp_target = trade['entry_price'] * (1 + settings['momentum_scalp_target_percent'] / 100)
                if current_price >= scalp_target:
                    await close_trade(trade, "ناجحة (Scalp Mode)", current_price, context)
                    logger.info(f"💸 Momentum Scalp: Closed #{trade['id']} at {current_price:.4f}")
                    return
            # Priority 3: Original Logic (TP, TSL, Notifications)
            if settings['trailing_sl_enabled']:
                new_highest_price = max(trade.get('highest_price', 0), current_price)
                if new_highest_price > trade.get('highest_price', 0):
                    await conn.execute("UPDATE trades SET highest_price = ? WHERE id = ?", (new_highest_price, trade['id']))
                if not trade['trailing_sl_active'] and current_price >= trade['entry_price'] * (1 + settings['trailing_sl_activation_percent'] / 100):
                    trade['trailing_sl_active'] = True; trade['stop_loss'] = trade['entry_price']
                    await conn.execute("UPDATE trades SET trailing_sl_active = 1, stop_loss = ? WHERE id = ?", (trade['entry_price'], trade['id']))
                    await safe_send_message(context.bot, f"**🚀 تأمين الأرباح! | #{trade['id']} {symbol}**\nتم رفع وقف الخسارة إلى نقطة الدخول: `${trade['entry_price']}`")
                if trade['trailing_sl_active']:
                    new_sl = new_highest_price * (1 - settings['trailing_sl_callback_percent'] / 100)
                    if new_sl > trade['stop_loss']:
                        trade['stop_loss'] = new_sl
                        await conn.execute("UPDATE trades SET stop_loss = ? WHERE id = ?", (new_sl, trade['id']))
            if settings.get('incremental_notifications_enabled', False):
                last_notified_price = trade.get('last_profit_notification_price', trade['entry_price'])
                entry_price = trade['entry_price']
                increment_percent = settings.get('incremental_notification_percent', 2.0)
                next_notification_target = last_notified_price * (1 + increment_percent / 100)
                if current_price >= next_notification_target:
                    total_profit_percent = ((current_price / entry_price) - 1) * 100
                    await safe_send_message(context.bot, f"📈 **ربح متزايد! | #{trade['id']} {symbol}**\n**الربح الحالي:** `{total_profit_percent:+.2f}%`")
                    await conn.execute("UPDATE trades SET last_profit_notification_price = ? WHERE id = ?", (current_price, trade['id']))
            await conn.commit()
            # TP Check (after priorities)
            if current_price >= trade['take_profit']: await close_trade(trade, "ناجحة (TP)", current_price, context)
    except Exception as e: logger.error(f"Guardian Ticker Error for {symbol}: {e}", exc_info=True)

# New: Task 5 - Market Regime Analyzer for Maestro
async def get_market_regime():
    try:
        # Fetch BTC data for ADX and ATR%
        ohlcv = await bot_data.exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.ta.adx(append=True)
        df.ta.atr(append=True)
        adx_col = find_col(df.columns, "ADX_14")
        atr_col = find_col(df.columns, "ATRr_14")
        if adx_col and atr_col:
            adx = df[adx_col].iloc[-1]
            atr_percent = (df[atr_col].iloc[-1] / df['close'].iloc[-1]) * 100
            if adx > 25:
                trend = "TRENDING"
            else:
                trend = "SIDEWAYS"
            if atr_percent > 2.0:
                vol = "HIGH_VOLATILITY"
            else:
                vol = "LOW_VOLATILITY"
            regime = f"{trend}_{vol}"
            bot_data.current_market_regime = regime
            return regime
    except Exception as e:
        logger.error(f"Market Regime Analysis failed: {e}")
    return "UNKNOWN"

# New: Task 5 - Maestro Job
async def maestro_job(context: ContextTypes.DEFAULT_TYPE):
    if not bot_data.settings.get('maestro_mode_enabled', True):
        return
    logger.info("🎼 Maestro: Analyzing market regime and adjusting tactics...")
    regime = await get_market_regime()
    try:
        with open(DECISION_MATRIX_FILE, 'r', encoding='utf-8') as f:
            matrix = json.load(f)
        if regime in matrix:
            config = matrix[regime]
            # Apply changes
            for key, value in config.items():
                if key in bot_data.settings:
                    old_value = bot_data.settings[key]
                    bot_data.settings[key] = value
                    if old_value != value:
                        logger.info(f"🎼 Maestro: Updated {key} from {old_value} to {value} for regime {regime}")
            save_settings()
            # Send report
            active_scanners_str = ' + '.join([STRATEGY_NAMES_AR.get(s, s) for s in config.get('active_scanners', [])])
            report = (f"🎼 **تقرير المايسترو | {regime}**\n"
                      f"تم تعديل التكوين ليتناسب مع حالة السوق.\n"
                      f"الاستراتيجيات النشطة: {active_scanners_str}\n"
                      f"نسبة المخاطرة/العائد: {config.get('risk_reward_ratio', 'N/A')}")
            await safe_send_message(context.bot, report)
        else:
            logger.warning(f"🎼 Maestro: Unknown regime {regime}, no config applied.")
    except Exception as e:
        logger.error(f"🎼 Maestro Job failed: {e}")

async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    async with scan_lock:
        if not bot_data.trading_enabled: logger.info("Scan skipped: Kill Switch is active."); return
        scan_start_time = time.time()
        logger.info("--- Starting new Adaptive Intelligence scan... ---")
        settings, bot = bot_data.settings, context.bot
        if settings.get('news_filter_enabled', True):
            mood_result_fundamental = await get_fundamental_market_mood()
            if mood_result_fundamental['mood'] in ["NEGATIVE", "DANGEROUS"]:
                bot_data.market_mood = mood_result_fundamental
                logger.warning(f"SCAN SKIPPED: Fundamental mood is {mood_result_fundamental['mood']}. Reason: {mood_result_fundamental['reason']}")
                await safe_send_message(bot, f"🚨 **تنبيه: فحص السوق تم إيقافه!**\n"
                                           f"━━━━━━━━━━━━━━━━━━━━\n"
                                           f"**السبب:** {mood_result_fundamental['reason']}\n"
                                           f"**الإجراء:** تم تخطي الفحص لحماية رأس المال من تقلبات الأخبار والبيانات الاقتصادية الهامة.")
                return
        mood_result = await get_market_mood()
        bot_data.market_mood = mood_result
        if mood_result['mood'] in ["NEGATIVE", "DANGEROUS"]:
            logger.warning(f"SCAN SKIPPED: {mood_result['reason']}")
            await safe_send_message(bot, f"🚨 **تنبيه: فحص السوق تم إيقافه!**\n"
                                       f"━━━━━━━━━━━━━━━━━━━━\n"
                                       f"**السبب الرئيسي:** {mood_result['reason']}\n"
                                       f"**التفاصيل:** تم تخطي الفحص التلقائي بسبب عدم استيفاء شروط الدخول الصارمة.\n"
                                       f"💡 **ماذا يعني هذا؟**\n"
                                       f"يُشير ذلك إلى أن السوق في حالة من عدم اليقين أو الاتجاه الهابط، مما يزيد من مخاطر التداول. يفضل البوت حماية رأس المال على الدخول في صفقات عالية المخاطر.\n"
                                       f"**حالة مؤشرات السوق:**\n"
                                       f"  - **اتجاه BTC:** {mood_result.get('btc_mood', 'N/A')}\n"
                                       f"  - **مزاج السوق:** {bot_data.market_mood.get('reason', 'N/A')}")
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
            if active_trades_count >= settings['max_concurrent_trades']: break
            if time.time() - bot_data.last_signal_time.get(signal['symbol'], 0) > (SCAN_INTERVAL_SECONDS * 0.9):
                bot_data.last_signal_time[signal['symbol']] = time.time()
                
                if await initiate_real_trade(signal):
                    active_trades_count += 1; trades_opened_count += 1
                await asyncio.sleep(2)

        scan_duration = time.time() - scan_start_time
        bot_data.last_scan_info = {"start_time": datetime.fromtimestamp(scan_start_time, EGYPT_TZ).strftime('%Y-%m-%d %H:%M:%S'), "duration_seconds": int(scan_duration), "checked_symbols": len(top_markets), "analysis_errors": len(analysis_errors)}
        await safe_send_message(bot, f"✅ **فحص السوق اكتمل بنجاح**\n"
                                   f"━━━━━━━━━━━━━━━━━━\n"
                                   f"**المدة:** {int(scan_duration)} ثانية | **العملات المفحوصة:** {len(top_markets)}\n"
                                   f"**النتائج:**\n"
                                   f"  - **إشارات جديدة:** {len(signals_found)}\n"
                                   f"  - **صفقات تم فتحها:** {trades_opened_count} صفقة\n"
                                   f"  - **مشكلات تحليل:** {len(analysis_errors)} عملة")

class BinanceWebSocketManager:
    def __init__(self):
        self.ws = None; self.uri = "wss://stream.binance.com:9443/ws/"; self.subscriptions = set()
        self.ticker_queue = asyncio.Queue(); self.is_connected = False; self.reconnect_task = None
    async def _handle_message(self, message):
        data = json.loads(message)
        if 'e' in data and data['e'] == 'kline':
            symbol = data['s'].replace('USDT', '/USDT'); price = float(data['k']['c'])
            await self.ticker_queue.put({'symbol': symbol, 'price': price})
    async def _manage_connections(self):
        while True:
            if not self.subscriptions: await asyncio.sleep(5); continue
            stream_name = '/'.join([f"{s.lower().replace('/', '')}@kline_1m" for s in self.subscriptions])
            current_uri = self.uri + stream_name
            if self.is_connected and self.ws and not self.ws.closed: await asyncio.sleep(5); continue
            try:
                self.ws = await websockets.connect(current_uri, ping_interval=30, ping_timeout=10)
                self.is_connected = True
                async for message in self.ws: await self._handle_message(message)
            except (websockets.exceptions.ConnectionClosed, Exception) as e:
                self.is_connected = False; await asyncio.sleep(5)
    def start(self): asyncio.create_task(self._manage_connections()); asyncio.create_task(self._process_queue())
    async def subscribe(self, symbols):
        new_symbols = [s for s in symbols if s not in self.subscriptions]
        if new_symbols: self.subscriptions.update(new_symbols); logger.info(f"Subscribed to WS for: {new_symbols}")
        if self.ws and not self.ws.closed: await self.ws.close()
    async def unsubscribe(self, symbols):
        for s in symbols: self.subscriptions.discard(s)
        if self.ws and not self.ws.closed: await self.ws.close()
    async def _process_queue(self):
        while True:
            ticker_data = await self.ticker_queue.get()
            await handle_ticker_update(ticker_data, bot_data.application)
            self.ticker_queue.task_done()

async def close_trade(trade, reason, close_price, context):
    symbol, trade_id = trade['symbol'], trade['id']
    bot, log_ctx = context.bot, {'trade_id': trade_id}
    logger.info(f"Guardian: Closing {symbol}. Reason: {reason}", extra=log_ctx)
    max_retries = bot_data.settings.get('close_retries', 3)
    for i in range(max_retries):
        try:
            asset_to_sell = symbol.split('/')[0]
            balance = await bot_data.exchange.fetch_balance()
            available_quantity = balance.get(asset_to_sell, {}).get('free', 0.0)
            if available_quantity <= 0:
                logger.critical(f"Attempted to close #{trade_id} but no balance for {asset_to_sell}.", extra=log_ctx)
                async with aiosqlite.connect(DB_FILE) as conn:
                    await conn.execute("UPDATE trades SET status = 'closure_failed', reason = 'Zero balance', close_retries = ? WHERE id = ?", (i + 1, trade_id)); await conn.commit()
                await safe_send_message(bot, f"🚨 **فشل حرج: لا يوجد رصيد**\n"
                                              f"لا يمكن إغلاق الصفقة #{trade_id} لعدم توفر رصيد كافٍ من {asset_to_sell}.")
                return
            formatted_quantity = bot_data.exchange.amount_to_precision(symbol, available_quantity)
            await bot_data.exchange.create_market_sell_order(symbol, formatted_quantity)
            pnl = (close_price - trade['entry_price']) * trade['quantity']
            pnl_percent = (close_price / trade['entry_price'] - 1) * 100 if trade['entry_price'] > 0 else 0
            if pnl > 0 and reason == "فاشلة (SL)": reason = "تم تأمين الربح (TSL)"; emoji = "✅"
            elif pnl > 0: emoji = "✅"
            else: emoji = "🛑"
            highest_price_val = max(trade.get('highest_price', 0), close_price)
            highest_pnl_percent = ((highest_price_val - trade['entry_price']) / trade['entry_price'] * 100) if trade['entry_price'] > 0 else 0
            exit_efficiency_percent = 0
            if highest_price_val > trade['entry_price']:
                highest_pnl_usdt = (highest_price_val - trade['entry_price']) * trade['quantity']
                if highest_pnl_usdt > 0: exit_efficiency_percent = (pnl / highest_pnl_usdt) * 100
                else: exit_efficiency_percent = 0
            async with aiosqlite.connect(DB_FILE) as conn:
                await conn.execute("UPDATE trades SET status = ?, close_price = ?, pnl_usdt = ?, close_retries = 0 WHERE id = ?", (reason, close_price, pnl, trade['id'])); await conn.commit()
            await bot_data.public_ws.unsubscribe([symbol])
            start_dt = datetime.fromisoformat(trade['timestamp']); end_dt = datetime.now(EGYPT_TZ)
            duration = end_dt - start_dt
            days, rem = divmod(duration.total_seconds(), 86400); hours, rem = divmod(rem, 3600); minutes, _ = divmod(rem, 60)
            duration_str = f"{int(days)}d {int(hours)}h {int(minutes)}m" if days > 0 else f"{int(hours)}h {int(minutes)}m"
            msg = (f"{emoji} **تم إغلاق الصفقة | #{trade_id} {symbol}**\n"
                   f"**السبب:** {reason}\n"
                   f"━━━━━━━━━━━━━━━━━━\n"
                   f"**إحصائيات الأداء**\n"
                   f"**الربح/الخسارة:** `${pnl:,.2f}` ({pnl_percent:+.2f}%)\n"
                   f"**أعلى ربح مؤقت:** {highest_pnl_percent:+.2f}%\n"
                   f"**كفاءة الخروج:** {exit_efficiency_percent:.2f}%\n"
                   f"**مدة الصفقة:** {duration_str}")
            await safe_send_message(bot, msg)
            return
        except Exception as e:
            logger.warning(f"Failed to close trade #{trade_id}. Retrying... ({i + 1}/{max_retries})", exc_info=True, extra=log_ctx)
            await asyncio.sleep(5)
    logger.critical(f"CRITICAL: Failed to close trade #{trade_id} after {max_retries} retries.", extra=log_ctx)
    async with aiosqlite.connect(DB_FILE) as conn:
        await conn.execute("UPDATE trades SET status = 'closure_failed', reason = 'Max retries exceeded' WHERE id = ?", (trade_id,)); await conn.commit()
    await safe_send_message(bot, f"🚨 **فشل حرج** 🚨\nفشل إغلاق الصفقة `#{trade_id}` بعد عدة محاولات. الرجاء مراجعة المنصة يدوياً.")
    await bot_data.public_ws.unsubscribe([symbol])

# --- واجهة تليجرام المتقدمة ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["Dashboard 🖥️"], ["الإعدادات ⚙️"]]
    await update.message.reply_text("أهلاً بك في **بوت التداول V9 (النسخة المايسترو متعدد الأوضاع)**", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True), parse_mode=ParseMode.MARKDOWN)

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
        [InlineKeyboardButton("🗓️ التقرير اليومي", callback_data="db_daily_report")],
        [InlineKeyboardButton(f"{ks_status_emoji} {ks_status_text}", callback_data="kill_switch_toggle"), InlineKeyboardButton("🎼 التحكم الاستراتيجي", callback_data="db_maestro_control")],  # New
        [InlineKeyboardButton("🕵️‍♂️ تقرير التشخيص", callback_data="db_diagnostics")]
    ]
    message_text = "🖥️ **لوحة تحكم قناص Binance**\n\nاختر نوع التقرير الذي تريد عرضه:"
    if not bot_data.trading_enabled: message_text += "\n\n**تحذير: تم تفعيل مفتاح الإيقاف.**"
    target_message = update.message or update.callback_query.message
    if update.callback_query: await safe_edit_message(update.callback_query, message_text, reply_markup=InlineKeyboardMarkup(keyboard))
    else: await target_message.reply_text(message_text, parse_mode=ParseMode.MARKDOWN, reply_markup=InlineKeyboardMarkup(keyboard))

# New: Task 6 - Maestro Control Panel
async def show_maestro_control(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = bot_data.settings
    regime = bot_data.current_market_regime
    maestro_enabled = s.get('maestro_mode_enabled', True)
    emoji = "✅" if maestro_enabled else "❌"
    active_scanners_str = ' + '.join([STRATEGY_NAMES_AR.get(scanner, scanner) for scanner in s.get('active_scanners', [])])
    message = (f"🎼 **لوحة التحكم الاستراتيجي (المايسترو)**\n"
               f"━━━━━━━━━━━━━━━━━━\n"
               f"**حالة المايسترو:** {emoji} مفعل\n"
               f"**تشخيص السوق الحالي:** {regime}\n"
               f"**الاستراتيجيات النشطة:** {active_scanners_str}\n\n"
               f"**التكوين الحالي:**\n"
               f"  - **المراجع الذكي:** {'✅' if s.get('intelligent_reviewer_enabled') else '❌'}\n"
               f"  - **اقتناص الزخم:** {'✅' if s.get('momentum_scalp_mode_enabled') else '❌'}\n"
               f"  - **فلتر التوافق:** {'✅' if s.get('multi_timeframe_confluence_enabled') else '❌'}\n"
               f"  - **استراتيجية الانعكاس:** {'✅' if 'bollinger_reversal' in s.get('active_scanners', []) else '❌'}")
    keyboard = [
        [InlineKeyboardButton(f"🎼 تبديل المايسترو ({'تعطيل' if maestro_enabled else 'تفعيل'})", callback_data="maestro_toggle")],
        [InlineKeyboardButton("🔙 العودة للوحة التحكم", callback_data="back_to_dashboard")]
    ]
    await safe_edit_message(update.callback_query, message, reply_markup=InlineKeyboardMarkup(keyboard))

async def toggle_maestro(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_data.settings['maestro_mode_enabled'] = not bot_data.settings.get('maestro_mode_enabled', True)
    save_settings()
    await update.callback_query.answer(f"المايسترو {'تم تفعيله' if bot_data.settings['maestro_mode_enabled'] else 'تم تعطيله'}")
    await show_maestro_control(update, context)

# باقي الوظائف كما هي، مع إضافة الإعدادات الجديدة في show_parameters_menu
async def show_parameters_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s = bot_data.settings
    def bool_format(key, text):
        val = s.get(key, False)
        emoji = "✅" if val else "❌"
        return f"{text}: {emoji} مفعل"
    def get_nested_value(d, keys):
        current_level = d
        for key in keys:
            if isinstance(current_level, dict) and key in current_level: current_level = current_level[key]
            else: return None
        return current_level
    keyboard = [
        [InlineKeyboardButton("--- إعدادات عامة ---", callback_data="noop")],
        [InlineKeyboardButton(f"عدد العملات للفحص: {s['top_n_symbols_by_volume']}", callback_data="param_set_top_n_symbols_by_volume"),
         InlineKeyboardButton(f"أقصى عدد للصفقات: {s['max_concurrent_trades']}", callback_data="param_set_max_concurrent_trades")],
        [InlineKeyboardButton(f"عمال الفحص المتزامنين: {s['worker_threads']}", callback_data="param_set_worker_threads")],
        [InlineKeyboardButton("--- إعدادات المخاطر ---", callback_data="noop")],
        [InlineKeyboardButton(f"حجم الصفقة ($): {s['real_trade_size_usdt']}", callback_data="param_set_real_trade_size_usdt"),
         InlineKeyboardButton(f"مضاعف وقف الخسارة (ATR): {s['atr_sl_multiplier']}", callback_data="param_set_atr_sl_multiplier")],
        [InlineKeyboardButton(f"نسبة المخاطرة/العائد: {s['risk_reward_ratio']}", callback_data="param_set_risk_reward_ratio")],
        [InlineKeyboardButton(bool_format('trailing_sl_enabled', 'تفعيل الوقف المتحرك'), callback_data="param_toggle_trailing_sl_enabled")],
        [InlineKeyboardButton(f"تفعيل الوقف المتحرك (%): {s['trailing_sl_activation_percent']}", callback_data="param_set_trailing_sl_activation_percent"),
         InlineKeyboardButton(f"مسافة الوقف المتحرك (%): {s['trailing_sl_callback_percent']}", callback_data="param_set_trailing_sl_callback_percent")],
        [InlineKeyboardButton(f"عدد محاولات الإغلاق: {s['close_retries']}", callback_data="param_set_close_retries")],
        [InlineKeyboardButton("--- إعدادات الإشعارات والفلترة ---", callback_data="noop")],
        [InlineKeyboardButton(bool_format('incremental_notifications_enabled', 'إشعارات الربح المتزايدة'), callback_data="param_toggle_incremental_notifications_enabled")],
        [InlineKeyboardButton(f"نسبة إشعار الربح (%): {s['incremental_notification_percent']}", callback_data="param_set_incremental_notification_percent")],
        [InlineKeyboardButton(f"مضاعف فلتر الحجم: {s['volume_filter_multiplier']}", callback_data="param_set_volume_filter_multiplier")],
        [InlineKeyboardButton(bool_format('multi_timeframe_enabled', 'فلتر الأطر الزمنية'), callback_data="param_toggle_multi_timeframe_enabled")],
        [InlineKeyboardButton(bool_format('btc_trend_filter_enabled', 'فلتر اتجاه BTC'), callback_data="param_toggle_btc_trend_filter_enabled")],
        [InlineKeyboardButton(f"فترة EMA للاتجاه: {get_nested_value(s, ['trend_filters', 'ema_period'])}", callback_data="param_set_trend_filters_ema_period")],
        [InlineKeyboardButton(f"أقصى سبريد مسموح (%): {get_nested_value(s, ['spread_filter', 'max_spread_percent'])}", callback_data="param_set_spread_filter_max_spread_percent")],
        [InlineKeyboardButton(f"أدنى ATR مسموح (%): {get_nested_value(s, ['volatility_filters', 'min_atr_percent'])}", callback_data="param_set_volatility_filters_min_atr_percent")],
        [InlineKeyboardButton(bool_format('market_mood_filter_enabled', 'فلتر الخوف والطمع'), callback_data="param_toggle_market_mood_filter_enabled"),
         InlineKeyboardButton(f"حد مؤشر الخوف: {s['fear_and_greed_threshold']}", callback_data="param_set_fear_and_greed_threshold")],
        [InlineKeyboardButton(bool_format('adx_filter_enabled', 'فلتر ADX'), callback_data="param_toggle_adx_filter_enabled"),
         InlineKeyboardButton(f"مستوى فلتر ADX: {s['adx_filter_level']}", callback_data="param_set_adx_filter_level")],
        [InlineKeyboardButton(bool_format('news_filter_enabled', 'فلتر الأخبار والبيانات'), callback_data="param_toggle_news_filter_enabled")],
        # New Settings
        [InlineKeyboardButton(bool_format('intelligent_reviewer_enabled', 'المراجع الذكي'), callback_data="param_toggle_intelligent_reviewer_enabled")],
        [InlineKeyboardButton(f"فاصل المراجع (دقائق): {s.get('intelligent_reviewer_interval_minutes', 30)}", callback_data="param_set_intelligent_reviewer_interval_minutes")],
        [InlineKeyboardButton(bool_format('momentum_scalp_mode_enabled', 'اقتناص الزخم'), callback_data="param_toggle_momentum_scalp_mode_enabled")],
        [InlineKeyboardButton(f"هدف اقتناص الزخم (%): {s.get('momentum_scalp_target_percent', 0.5)}", callback_data="param_set_momentum_scalp_target_percent")],
        [InlineKeyboardButton(bool_format('multi_timeframe_confluence_enabled', 'فلتر التوافق الزمني'), callback_data="param_toggle_multi_timeframe_confluence_enabled")],
        [InlineKeyboardButton(bool_format('maestro_mode_enabled', 'المايسترو'), callback_data="param_toggle_maestro_mode_enabled")],
        [InlineKeyboardButton("🔙 العودة للإعدادات", callback_data="settings_main")]
    ]
    await safe_edit_message(update.callback_query, "🎛️ **تعديل المعايير المتقدمة**\n\nاضغط على أي معيار لتعديل قيمته مباشرة:", reply_markup=InlineKeyboardMarkup(keyboard))

# باقي الوظائف كما هي (show_scanners_menu, show_presets_menu, etc.)، مع إضافة دعم للاستراتيجية الجديدة في show_scanners_menu

# في post_init، أضف الجدول الجديد
async def post_init(application: Application):
    logger.info("Performing post-initialization setup for Maestro Edition Bot...")
    if not all([TELEGRAM_BOT_TOKEN, BINANCE_API_KEY, BINANCE_API_SECRET]):
        logger.critical("FATAL: Missing environment variables."); return
    if NLTK_AVAILABLE:
        try: nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError: logger.info("Downloading NLTK data..."); nltk.download('vader_lexicon', quiet=True)
    bot_data.application = application
    bot_data.exchange = ccxt.binance({'apiKey': BINANCE_API_KEY, 'secret': BINANCE_API_SECRET, 'enableRateLimit': True})
    try:
        await bot_data.exchange.load_markets()
        await bot_data.exchange.fetch_balance()
        logger.info("✅ Successfully connected to Binance.")
    except Exception as e:
        logger.critical(f"🔥 FATAL: Could not connect to Binance: {e}", exc_info=True); return
    load_settings()
    await init_database()
    bot_data.public_ws = BinanceWebSocketManager()
    bot_data.public_ws.start()
    async with aiosqlite.connect(DB_FILE) as conn:
        active_symbols = [row[0] for row in await (await conn.execute("SELECT DISTINCT symbol FROM trades WHERE status = 'active'")).fetchall()]
        if active_symbols: await bot_data.public_ws.subscribe(active_symbols)
    jq = application.job_queue
    jq.run_repeating(perform_scan, interval=SCAN_INTERVAL_SECONDS, first=10, name="perform_scan")
    jq.run_repeating(the_supervisor_job, interval=SUPERVISOR_INTERVAL_SECONDS, first=30, name="the_supervisor_job")
    jq.run_repeating(intelligent_reviewer_job, interval=INTELLIGENT_REVIEWER_INTERVAL_MINUTES * 60, first=60, name="intelligent_reviewer_job")  # New
    jq.run_repeating(maestro_job, interval=MAESTRO_INTERVAL_HOURS * 3600, first=300, name="maestro_job")  # New
    jq.run_daily(send_daily_report, time=dt_time(hour=23, minute=55, tzinfo=EGYPT_TZ), name='daily_report')
    jq.run_repeating(update_strategy_performance, interval=3600, first=60, name="update_strategy_performance")
    jq.run_repeating(propose_strategy_changes, interval=3600, first=120, name="propose_strategy_changes")
    logger.info(f"Jobs scheduled. Daily report at 23:55. Strategy analysis every hour. Maestro every {MAESTRO_INTERVAL_HOURS}h.")
    try: await application.bot.send_message(TELEGRAM_CHAT_ID, "*🤖 بوت التداول V9 (النسخة المايسترو متعدد الأوضاع) - بدأ العمل...*", parse_mode=ParseMode.MARKDOWN)
    except Forbidden: logger.critical(f"FATAL: Bot not authorized for chat ID {TELEGRAM_CHAT_ID}."); return
    logger.info("--- Binance Maestro Edition Bot V9 is now fully operational ---")

async def post_shutdown(application: Application):
    if bot_data.exchange: await bot_data.exchange.close()
    if bot_data.public_ws and bot_data.public_ws.ws:
        await bot_data.public_ws.ws.close()
    logger.info("Bot has shut down.")

def main():
    logger.info("--- Starting Binance Maestro Edition Bot V9 ---")
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
