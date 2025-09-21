# -*- coding: utf-8 -*-
# ===================================================================
# --- 🚀 بوت التداول المتقدم V2 🚀 ---
# ===================================================================
#
#  هذا الإصدار يدمج الماسحات والعقل وقدرات التداول في الأساس
#  المستقر الذي بنيناه. تم إعادة كتابة الأجزاء المعقدة لضمان
#  أعلى درجات الموثوقية.
#
#  - يعتمد 100% على مكتبة ccxt.
#  - نظام مراقبة جديد ومستقر للصفقات (Polling Supervisor).
#  - يبدأ الفحص متوقفًا كإجراء أمان.
#
# ===================================================================

# --- المكتبات الأساسية ---
import os
import logging
import asyncio
import json
import time
import copy
from datetime import datetime
from zoneinfo import ZoneInfo

# --- مكتبات التحليل والتداول ---
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt
import feedparser
import nltk

# --- مكتبات تليجرام ---
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.constants import ParseMode

# --- إعدادات أساسية ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- جلب المتغيرات من بيئة التشغيل (PM2) ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') # للأخبار (اختياري)

# --- إعدادات البوت القابلة للتعديل ---
DB_FILE = 'trading_bot_v2.db'
SETTINGS_FILE = 'trading_bot_v2_settings.json'
TIMEFRAME = '15m'
EGYPT_TZ = ZoneInfo("Africa/Cairo")

DEFAULT_SETTINGS = {
    "real_trade_size_usdt": 15.0,
    "max_concurrent_trades": 5,
    "top_n_symbols_by_volume": 150,
    "atr_sl_multiplier": 2.5,
    "risk_reward_ratio": 2.0,
    "active_scanners": ["momentum_breakout", "breakout_squeeze_pro"],
    "min_quote_volume_24h_usd": 1000000,
    "min_rvol": 1.5,
}

# --- الحالة العامة للبوت ---
class BotState:
    def __init__(self):
        self.settings = {}
        self.scanning_enabled = False # يبدأ متوقفًا
        self.exchange = None
        self.application = None

bot_data = BotState()
scan_lock = asyncio.Lock()

# --- وظائف مساعدة وقاعدة البيانات ---

def load_settings():
    """تحميل الإعدادات من ملف JSON."""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                bot_data.settings = json.load(f)
        else:
            bot_data.settings = copy.deepcopy(DEFAULT_SETTINGS)
    except Exception:
        bot_data.settings = copy.deepcopy(DEFAULT_SETTINGS)
    
    # التأكد من وجود كل الإعدادات الافتراضية
    for key, value in DEFAULT_SETTINGS.items():
        bot_data.settings.setdefault(key, value)
    
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(bot_data.settings, f, indent=4)
    logger.info("Settings loaded successfully.")

async def init_database():
    """إنشاء الجداول اللازمة في قاعدة البيانات."""
    import aiosqlite
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute(
                '''CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    entry_price REAL,
                    take_profit REAL,
                    stop_loss REAL,
                    quantity REAL,
                    status TEXT,
                    reason TEXT,
                    order_id TEXT,
                    close_price REAL,
                    pnl_usdt REAL
                )'''
            )
            await conn.commit()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.critical(f"Database initialization failed: {e}")

# --- العقل: تحليل مزاج السوق ---

def get_latest_crypto_news(limit=10):
    """جلب آخر الأخبار من مصادر موثوقة."""
    urls = ["https://cointelegraph.com/rss", "https://www.coindesk.com/arc/outboundfeeds/rss/"]
    headlines = []
    for url in urls:
        feed = feedparser.parse(url)
        headlines.extend([entry.title for entry in feed.entries[:5]])
    return list(set(headlines))[:limit]

def analyze_sentiment_of_headlines(headlines):
    """تحليل مشاعر العناوين باستخدام NLTK."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        logger.info("Downloading NLTK vader_lexicon...")
        nltk.download('vader_lexicon', quiet=True)
    
    if not headlines:
        return "محايدة", 0.0
        
    sia = SentimentIntensityAnalyzer()
    score = sum(sia.polarity_scores(h)['compound'] for h in headlines) / len(headlines)
    
    if score > 0.15:
        mood = "إيجابية"
    elif score < -0.15:
        mood = "سلبية"
    else:
        mood = "محايدة"
    return mood, score

# --- الماسحات: تحليل الأنماط الفنية ---

def find_col(df_columns, prefix):
    try:
        return next(col for col in df_columns if col.startswith(prefix))
    except StopIteration:
        return None

def analyze_momentum_breakout(df):
    """ماسح الزخم الاختراقي."""
    df.ta.bbands(length=20, append=True)
    df.ta.macd(append=True)
    df.ta.rsi(append=True)
    
    last = df.iloc[-2]
    prev = df.iloc[-3]
    
    macd_col = find_col(df.columns, "MACD_")
    macds_col = find_col(df.columns, "MACDs_")
    bbu_col = find_col(df.columns, "BBU_")
    rsi_col = find_col(df.columns, "RSI_")

    if not all([macd_col, macds_col, bbu_col, rsi_col]):
        return None
        
    if (prev[macd_col] <= prev[macds_col] and last[macd_col] > last[macds_col] and
            last['close'] > last[bbu_col] and last[rsi_col] < 68):
        return {"reason": "momentum_breakout"}
    return None

def analyze_breakout_squeeze_pro(df):
    """ماسح اختراق الانضغاط."""
    df.ta.bbands(length=20, append=True)
    df.ta.kc(length=20, scalar=1.5, append=True)
    
    bbu_col = find_col(df.columns, "BBU_")
    bbl_col = find_col(df.columns, "BBL_")
    kcu_col = find_col(df.columns, "KCUe_")
    kcl_col = find_col(df.columns, "KCLEe_")

    if not all([bbu_col, bbl_col, kcu_col, kcl_col]):
        return None

    last = df.iloc[-2]
    prev = df.iloc[-3]

    is_in_squeeze = prev[bbl_col] > prev[kcl_col] and prev[bbu_col] < prev[kcu_col]
    if (is_in_squeeze and last['close'] > last[bbu_col] and
            last['volume'] > df['volume'].rolling(20).mean().iloc[-2] * 1.5):
        return {"reason": "breakout_squeeze_pro"}
    return None

SCANNERS = {
    "momentum_breakout": analyze_momentum_breakout,
    "breakout_squeeze_pro": analyze_breakout_squeeze_pro,
}

# --- محرك التداول ---

async def worker(queue, signals_list):
    """عامل يقوم بتحليل العملات من قائمة الانتظار."""
    while not queue.empty():
        market = await queue.get()
        symbol = market['symbol']
        try:
            ohlcv = await bot_data.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=100)
            if len(ohlcv) < 50:
                continue
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # حساب RVOL
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            if pd.isna(df['volume_sma'].iloc[-2]) or df['volume_sma'].iloc[-2] == 0:
                continue
            rvol = df['volume'].iloc[-2] / df['volume_sma'].iloc[-2]
            if rvol < bot_data.settings['min_rvol']:
                continue

            confirmed_reasons = []
            for name in bot_data.settings['active_scanners']:
                if (strategy_func := SCANNERS.get(name)):
                    if result := strategy_func(df.copy()):
                        confirmed_reasons.append(result['reason'])
            
            if confirmed_reasons:
                reason_str = ' + '.join(set(confirmed_reasons))
                entry_price = df.iloc[-2]['close']
                df.ta.atr(length=14, append=True)
                atr_col = find_col(df.columns, "ATRr_14")
                if not atr_col: continue

                risk = df.iloc[-2][atr_col] * bot_data.settings['atr_sl_multiplier']
                stop_loss = entry_price - risk
                take_profit = entry_price + (risk * bot_data.settings['risk_reward_ratio'])
                
                signals_list.append({
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "reason": reason_str
                })

        except Exception as e:
            logger.debug(f"Worker error for {symbol}: {e}")
        finally:
            queue.task_done()

async def initiate_real_trade(signal):
    """تنفيذ صفقة شراء حقيقية."""
    import aiosqlite
    try:
        trade_size = bot_data.settings['real_trade_size_usdt']
        
        # تحويل المبلغ المطلوب بالدولار إلى كمية من العملة
        ticker = await bot_data.exchange.fetch_ticker(signal['symbol'])
        last_price = ticker['last']
        amount = trade_size / last_price
        
        # تنفيذ أمر الشراء
        buy_order = await bot_data.exchange.create_market_buy_order(signal['symbol'], amount)
        logger.info(f"Buy order sent for {signal['symbol']}. Order ID: {buy_order['id']}")
        
        # تسجيل الصفقة في قاعدة البيانات كـ 'active' مباشرة
        async with aiosqlite.connect(DB_FILE) as conn:
            await conn.execute(
                "INSERT INTO trades (timestamp, symbol, reason, order_id, status, entry_price, take_profit, stop_loss, quantity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(EGYPT_TZ).isoformat(),
                    signal['symbol'],
                    signal['reason'],
                    buy_order['id'],
                    'active', # مباشرة نشطة
                    buy_order.get('average', last_price), # استخدام السعر الفعلي إن وجد
                    signal['take_profit'],
                    signal['stop_loss'],
                    buy_order['filled'] # الكمية الفعلية التي تم شراؤها
                )
            )
            await conn.commit()

        await bot_data.application.bot.send_message(
            chat_id=update.message.chat_id,
            text=f"🚀 **تم فتح صفقة جديدة!**\n\n- **العملة:** `{signal['symbol']}`\n- **السبب:** `{signal['reason']}`\n- **سعر الدخول:** `${buy_order.get('average', last_price):.4f}`",
            parse_mode=ParseMode.MARKDOWN
        )
        return True

    except ccxt.InsufficientFunds:
        logger.error(f"Trade failed for {signal['symbol']}: Insufficient funds.")
        await bot_data.application.bot.send_message(chat_id=update.message.chat_id, text="⚠️ رصيد USDT غير كافٍ لفتح صفقة جديدة.")
        return False
    except Exception as e:
        logger.error(f"Trade failed for {signal['symbol']}: {e}", exc_info=True)
        return False

async def perform_scan(context: ContextTypes.DEFAULT_TYPE):
    """الدالة الرئيسية للفحص الدوري."""
    async with scan_lock:
        if not bot_data.scanning_enabled:
            logger.info("Scan skipped: Scanning is disabled.")
            return

        logger.info("--- 🚀 Starting New Market Scan... ---")
        
        try:
            # التحقق من عدد الصفقات المفتوحة
            import aiosqlite
            async with aiosqlite.connect(DB_FILE) as conn:
                cursor = await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active'")
                active_trades_count = (await cursor.fetchone())[0]

            if active_trades_count >= bot_data.settings['max_concurrent_trades']:
                logger.info(f"Scan skipped: Max concurrent trades ({active_trades_count}) reached.")
                return

            # جلب الأسواق وفلترتها
            all_tickers = await bot_data.exchange.fetch_tickers()
            valid_markets = [
                t for t in all_tickers.values() if
                'USDT' in t['symbol'] and t.get('quoteVolume', 0) > bot_data.settings['min_quote_volume_24h_usd']
            ]
            valid_markets.sort(key=lambda m: m.get('quoteVolume', 0), reverse=True)
            top_markets = valid_markets[:bot_data.settings['top_n_symbols_by_volume']]
            
            # تشغيل العمال
            queue = asyncio.Queue()
            signals_found = []
            for market in top_markets:
                await queue.put(market)
            
            worker_tasks = [asyncio.create_task(worker(queue, signals_found)) for _ in range(10)]
            await queue.join()

            # تنفيذ الصفقات
            trades_to_open = bot_data.settings['max_concurrent_trades'] - active_trades_count
            for signal in signals_found[:trades_to_open]:
                await initiate_real_trade(signal)
            
            logger.info(f"--- ✅ Scan Finished. Found {len(signals_found)} signals. ---")

        except Exception as e:
            logger.error(f"An error occurred during the scan: {e}", exc_info=True)

async def trade_supervisor_job(context: ContextTypes.DEFAULT_TYPE):
    """المشرف: يراقب الصفقات المفتوحة ويغلقها عند الحاجة."""
    import aiosqlite
    logger.info("--- 🕵️ Supervisor job starting... ---")
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM trades WHERE status = 'active'")
            active_trades = await cursor.fetchall()

        if not active_trades:
            logger.info("Supervisor: No active trades to monitor.")
            return

        symbols = [trade['symbol'] for trade in active_trades]
        tickers = await bot_data.exchange.fetch_tickers(symbols)

        for trade in active_trades:
            symbol = trade['symbol']
            current_price = tickers[symbol]['last']
            
            close_reason = None
            if current_price >= trade['take_profit']:
                close_reason = "Take Profit Hit"
            elif current_price <= trade['stop_loss']:
                close_reason = "Stop Loss Hit"

            if close_reason:
                logger.info(f"Closing trade for {symbol}. Reason: {close_reason}")
                try:
                    # تنفيذ أمر بيع
                    sell_order = await bot_data.exchange.create_market_sell_order(symbol, trade['quantity'])
                    
                    # تحديث قاعدة البيانات
                    pnl = (sell_order.get('average', current_price) - trade['entry_price']) * trade['quantity']
                    async with aiosqlite.connect(DB_FILE) as conn:
                        await conn.execute(
                            "UPDATE trades SET status = 'closed', close_price = ?, pnl_usdt = ? WHERE id = ?",
                            (sell_order.get('average', current_price), pnl, trade['id'])
                        )
                        await conn.commit()
                    
                    emoji = "✅" if pnl > 0 else "🛑"
                    await context.bot.send_message(
                        chat_id=update.message.chat_id,
                        text=f"**{emoji} تم إغلاق الصفقة**\n\n- **العملة:** `{symbol}`\n- **السبب:** `{close_reason}`\n- **الربح/الخسارة:** `${pnl:.2f}`",
                        parse_mode=ParseMode.MARKDOWN
                    )
                except Exception as e:
                    logger.error(f"Failed to close trade for {symbol}: {e}")

    except Exception as e:
        logger.error(f"An error occurred in the supervisor job: {e}", exc_info=True)


# --- أوامر تليجرام ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """يرسل رسالة ترحيبية ويعرض لوحة التحكم."""
    status = "نشط 🟢" if bot_data.scanning_enabled else "متوقف 🔴"
    keyboard = [
        ["💰 عرض الرصيد", "📊 حالة البوت"],
        [f"🔬 تشغيل/إيقاف الفحص ({status})"]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("أهلاً بك في بوت التداول المتقدم V2.", reply_markup=reply_markup)

async def show_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """يعرض رصيد الحساب."""
    await update.message.reply_text("⏳ جارٍ جلب الرصيد...")
    try:
        balance = await bot_data.exchange.fetch_balance()
        message_parts = ["**💰 رصيد محفظتك:**\n"]
        assets_found = False
        for currency, amount in balance['total'].items():
            if amount > 0.00001: # عرض الأصول ذات القيمة فقط
                assets_found = True
                message_parts.append(f"- `{currency}`: `{amount}`")
        if not assets_found:
            final_message = "لا توجد أصول ذات قيمة."
        else:
            final_message = "\n".join(message_parts)
    except Exception as e:
        logger.error(f"Error fetching balance: {e}")
        final_message = f"🔴 حدث خطأ: {e}"
    await update.message.reply_text(final_message, parse_mode=ParseMode.MARKDOWN)

async def toggle_scanning(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """تشغيل أو إيقاف عملية الفحص."""
    bot_data.scanning_enabled = not bot_data.scanning_enabled
    await start_command(update, context) # تحديث الأزرار

async def show_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض حالة البوت الحالية."""
    import aiosqlite
    scan_status = "نشط 🟢" if bot_data.scanning_enabled else "متوقف 🔴"
    
    # جلب عدد الصفقات المفتوحة
    async with aiosqlite.connect(DB_FILE) as conn:
        cursor = await conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'active'")
        active_trades = (await cursor.fetchone())[0]

    # تحليل مزاج السوق
    headlines = await asyncio.to_thread(get_latest_crypto_news)
    mood, score = analyze_sentiment_of_headlines(headlines)

    status_message = (
        f"**📊 حالة البوت الحالية:**\n\n"
        f"- **حالة الفحص:** {scan_status}\n"
        f"- **الصفقات النشطة:** `{active_trades} / {bot_data.settings['max_concurrent_trades']}`\n"
        f"- **مزاج السوق (الأخبار):** {mood} (الدرجة: {score:.2f})"
    )
    await update.message.reply_text(status_message, parse_mode=ParseMode.MARKDOWN)

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """يتعامل مع الأزرار."""
    text = update.message.text
    if "عرض الرصيد" in text:
        await show_balance(update, context)
    elif "تشغيل/إيقاف الفحص" in text:
        await toggle_scanning(update, context)
    elif "حالة البوت" in text:
        await show_status(update, context)

async def post_init(application: Application):
    """يتم تشغيله بعد إعداد البوت وقبل بدء التشغيل."""
    logger.info("Performing post-initialization setup...")
    if not all([TELEGRAM_BOT_TOKEN, BINANCE_API_KEY, BINANCE_API_SECRET]):
        logger.critical("FATAL: One or more required environment variables are missing.")
        return

    bot_data.application = application
    bot_data.exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_API_SECRET,
        'enableRateLimit': True,
    })

    load_settings()
    await init_database()

    # جدولة المهام الدورية
    job_queue = application.job_queue
    job_queue.run_repeating(perform_scan, interval=300, first=10) # كل 5 دقائق
    job_queue.run_repeating(trade_supervisor_job, interval=60, first=15) # كل دقيقة

    logger.info("Bot is fully initialized and jobs are scheduled.")

def main():
    """تبدأ تشغيل البوت."""
    logger.info("Starting bot...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    
    application.run_polling()
    
    # إغلاق الاتصال عند إيقاف البوت
    if bot_data.exchange:
        asyncio.run(bot_data.exchange.close())
    logger.info("Bot has stopped.")

if __name__ == '__main__':
    main()
