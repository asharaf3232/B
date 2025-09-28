import logging
import aiosqlite
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt
from telegram.ext import Application

# --- إعدادات أساسية ---
logger = logging.getLogger(__name__)
DB_FILE = 'trading_bot_v6.6_binance.db'

class WiseMan:
    def __init__(self, exchange: ccxt.Exchange, application: Application):
        """
        يتم تهيئة الرجل الحكيم مع وصول للمنصة ولتطبيق تليجرام.
        """
        self.exchange = exchange
        self.application = application
        logger.info("🧠 Wise Man module initialized.")

    async def review_open_trades(self):
        """
        الدالة الرئيسية التي تمر على كل الصفقات المفتوحة وتطبق المنطق التكتيكي.
        """
        logger.info("🧠 Wise Man: Starting periodic review of open trades...")
        async with aiosqlite.connect(DB_FILE) as conn:
            conn.row_factory = aiosqlite.Row
            active_trades = await (await conn.execute("SELECT * FROM trades WHERE status = 'active'")).fetchall()

        if not active_trades:
            logger.info("🧠 Wise Man: No active trades to review.")
            return

        # سنقوم بجلب بيانات البيتكوين مرة واحدة لاستخدامها في المقارنة
        try:
            btc_ohlcv = await self.exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)
            btc_df = pd.DataFrame(btc_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            btc_df['btc_momentum'] = ta.mom(btc_df['close'], length=10) # حساب زخم البيتكوين
        except Exception as e:
            logger.error(f"Wise Man: Could not fetch BTC data for comparison: {e}")
            btc_df = None

        for trade_data in active_trades:
            trade = dict(trade_data)
            symbol = trade['symbol']
            try:
                # جلب أحدث البيانات للعملة التي في الصفقة
                ohlcv = await self.exchange.fetch_ohlcv(symbol, '15m', limit=100)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # --- 1. منطق "اقطع خسائرك مبكرًا" ---
                df['ema_fast'] = ta.ema(df['close'], length=10)
                df['ema_slow'] = ta.ema(df['close'], length=30)
                
                # إذا السعر أغلق تحت المتوسطين السريع والبطيء، فهذه علامة ضعف
                is_weak = df['close'].iloc[-1] < df['ema_fast'].iloc[-1] and df['close'].iloc[-1] < df['ema_slow'].iloc[-1]
                
                # إذا كانت العملة ضعيفة والبيتكوين زخمه سلبي، فالوضع سيء
                if is_weak and (btc_df is not None and btc_df['btc_momentum'].iloc[-1] < 0):
                    logger.warning(f"Wise Man recommends early exit for {symbol}. Reason: Momentum Failure & Negative BTC drift.")
                    # نحدث وقف الخسارة ليكون أعلى بقليل من السعر الحالي لنجبر الحارس على البيع
                    new_sl = df['close'].iloc[-1] * 1.002
                    await conn.execute("UPDATE trades SET stop_loss = ? WHERE id = ?", (new_sl, trade['id']))
                    await self.application.bot.send_message(self.application.bot.TELEGRAM_CHAT_ID, f"🧠 **نصيحة من الرجل الحكيم | #{trade['id']} {symbol}**\nتم رصد ضعف في الزخم. تم تضييق وقف الخسارة للخروج المبكر.")
                    continue # ننتقل للصفقة التالية

                # --- 2. منطق "دع أرباحك تنمو" ---
                # إذا كانت الصفقة رابحة بأكثر من 3% والزخم قوي جدًا
                current_profit_pct = (df['close'].iloc[-1] / trade['entry_price'] - 1) * 100
                df['adx'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
                is_strong = current_profit_pct > 3.0 and df['adx'].iloc[-1] > 30

                if is_strong:
                    # حساب الهدف الجديد، مثلاً 5% أعلى من الهدف الحالي
                    new_tp = trade['take_profit'] * 1.05
                    await conn.execute("UPDATE trades SET take_profit = ? WHERE id = ?", (new_tp, trade['id']))
                    logger.info(f"Wise Man recommends extending target for {symbol}. New TP: {new_tp}")
                    await self.application.bot.send_message(self.application.bot.TELEGRAM_CHAT_ID, f"🧠 **نصيحة من الرجل الحكيم | #{trade['id']} {symbol}**\nتم رصد زخم قوي. تم تمديد الهدف إلى ${new_tp:.4f} للسماح للأرباح بالنمو.")

            except Exception as e:
                logger.error(f"Wise Man: Failed to analyze trade #{trade['id']} for {symbol}: {e}")
        
        await conn.commit()
        logger.info("🧠 Wise Man: Review complete.")

    async def review_portfolio_risk(self):
        """
        (سيتم برمجتها لاحقًا)
        تقوم هذه الدالة بفحص المحفظة ككل وإعطاء تنبيهات حول التركيز.
        """
        logger.info("🧠 Wise Man: Portfolio risk review (Not yet implemented).")
        pass
