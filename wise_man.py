import logging
import aiosqlite
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt
from telegram.ext import Application
from collections import defaultdict
import asyncio

# لا يوجد أي استيراد من ملف BN.py هنا

# --- إعدادات أساسية ---
logger = logging.getLogger(__name__)
DB_FILE = 'trading_bot_v6.6_binance.db' 

# --- قواعد إدارة مخاطر المحفظة ---
PORTFOLIO_RISK_RULES = {
    "max_asset_concentration_pct": 30.0,
    "max_sector_concentration_pct": 50.0,
}

# --- قاموس تصنيف العملات حسب القطاع ---
SECTOR_MAP = {
    'RNDR': 'AI', 'FET': 'AI', 'AGIX': 'AI',
    'UNI': 'DeFi', 'AAVE': 'DeFi', 'LDO': 'DeFi',
    'SOL': 'Layer 1', 'ETH': 'Layer 1', 'AVAX': 'Layer 1',
    'DOGE': 'Memecoin', 'PEPE': 'Memecoin', 'SHIB': 'Memecoin',
    'LINK': 'Oracle', 'BAND': 'Oracle',
    'BTC': 'Layer 1'
}

class WiseMan:
    def __init__(self, exchange: ccxt.Exchange, application: Application, bot_data: object):
        self.exchange = exchange
        self.application = application
        self.bot_data = bot_data
        self.telegram_chat_id = application.bot_data.get('TELEGRAM_CHAT_ID')
        logger.info("🧠 Wise Man module initialized.")

    async def send_telegram_message(self, text):
        """دالة مساعدة لإرسال رسائل تليجرام بشكل آمن."""
        try:
            if self.application and self.application.bot:
                await self.application.bot.send_message(self.telegram_chat_id, text)
        except Exception as e:
            logger.error(f"Wise Man failed to send Telegram message: {e}")

    async def review_open_trades(self, context: object = None):
        logger.info("🧠 Wise Man: Starting periodic review of open trades...")
        async with aiosqlite.connect(DB_FILE) as conn:
            conn.row_factory = aiosqlite.Row
            active_trades = await (await conn.execute("SELECT * FROM trades WHERE status = 'active'")).fetchall()

            if not active_trades:
                logger.info("🧠 Wise Man: No active trades to review. Review cycle finished.")
                return

            logger.info(f"🧠 Wise Man: Found {len(active_trades)} active trades to analyze.")

            try:
                btc_ohlcv = await self.exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)
                btc_df = pd.DataFrame(btc_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                btc_df['btc_momentum'] = ta.mom(btc_df['close'], length=10)
                btc_is_bearish = btc_df['btc_momentum'].iloc[-1] < 0
                logger.info(f"🧠 Wise Man: BTC momentum check complete. Bearish signal: {btc_is_bearish}")
            except Exception as e:
                logger.error(f"Wise Man: Could not fetch BTC data for comparison: {e}")
                btc_df = None
                btc_is_bearish = False # نفترض أن الوضع ليس سلبيًا إذا فشلنا في جلب البيانات

            for trade_data in active_trades:
                trade = dict(trade_data)
                symbol = trade['symbol']
                trade_id = trade['id']
                logger.info(f"--- Analyzing trade #{trade_id} for {symbol} ---")
                
                try:
                    ohlcv = await self.exchange.fetch_ohlcv(symbol, '15m', limit=100)
                    if not ohlcv or len(ohlcv) < 35: # زدنا الحد الأدنى لضمان حساب المؤشرات
                        logger.warning(f"Wise Man: Skipped {symbol} due to insufficient OHLCV data ({len(ohlcv) if ohlcv else 0} candles).")
                        continue
                    
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    df['ema_fast'] = ta.ema(df['close'], length=10)
                    df['ema_slow'] = ta.ema(df['close'], length=30)
                    
                    if df['ema_fast'].isnull().iloc[-1] or df['ema_slow'].isnull().iloc[-1]:
                        logger.warning(f"Wise Man: Skipped {symbol} because EMA indicators could not be calculated (resulted in NaN).")
                        continue

                    is_weak = df['close'].iloc[-1] < df['ema_fast'].iloc[-1] and df['close'].iloc[-1] < df['ema_slow'].iloc[-1]
                    
                    # الشرط المركب: ضعف في العملة مع زخم هابط للبيتكوين
                    if is_weak and btc_is_bearish:
                        logger.warning(f"Wise Man: DETECTED WEAKNESS for {symbol}. Price is below EMAs and BTC momentum is negative.")
                        if self.bot_data.settings.get("wise_man_auto_close", True):
                            logger.warning(f"Wise Man: Auto-close is ENABLED. Flagging trade #{trade_id} with 'force_exit' for Guardian.")
                            await conn.execute("UPDATE trades SET status = 'force_exit' WHERE id = ?", (trade_id,))
                            await self.send_telegram_message(f"🧠 **إغلاق آلي | #{trade_id} {symbol}**\nرصد الرجل الحكيم ضعفًا وقام بالخروج الفوري لحماية الأرباح.")
                        else:
                            logger.info(f"Wise Man: Auto-close is DISABLED. Sending advice message for {symbol}.")
                            await self.send_telegram_message(f"💡 **نصيحة من الرجل الحكيم | #{trade_id} {symbol}**\nتم رصد ضعف. يُنصح بالخروج اليدوي من هذه الصفقة.")
                        continue # ننتقل للصفقة التالية بعد اتخاذ إجراء

                    current_profit_pct = (df['close'].iloc[-1] / trade['entry_price'] - 1) * 100 if trade['entry_price'] > 0 else 0
                    adx_data = ta.adx(df['high'], df['low'], df['close'])
                    
                    if adx_data is None or adx_data.empty:
                        logger.warning(f"Wise Man: Could not calculate ADX for {symbol}.")
                        continue
                        
                    current_adx = adx_data['ADX_14'].iloc[-1]
                    
                    is_strong = current_profit_pct > 3.0 and current_adx > 30

                    if is_strong:
                        logger.info(f"Wise Man: DETECTED STRENGTH for {symbol}. PNL > 3% and ADX > 30. Extending TP.")
                        new_tp = trade['take_profit'] * 1.05
                        await conn.execute("UPDATE trades SET take_profit = ? WHERE id = ?", (new_tp, trade_id))
                        await self.send_telegram_message(f"🧠 **نصيحة من الرجل الحكيم | #{trade_id} {symbol}**\nتم رصد زخم قوي. تم تمديد الهدف إلى ${new_tp:.4f} للسماح للأرباح بالنمو.")
                    else:
                        # رسالة أخيرة لتأكيد أن الصفقة تم فحصها وكل شيء على ما يرام
                        logger.info(f"Wise Man: Analysis for {symbol} complete. No weakness or significant strength detected. Current PNL: {current_profit_pct:.2f}%")


                except Exception as e:
                    logger.error(f"Wise Man: An unexpected error occurred while analyzing trade #{trade_id} for {symbol}: {e}")
                
                await asyncio.sleep(2) # فاصل بسيط بين تحليل كل عملة
            
            await conn.commit()
        logger.info("🧠 Wise Man: Trade review complete.")

    async def review_portfolio_risk(self, context: object = None):
        logger.info("🧠 Wise Man: Starting portfolio risk review...")
        try:
            balance = await self.exchange.fetch_balance()
            
            assets = {
                asset: data['total'] 
                for asset, data in balance.items() 
                if isinstance(data, dict) and data.get('total', 0) > 0.00001 and asset != 'USDT'
            }
            
            if not assets:
                logger.info("🧠 Wise Man: Portfolio is empty (only USDT). No risks to analyze.")
                return

            asset_list = [f"{asset}/USDT" for asset in assets.keys() if asset != 'USDT']
            if not asset_list: return

            tickers = await self.exchange.fetch_tickers(asset_list)
            
            usdt_total = balance.get('USDT', {}).get('total', 0.0)
            if not isinstance(usdt_total, float): usdt_total = 0.0
            total_portfolio_value = usdt_total

            asset_values = {}
            for asset, amount in assets.items():
                symbol = f"{asset}/USDT"
                if symbol in tickers and tickers[symbol] and tickers[symbol]['last'] is not None:
                    value_usdt = amount * tickers[symbol]['last']
                    if value_usdt > 1.0:
                        asset_values[asset] = value_usdt
                        total_portfolio_value += value_usdt

            if total_portfolio_value < 1.0: return

            for asset, value in asset_values.items():
                concentration_pct = (value / total_portfolio_value) * 100
                if concentration_pct > PORTFOLIO_RISK_RULES['max_asset_concentration_pct']:
                    message = (f"⚠️ **تنبيه من الرجل الحكيم (إدارة المخاطر):**\n"
                               f"تركيز المخاطر عالٍ! عملة `{asset}` تشكل **{concentration_pct:.1f}%** من قيمة المحفظة، "
                               f"وهو ما يتجاوز الحد المسموح به ({PORTFOLIO_RISK_RULES['max_asset_concentration_pct']}%).")
                    await self.send_telegram_message(message)

            sector_values = defaultdict(float)
            for asset, value in asset_values.items():
                sector = SECTOR_MAP.get(asset, 'Other')
                sector_values[sector] += value
            
            for sector, value in sector_values.items():
                concentration_pct = (value / total_portfolio_value) * 100
                if concentration_pct > PORTFOLIO_RISK_RULES['max_sector_concentration_pct']:
                    message = (f"⚠️ **تنبيه من الرجل الحكيم (إدارة المخاطر):**\n"
                               f"تركيز قطاعي! أصول قطاع **'{sector}'** تشكل **{concentration_pct:.1f}%** من المحفظة، "
                               f"مما يعرضك لتقلبات هذا القطاع بشكل كبير (الحد المسموح به: {PORTFOLIO_RISK_RULES['max_sector_concentration_pct']}%).")
                    await self.send_telegram_message(message)

        except Exception as e:
            logger.error(f"Wise Man: Error during portfolio risk review: {e}", exc_info=True)

        logger.info("🧠 Wise Man: Portfolio risk review complete.")
