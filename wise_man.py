import logging
import aiosqlite
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt
from telegram.ext import Application
from collections import defaultdict
import asyncio

# --- [تحسين] نقل الإعدادات لتكون قابلة للتعديل وواضحة ---
logger = logging.getLogger(__name__)
DB_FILE = 'trading_bot_v6.6_binance.db' 

# قواعد إدارة مخاطر المحفظة
PORTFOLIO_RISK_RULES = {
    "max_asset_concentration_pct": 30.0,
    "max_sector_concentration_pct": 50.0,
}

# قاموس تصنيف العملات حسب القطاع
SECTOR_MAP = {
    'RNDR': 'AI', 'FET': 'AI', 'AGIX': 'AI',
    'UNI': 'DeFi', 'AAVE': 'DeFi', 'LDO': 'DeFi',
    'SOL': 'Layer 1', 'ETH': 'Layer 1', 'AVAX': 'Layer 1', 'BTC': 'Layer 1',
    'DOGE': 'Memecoin', 'PEPE': 'Memecoin', 'SHIB': 'Memecoin',
    'LINK': 'Oracle', 'BAND': 'Oracle',
    # أضف المزيد من العملات وتصنيفاتها هنا
}

class WiseMan:
    def __init__(self, exchange: ccxt.Exchange, application: Application, bot_data: object):
        self.exchange = exchange
        self.application = application
        self.bot_data = bot_data
        self.telegram_chat_id = application.bot_data.get('TELEGRAM_CHAT_ID')
        logger.info("🧠 Wise Man module initialized as an On-Demand Tactical Advisor.")

    async def send_telegram_message(self, text):
        try:
            if self.application and self.application.bot:
                await self.application.bot.send_message(self.telegram_chat_id, text, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Wise Man failed to send Telegram message: {e}")

    async def perform_deep_analysis(self, trade: dict):
        symbol = trade['symbol']
        trade_id = trade['id']
        logger.info(f"🧠 Wise Man summoned for deep analysis of trade #{trade_id} [{symbol}]...")

        try:
            async with aiosqlite.connect(DB_FILE) as conn:
                ohlcv_task = self.exchange.fetch_ohlcv(symbol, '15m', limit=100)
                btc_ohlcv_task = self.exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)
                ohlcv, btc_ohlcv = await asyncio.gather(ohlcv_task, btc_ohlcv_task)

                if not ohlcv:
                    logger.warning(f"Wise Man Analysis Canceled: Could not fetch OHLCV for {symbol}.")
                    return

                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # --- [تحسين] إضافة فحص لطول البيانات لتجنب الأخطاء ---
                if len(df) < 50:
                    logger.warning(f"Wise Man Analysis Canceled: Not enough data for {symbol} (got {len(df)} candles).")
                    return

                # حساب المؤشرات الفنية
                df['ema_fast'] = ta.ema(df['close'], length=10)
                df['ema_slow'] = ta.ema(df['close'], length=30)
                
                # --- [تحسين] إضافة مؤشري ADX و RSI لتحليل أعمق ---
                adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
                if adx_data is not None and not adx_data.empty:
                    df['adx'] = adx_data['ADX_14']
                else:
                    df['adx'] = 0

                df['rsi'] = ta.rsi(df['close'], length=14)
                
                # استخلاص آخر قيمة للمؤشرات
                last_row = df.iloc[-1]
                is_below_emas = last_row['close'] < last_row['ema_fast'] and last_row['close'] < last_row['ema_slow']
                is_strong_downtrend = last_row['adx'] > 25
                is_not_oversold = last_row['rsi'] > 30 # تجنب الخروج إذا كانت في منطقة تشبع بيعي

                # تحليل البيتكوين
                btc_is_bearish = False
                if btc_ohlcv:
                    btc_df = pd.DataFrame(btc_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    if len(btc_df) > 10:
                        btc_df['btc_momentum'] = ta.mom(btc_df['close'], length=10)
                        btc_is_bearish = btc_df['btc_momentum'].iloc[-1] < 0
                
                # --- [تحسين] تحديث منطق القرار ليشمل المؤشرات الجديدة ---
                # القرار الآن يعتمد على ضعف السعر + وجود اتجاه هابط قوي + ضعف عام في السوق (البيتكوين)
                is_critical_weakness = is_below_emas and is_strong_downtrend and btc_is_bearish
                
                logger.info(f"Analysis for {symbol}: BelowEMAs={is_below_emas}, StrongDowntrend(ADX>25)={is_strong_downtrend}, NotOversold(RSI>30)={is_not_oversold}, BtcBearish={btc_is_bearish}")

                if is_critical_weakness and is_not_oversold:
                    settings = self.bot_data.settings
                    if settings.get("wise_man_auto_close", True):
                        await conn.execute("UPDATE trades SET status = 'force_exit' WHERE id = ?", (trade_id,))
                        await self.send_telegram_message(f"🧠 **إغلاق آلي فوري** | `#{trade_id} {symbol}`\nأظهر التحليل العميق ضعفاً حاداً ومؤكداً.")
                    else:
                        await self.send_telegram_message(f"💡 **تحذير تكتيكي** | `#{trade_id} {symbol}`\nرصد ضعف حاد ومؤكد. يُنصح بالخروج اليدوي.")
                    await conn.commit()
                else:
                    logger.info(f"Wise Man Deep Analysis Concluded: No critical weakness found for {symbol}.")

        except Exception as e:
            logger.error(f"Wise Man: Deep analysis failed for trade #{trade_id}: {e}", exc_info=True)

    async def review_portfolio_risk(self, context: object = None):
        """
        تقوم هذه الدالة بفحص المحفظة ككل وإعطاء تنبيهات حول التركيز.
        """
        logger.info("🧠 Wise Man: Starting periodic portfolio risk review...")
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
            
            asset_values = {}
            total_assets_value = 0.0
            for asset, amount in assets.items():
                symbol = f"{asset}/USDT"
                if symbol in tickers and tickers[symbol] and tickers[symbol]['last'] is not None:
                    value_usdt = amount * tickers[symbol]['last']
                    if value_usdt > 1.0: # تجاهل الأرصدة الصغيرة جداً
                        asset_values[asset] = value_usdt
                        total_assets_value += value_usdt
            
            total_portfolio_value = usdt_total + total_assets_value
            if total_portfolio_value < 1.0: return

            # فحص تركيز الأصول الفردية
            for asset, value in asset_values.items():
                concentration_pct = (value / total_portfolio_value) * 100
                if concentration_pct > PORTFOLIO_RISK_RULES['max_asset_concentration_pct']:
                    message = (f"⚠️ **تنبيه مخاطر** | `تركيز الأصول`\n"
                               f"عملة `{asset}` تشكل **{concentration_pct:.1f}%** من محفظتك، "
                               f"متجاوزة الحد المسموح به ({PORTFOLIO_RISK_RULES['max_asset_concentration_pct']}%).")
                    await self.send_telegram_message(message)

            # فحص تركيز القطاعات
            sector_values = defaultdict(float)
            for asset, value in asset_values.items():
                sector = SECTOR_MAP.get(asset, 'Other')
                sector_values[sector] += value
            
            for sector, value in sector_values.items():
                concentration_pct = (value / total_portfolio_value) * 100
                if concentration_pct > PORTFOLIO_RISK_RULES['max_sector_concentration_pct']:
                    message = (f"⚠️ **تنبيه مخاطر** | `تركيز القطاعات`\n"
                               f"قطاع **'{sector}'** يشكل **{concentration_pct:.1f}%** من محفظتك، "
                               f"مما يعرضك لتقلباته (الحد: {PORTFOLIO_RISK_RULES['max_sector_concentration_pct']}%).")
                    await self.send_telegram_message(message)

        except Exception as e:
            logger.error(f"Wise Man: Error during portfolio risk review: {e}", exc_info=True)

        logger.info("🧠 Wise Man: Portfolio risk review complete.")
