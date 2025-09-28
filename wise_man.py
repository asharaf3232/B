import logging
import aiosqlite
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt
from telegram.ext import Application
from collections import defaultdict
import asyncio

# --- إعدادات أساسية ---
logger = logging.getLogger(__name__)
DB_FILE = 'trading_bot_v6.6_binance.db' 

# --- قواعد إدارة مخاطر المحفظة ---
# ملاحظة: من الأفضل لاحقًا نقل هذه القواعد إلى ملف الإعدادات الرئيسي
PORTFOLIO_RISK_RULES = {
    "max_asset_concentration_pct": 30.0,  # أقصى نسبة مئوية لأي أصل واحد في المحفظة
    "max_sector_concentration_pct": 50.0, # أقصى نسبة مئوية لأي قطاع واحد في المحفظة
}

# --- قاموس تصنيف العملات حسب القطاع ---
# هذا قاموس مبدئي يمكنك توسيعه بنفسك
SECTOR_MAP = {
    'RNDR': 'AI', 'FET': 'AI', 'AGIX': 'AI',
    'UNI': 'DeFi', 'AAVE': 'DeFi', 'LDO': 'DeFi',
    'SOL': 'Layer 1', 'ETH': 'Layer 1', 'AVAX': 'Layer 1',
    'DOGE': 'Memecoin', 'PEPE': 'Memecoin', 'SHIB': 'Memecoin',
    'LINK': 'Oracle', 'BAND': 'Oracle',
    'BTC': 'Layer 1'
}

class WiseMan:
    def __init__(self, exchange: ccxt.Exchange, application: Application):
        """
        يتم تهيئة الرجل الحكيم مع وصول للمنصة ولتطبيق تليجرام.
        """
        self.exchange = exchange
        self.application = application
        # نحتاج إلى الوصول إلى chat_id لإرسال الرسائل
        self.telegram_chat_id = application.bot_data.get('TELEGRAM_CHAT_ID')
        logger.info("🧠 Wise Man module initialized.")

    async def review_open_trades(self, context: object = None):
        """
        يقدم توصيات للحارس بتغيير حالة الصفقة.
        """
        logger.info("🧠 Wise Man: Starting periodic review of open trades...")
        async with aiosqlite.connect(DB_FILE) as conn:
            conn.row_factory = aiosqlite.Row
            active_trades = await (await conn.execute("SELECT * FROM trades WHERE status = 'active'")).fetchall()

            if not active_trades:
                logger.info("🧠 Wise Man: No active trades to review.")
                return
            
            # ... (باقي كود جلب بيانات البيتكوين يبقى كما هو) ...

            for trade_data in active_trades:
                trade = dict(trade_data)
                symbol = trade['symbol']
                try:
                    # ... (باقي كود جلب بيانات العملة وحساب المؤشرات يبقى كما هو) ...
                    
                    # --- منطق "اقطع خسائرك مبكرًا" المعدل ---
                    if is_weak and (btc_df is not None and btc_df['btc_momentum'].iloc[-1] < 0):
                        logger.warning(f"Wise Man recommends early exit for {symbol}. Flagging for Guardian.")
                        # **التعديل: نغير الحالة بدلاً من تعديل وقف الخسارة**
                        await conn.execute("UPDATE trades SET status = 'force_exit' WHERE id = ?", (trade['id'],))
                        await self.application.bot.send_message(self.telegram_chat_id, f"🧠 **توصية من الرجل الحكيم | #{trade['id']} {symbol}**\nتم رصد ضعف. تم طلب الخروج المبكر من الحارس.")
                        continue

                    # --- منطق "دع أرباحك تنمو" يبقى كما هو (يعدل الهدف فقط) ---
                    if is_strong:
                        new_tp = trade['take_profit'] * 1.05
                        await conn.execute("UPDATE trades SET take_profit = ? WHERE id = ?", (new_tp, trade['id']))
                        logger.info(f"Wise Man recommends extending target for {symbol}. New TP: {new_tp}")
                        await self.application.bot.send_message(self.telegram_chat_id, f"🧠 **نصيحة من الرجل الحكيم | #{trade['id']} {symbol}**\nتم رصد زخم قوي. تم تمديد الهدف إلى ${new_tp:.4f}.")

                except Exception as e:
                    logger.error(f"Wise Man: Failed to analyze trade #{trade['id']} for {symbol}: {e}")
            
            await conn.commit()
        logger.info("🧠 Wise Man: Trade review complete.")

    async def review_portfolio_risk(self, context: object = None):
        """
        تقوم هذه الدالة بفحص المحفظة ككل وإعطاء تنبيهات حول التركيز.
        """
        logger.info("🧠 Wise Man: Starting portfolio risk review...")
        try:
            balance = await self.exchange.fetch_balance()
            assets = {asset: data['total'] for asset, data in balance.items() if data.get('total', 0) > 0.00001 and asset != 'USDT'}
            
            if not assets:
                logger.info("🧠 Wise Man: Portfolio is empty (only USDT). No risks to analyze.")
                return

            asset_list = [f"{asset}/USDT" for asset in assets.keys() if asset != 'USDT']
            if not asset_list: return

            tickers = await self.exchange.fetch_tickers(asset_list)
            
            total_portfolio_value = balance.get('USDT', {}).get('total', 0.0)
            asset_values = {}
            for asset, amount in assets.items():
                symbol = f"{asset}/USDT"
                if symbol in tickers:
                    value_usdt = amount * tickers[symbol]['last']
                    if value_usdt > 1.0: # تجاهل الأرصدة الصغيرة جدًا
                        asset_values[asset] = value_usdt
                        total_portfolio_value += value_usdt

            if total_portfolio_value < 1.0: return

            for asset, value in asset_values.items():
                concentration_pct = (value / total_portfolio_value) * 100
                if concentration_pct > PORTFOLIO_RISK_RULES['max_asset_concentration_pct']:
                    message = (f"⚠️ **تنبيه من الرجل الحكيم (إدارة المخاطر):**\n"
                               f"تركيز المخاطر عالٍ! عملة `{asset}` تشكل **{concentration_pct:.1f}%** من قيمة المحفظة، "
                               f"وهو ما يتجاوز الحد المسموح به ({PORTFOLIO_RISK_RULES['max_asset_concentration_pct']}%).")
                    await self.application.bot.send_message(self.telegram_chat_id, message)

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
                     await self.application.bot.send_message(self.telegram_chat_id, message)

        except Exception as e:
            logger.error(f"Wise Man: Error during portfolio risk review: {e}", exc_info=True)

        logger.info("🧠 Wise Man: Portfolio risk review complete.")
