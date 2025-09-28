import logging
import aiosqlite
import asyncio
import json
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)
DB_FILE = 'trading_bot_v6.6_binance.db'
ANALYSIS_PERIOD_CANDLES = 24 # عدد الشموع التي سنراقبها بعد الخروج (24 شمعة * 15 دقيقة = 6 ساعات)

class EvolutionaryEngine:
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        logger.info("🧬 Evolutionary Engine Initialized. Ready to build memory.")

    async def _capture_market_snapshot(self, symbol: str) -> dict:
        """يلتقط صورة لحالة المؤشرات الفنية للسوق في لحظة معينة."""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, '15m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            rsi = ta.rsi(df['close'], length=14).iloc[-1]
            adx_data = ta.adx(df['high'], df['low'], df['close'])
            adx = adx_data['ADX_14'].iloc[-1] if adx_data is not None else None
            
            return {"rsi": round(rsi, 2), "adx": round(adx, 2) if adx is not None else None}
        except Exception as e:
            logger.error(f"Smart Engine: Could not capture market snapshot for {symbol}: {e}")
            return {}

    async def add_trade_to_journal(self, trade_details: dict):
        """
        الوظيفة الرئيسية: تسجل الصفقة المغلقة في الذاكرة وتبدأ تحليل "ماذا لو؟"
        """
        trade_id = trade_details.get('id')
        symbol = trade_details.get('symbol')
        logger.info(f"🧬 Journaling trade #{trade_id} for {symbol}...")

        try:
            snapshot = await self._capture_market_snapshot(symbol)
            
            async with aiosqlite.connect(DB_FILE) as conn:
                await conn.execute("""
                    INSERT INTO trade_journal (trade_id, entry_strategy, entry_indicators_snapshot, exit_reason)
                    VALUES (?, ?, ?, ?)
                """, (
                    trade_id,
                    trade_details.get('reason'),
                    json.dumps(snapshot),
                    trade_details.get('status')
                ))
                await conn.commit()
            
            asyncio.create_task(self._perform_what_if_analysis(trade_details))

        except Exception as e:
            logger.error(f"Smart Engine: Failed to journal trade #{trade_id}: {e}", exc_info=True)

    async def _perform_what_if_analysis(self, trade_details: dict):
        """
        تحلل سلوك العملة بعد الخروج منها لتقييم جودة القرار.
        """
        trade_id = trade_details.get('id')
        symbol = trade_details.get('symbol')
        exit_reason = trade_details.get('status', '')
        original_tp = trade_details.get('take_profit')
        original_sl = trade_details.get('stop_loss')

        # ننتظر قليلاً حتى لا نحلل نفس الشمعة التي خرجنا منها
        await asyncio.sleep(60) 
        
        logger.info(f"🔬 Smart Engine: Performing 'What-If' analysis for closed trade #{trade_id} ({symbol})...")
        
        try:
            # 1. جلب البيانات المستقبلية
            future_ohlcv = await self.exchange.fetch_ohlcv(symbol, '15m', limit=ANALYSIS_PERIOD_CANDLES)
            df_future = pd.DataFrame(future_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            highest_price_after = df_future['high'].max()
            lowest_price_after = df_future['low'].min()
            
            score = 0
            notes = ""

            # 2. منطق تقييم القرار
            if 'SL' in exit_reason: # إذا كان الخروج بسبب وقف الخسارة
                if highest_price_after >= original_tp:
                    score = -10 # قرار سيء جدًا (ندم)
                    notes = f"Stop Loss Regret: Price recovered and hit original TP ({original_tp})."
                else:
                    score = 10 # قرار ممتاز
                    notes = f"Good Save: Price continued to drop to {lowest_price_after} after SL."
            
            elif 'TP' in exit_reason: # إذا كان الخروج بسبب جني الربح
                missed_profit_pct = ((highest_price_after / original_tp) - 1) * 100 if original_tp > 0 else 0
                if missed_profit_pct > (trade_details.get('risk_reward_ratio', 2.0) * 100):
                    score = -5 # فرصة ضائعة كبيرة
                    notes = f"Missed Opportunity: Price rallied an additional {missed_profit_pct:.2f}% after TP."
                elif missed_profit_pct > 1.0:
                    score = 5 # قرار جيد
                    notes = f"Good Exit: Price rallied a little more but was a good exit point."
                else:
                    score = 10 # قرار مثالي
                    notes = f"Perfect Exit: Price dropped or stalled after hitting TP."

            post_performance_data = {
                "highest_price_after": highest_price_after,
                "lowest_price_after": lowest_price_after,
                "analysis_period_hours": (ANALYSIS_PERIOD_CANDLES * 15) / 60
            }

            # 3. تحديث قاعدة البيانات بالنتائج
            async with aiosqlite.connect(DB_FILE) as conn:
                await conn.execute("""
                    UPDATE trade_journal
                    SET exit_quality_score = ?, post_exit_performance = ?, notes = ?
                    WHERE trade_id = ?
                """, (
                    score,
                    json.dumps(post_performance_data),
                    notes,
                    trade_id
                ))
                await conn.commit()
            
            logger.info(f"🔬 Analysis complete for trade #{trade_id}. Exit Quality Score: {score}. Notes: {notes}")

        except Exception as e:
            logger.error(f"Smart Engine: 'What-If' analysis failed for trade #{trade_id}: {e}", exc_info=True)
