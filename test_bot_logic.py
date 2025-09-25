# test_trading_logic.py (النسخة النهائية الكاملة 100%)

import pytest
import pytest_asyncio
import asyncio
import aiosqlite
from unittest.mock import AsyncMock

# --- استيراد المكونات التي سنختبرها من ملف البوت الرئيسي ---
from BN import calculate_sl_tp, TradeGuardian, bot_data, close_trade

# =======================================================================================
# --- الجزء الأول: اختبارات الوحدات (Unit Tests) ---
# =======================================================================================

def test_calculate_sl_tp_normal_case():
    entry_price = 100.0
    atr = 2.0
    settings = {"atr_sl_multiplier": 2.5, "risk_reward_ratio": 2.0}
    stop_loss, take_profit = calculate_sl_tp(entry_price, atr, settings)
    assert stop_loss == 95.0
    assert take_profit == 110.0

def test_calculate_sl_tp_zero_atr():
    entry_price = 50.0
    atr = 0.0
    settings = {"atr_sl_multiplier": 3.0, "risk_reward_ratio": 1.5}
    stop_loss, take_profit = calculate_sl_tp(entry_price, atr, settings)
    assert stop_loss == 50.0
    assert take_profit == 50.0

# =======================================================================================
# --- الجزء الثاني: اختبارات التكامل (Integration Tests) ---
# =======================================================================================

@pytest_asyncio.fixture
async def setup_test_environment(mocker, tmp_path):
    test_db_path = tmp_path / "test_db.sqlite"
    
    async with aiosqlite.connect(test_db_path) as db:
        await db.execute('''
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY, symbol TEXT, entry_price REAL, take_profit REAL, stop_loss REAL, 
                quantity REAL, status TEXT, order_id TEXT, highest_price REAL, trailing_sl_active BOOLEAN, 
                last_profit_notification_price REAL, pnl_usdt REAL, close_price REAL, reason TEXT, 
                signal_strength INTEGER, trade_weight REAL, close_retries INTEGER
            )
        ''')
        await db.commit()
    
    mock_exchange = AsyncMock()
    # محاكاة رد وهمي بسعر إغلاق الصفقة
    mock_exchange.fetch_order.return_value = {'average': 48990.0, 'filled': 0.1}

    mocker.patch('BN.DB_FILE', str(test_db_path))
    bot_data.exchange = mock_exchange
    
    bot_data.settings = {
        "trailing_sl_enabled": True,
        "trailing_sl_activation_percent": 1.5,
        "trailing_sl_callback_percent": 1.0,
        "incremental_notifications_enabled": True,
        "incremental_notification_percent": 2.0,
        "risk_reward_ratio": 2.0
    }
    
    yield test_db_path, mock_exchange

@pytest.mark.asyncio
async def test_stop_loss_trigger_scenario(setup_test_environment):
    test_db_path, mock_exchange = setup_test_environment
    
    async with aiosqlite.connect(test_db_path) as db:
        trade_details = ('BTC/USDT', 50000.0, 52000.0, 49000.0, 0.1, 'active', '123', 50000.0, False, 50000.0)
        await db.execute("INSERT INTO trades (symbol, entry_price, take_profit, stop_loss, quantity, status, order_id, highest_price, trailing_sl_active, last_profit_notification_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", trade_details)
        await db.commit()
    
    # [الإصلاح] إنشاء تطبيق تليجرام وهمي لمنع الخطأ
    mock_application = AsyncMock()
    guardian = TradeGuardian(application=mock_application)

    trigger_price = 48990.0
    await guardian.check_trade_conditions('BTC/USDT', trigger_price)

    mock_exchange.create_market_sell_order.assert_called_once_with('BTC/USDT', 0.1)
    
    async with aiosqlite.connect(test_db_path) as db:
        res = await (await db.execute("SELECT status FROM trades WHERE order_id = '123'")).fetchone()
        assert 'فاشلة' in res[0]

# استبدل دالة اختبار الوقف المتحرك القديمة بهذه النسخة النهائية
@pytest.mark.asyncio
async def test_trailing_stop_loss_scenario(setup_test_environment):
    """
    سيناريو متكامل: اختبار الوقف المتحرك (رفع الهدف).
    """
    test_db_path, mock_exchange = setup_test_environment
    
    # --- الإعداد ---
    # [تعديل] تعطيل إشعارات الربح مؤقتاً للتركيز على اختبار الوقف المتحرك فقط
    bot_data.settings['incremental_notifications_enabled'] = False
    
    async with aiosqlite.connect(test_db_path) as db:
        trade_details = ('ETH/USDT', 3000.0, 3200.0, 2950.0, 1.0, 'active', '456', 3000.0, False, 3000.0)
        await db.execute("INSERT INTO trades (symbol, entry_price, take_profit, stop_loss, quantity, status, order_id, highest_price, trailing_sl_active, last_profit_notification_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", trade_details)
        await db.commit()
    
    mock_application = AsyncMock()
    guardian = TradeGuardian(application=mock_application)

    # --- التنفيذ 1: محاكاة ارتفاع السعر ---
    activation_price = 3061.0
    await guardian.check_trade_conditions('ETH/USDT', activation_price)
    
    # --- التحقق 1: نتأكد من أن وقف الخسارة ارتفع ---
    async with aiosqlite.connect(test_db_path) as db:
        res = await (await db.execute("SELECT stop_loss FROM trades WHERE order_id = '456'")).fetchone()
        # التأكد من أن القيمة قريبة جداً من المتوقع لتجنب أخطاء التقريب
        assert abs(res[0] - 3030.39) < 0.001

    # --- التنفيذ 2: محاكاة هبوط السعر ليضرب الوقف المتحرك ---
    trailing_trigger_price = 3030.0
    mock_exchange.fetch_order.return_value = {'average': trailing_trigger_price, 'filled': 1.0}
    await guardian.check_trade_conditions('ETH/USDT', trailing_trigger_price)

    # --- التحقق 2: نتأكد من أن أمر البيع تم إرساله ---
    mock_exchange.create_market_sell_order.assert_called_once_with('ETH/USDT', 1.0)
