# test_trading_logic.py (النسخة الكاملة والمعدلة)

import pytest
import pytest_asyncio # <-- تم إضافة هذا السطر
import asyncio
import aiosqlite
from unittest.mock import AsyncMock

# --- استيراد المكونات التي سنختبرها من ملف البوت الرئيسي ---
from BN import calculate_sl_tp, TradeGuardian, bot_data, close_trade

# =======================================================================================
# --- الجزء الأول: اختبارات الوحدات (Unit Tests) ---
# =======================================================================================

def test_calculate_sl_tp_normal_case():
    """
    اختبار دالة حساب وقف الخسارة والهدف في الحالة الطبيعية.
    """
    entry_price = 100.0
    atr = 2.0
    settings = {"atr_sl_multiplier": 2.5, "risk_reward_ratio": 2.0}
    stop_loss, take_profit = calculate_sl_tp(entry_price, atr, settings)
    assert stop_loss == 95.0
    assert take_profit == 110.0

def test_calculate_sl_tp_zero_atr():
    """
    اختبار حالة خاصة: ماذا لو كان مؤشر ATR يساوي صفراً.
    """
    entry_price = 50.0
    atr = 0.0
    settings = {"atr_sl_multiplier": 3.0, "risk_reward_ratio": 1.5}
    stop_loss, take_profit = calculate_sl_tp(entry_price, atr, settings)
    assert stop_loss == 50.0
    assert take_profit == 50.0

# =======================================================================================
# --- الجزء الثاني: اختبارات التكامل (Integration Tests) ---
# =======================================================================================

@pytest_asyncio.fixture # <-- تم تعديل هذا السطر
async def setup_test_environment(mocker):
    """
    هذه الدالة الخاصة تقوم بإعداد "مختبر" نظيف ومعزول لكل اختبار تكاملي.
    """
    db_conn = await aiosqlite.connect(":memory:")
    await db_conn.execute('CREATE TABLE trades (id INTEGER PRIMARY KEY, symbol TEXT, entry_price REAL, take_profit REAL, stop_loss REAL, quantity REAL, status TEXT, order_id TEXT, highest_price REAL, trailing_sl_active BOOLEAN, last_profit_notification_price REAL)')
    
    mock_exchange = AsyncMock()
    
    mocker.patch('BN.DB_FILE', ":memory:")
    bot_data.exchange = mock_exchange
    
    yield db_conn, mock_exchange
    
    await db_conn.close()


@pytest.mark.asyncio
async def test_stop_loss_trigger_scenario(setup_test_environment):
    """
    سيناريو متكامل: اختبار إغلاق الصفقة عند ضرب وقف الخسارة.
    """
    db_conn, mock_exchange = setup_test_environment
    trade_details = ('BTC/USDT', 50000.0, 52000.0, 49000.0, 0.1, 'active', '123', 50000.0, False, 50000.0)
    cursor = await db_conn.execute("INSERT INTO trades (symbol, entry_price, take_profit, stop_loss, quantity, status, order_id, highest_price, trailing_sl_active, last_profit_notification_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", trade_details)
    await db_conn.commit()
    trade_id = cursor.lastrowid
    
    guardian = TradeGuardian(application=None)

    trigger_price = 48990.0
    await guardian.check_trade_conditions('BTC/USDT', trigger_price)

    mock_exchange.create_market_sell_order.assert_called_once_with('BTC/USDT', 0.1)
    res = await (await db_conn.execute("SELECT status FROM trades WHERE id = ?", (trade_id,))).fetchone()
    assert 'فاشلة' in res[0]


@pytest.mark.asyncio
async def test_trailing_stop_loss_scenario(setup_test_environment):
    """
    سيناريو متكامل: اختبار الوقف المتحرك (رفع الهدف).
    """
    db_conn, mock_exchange = setup_test_environment
    bot_data.settings['trailing_sl_enabled'] = True
    bot_data.settings['trailing_sl_activation_percent'] = 2.0
    bot_data.settings['trailing_sl_callback_percent'] = 1.0

    trade_details = ('ETH/USDT', 3000.0, 3200.0, 2950.0, 1.0, 'active', '456', 3000.0, False, 3000.0)
    cursor = await db_conn.execute("INSERT INTO trades (symbol, entry_price, take_profit, stop_loss, quantity, status, order_id, highest_price, trailing_sl_active, last_profit_notification_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", trade_details)
    await db_conn.commit()
    trade_id = cursor.lastrowid
    
    guardian = TradeGuardian(application=None)

    activation_price = 3061.0
    await guardian.check_trade_conditions('ETH/USDT', activation_price)
    
    res = await (await db_conn.execute("SELECT stop_loss FROM trades WHERE id = ?", (trade_id,))).fetchone()
    assert res[0] == 3030.39

    trailing_trigger_price = 3030.0
    await guardian.check_trade_conditions('ETH/USDT', trailing_trigger_price)

    mock_exchange.create_market_sell_order.assert_called_once_with('ETH/USDT', 1.0)
