# test_trading_logic.py (النسخة النهائية مع إصلاح مشكلة قاعدة البيانات)

import pytest
import pytest_asyncio
import asyncio
import aiosqlite
from unittest.mock import AsyncMock

# --- استيراد المكونات التي سنختبرها من ملف البوت الرئيسي ---
from BN import calculate_sl_tp, TradeGuardian, bot_data, close_trade

# =======================================================================================
# --- الجزء الأول: اختبارات الوحدات (Unit Tests) - (تعمل بشكل سليم) ---
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
    """
    [تم التعديل] هذه الدالة تقوم بإعداد "مختبر" نظيف باستخدام ملف قاعدة بيانات مؤقت.
    """
    # 1. إنشاء ملف قاعدة بيانات مؤقت ومنفصل لكل اختبار
    test_db_path = tmp_path / "test_db.sqlite"
    
    # 2. إنشاء الجدول داخل قاعدة البيانات المؤقتة
    async with aiosqlite.connect(test_db_path) as db_conn:
        await db_conn.execute('CREATE TABLE trades (id INTEGER PRIMARY KEY, symbol TEXT, entry_price REAL, take_profit REAL, stop_loss REAL, quantity REAL, status TEXT, order_id TEXT, highest_price REAL, trailing_sl_active BOOLEAN, last_profit_notification_price REAL)')
        await db_conn.commit()
    
    # 3. إنشاء محاكي وهمي للمنصة
    mock_exchange = AsyncMock()
    
    # 4. "خداع" البوت ليستخدم مكوناتنا الوهمية
    mocker.patch('BN.DB_FILE', str(test_db_path)) # اجعل الكود يستخدم ملفنا المؤقت
    bot_data.exchange = mock_exchange
    
    # "تسليم" المسار والمحاكي لدالة الاختبار
    yield test_db_path, mock_exchange
    
    # pytest سيقوم بحذف المجلد المؤقت وملف قاعدة البيانات تلقائياً


@pytest.mark.asyncio
async def test_stop_loss_trigger_scenario(setup_test_environment):
    """
    سيناريو متكامل: اختبار إغلاق الصفقة عند ضرب وقف الخسارة.
    """
    test_db_path, mock_exchange = setup_test_environment
    
    # الإعداد: إضافة صفقة وهمية إلى قاعدة البيانات المؤقتة
    async with aiosqlite.connect(test_db_path) as db_conn:
        trade_details = ('BTC/USDT', 50000.0, 52000.0, 49000.0, 0.1, 'active', '123', 50000.0, False, 50000.0)
        cursor = await db_conn.execute("INSERT INTO trades (symbol, entry_price, take_profit, stop_loss, quantity, status, order_id, highest_price, trailing_sl_active, last_profit_notification_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", trade_details)
        await db_conn.commit()
        trade_id = cursor.lastrowid
    
    guardian = TradeGuardian(application=None)

    # التنفيذ: محاكاة وصول سعر أقل من وقف الخسارة
    trigger_price = 48990.0
    await guardian.check_trade_conditions('BTC/USDT', trigger_price)

    # التحقق
    mock_exchange.create_market_sell_order.assert_called_once_with('BTC/USDT', 0.1)
    async with aiosqlite.connect(test_db_path) as db_conn:
        res = await (await db_conn.execute("SELECT status FROM trades WHERE id = ?", (trade_id,))).fetchone()
        assert 'فاشلة' in res[0]


@pytest.mark.asyncio
async def test_trailing_stop_loss_scenario(setup_test_environment):
    """
    سيناريو متكامل: اختبار الوقف المتحرك (رفع الهدف).
    """
    test_db_path, mock_exchange = setup_test_environment
    bot_data.settings['trailing_sl_enabled'] = True
    bot_data.settings['trailing_sl_activation_percent'] = 2.0
    bot_data.settings['trailing_sl_callback_percent'] = 1.0
    
    async with aiosqlite.connect(test_db_path) as db_conn:
        trade_details = ('ETH/USDT', 3000.0, 3200.0, 2950.0, 1.0, 'active', '456', 3000.0, False, 3000.0)
        cursor = await db_conn.execute("INSERT INTO trades (symbol, entry_price, take_profit, stop_loss, quantity, status, order_id, highest_price, trailing_sl_active, last_profit_notification_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", trade_details)
        await db_conn.commit()
        trade_id = cursor.lastrowid
    
    guardian = TradeGuardian(application=None)

    # التنفيذ 1: محاكاة ارتفاع السعر
    activation_price = 3061.0
    await guardian.check_trade_conditions('ETH/USDT', activation_price)
    
    # التحقق 1: نتأكد من أن وقف الخسارة ارتفع
    async with aiosqlite.connect(test_db_path) as db_conn:
        res = await (await db_conn.execute("SELECT stop_loss FROM trades WHERE id = ?", (trade_id,))).fetchone()
        assert res[0] == 3030.39

    # التنفيذ 2: محاكاة هبوط السعر ليضرب الوقف المتحرك
    trailing_trigger_price = 3030.0
    await guardian.check_trade_conditions('ETH/USDT', trailing_trigger_price)

    # التحقق 2: نتأكد من أن أمر البيع تم إرساله
    mock_exchange.create_market_sell_order.assert_called_once_with('ETH/USDT', 1.0)
