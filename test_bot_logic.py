# test_guardian_logic.py (اختبار التكامل النهائي للحارس)

import pytest
import pytest_asyncio
import asyncio
import aiosqlite
from unittest.mock import AsyncMock, patch

# ملاحظة: تم تعديل الاستيرادات لتتوافق مع بنية BN.py الحالية (V6.3)
from BN import TradeGuardian, bot_data, DB_FILE

# =======================================================================================
# --- إعداد بيئة الاختبار (Fixtures) ---
# =======================================================================================

@pytest_asyncio.fixture
async def setup_test_environment(mocker, tmp_path):
    """تهيئة بيئة وهمية مع قاعدة بيانات مؤقتة وإعدادات البوت."""
    
    # 1. إعداد مسار قاعدة بيانات مؤقت
    test_db_path = tmp_path / "test_db.sqlite"
    
    # 2. إنشاء جدول trades في قاعدة البيانات المؤقتة
    async with aiosqlite.connect(test_db_path) as db:
        await db.execute('''
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY, symbol TEXT, entry_price REAL, take_profit REAL, stop_loss REAL, 
                quantity REAL, status TEXT, order_id TEXT, highest_price REAL DEFAULT 0, trailing_sl_active BOOLEAN DEFAULT 0, 
                last_profit_notification_price REAL DEFAULT 0, pnl_usdt REAL, close_price REAL, reason TEXT, 
                signal_strength INTEGER DEFAULT 1, trade_weight REAL DEFAULT 1.0, close_retries INTEGER DEFAULT 0
            )
        ''')
        await db.commit()
    
    # 3. محاكاة CCXT و Telegram
    mock_exchange = AsyncMock()
    
    # Mock لـ safe_send_message لتجنب إرسال رسائل تيليجرام حقيقية
    mocker.patch('BN.safe_send_message', new=AsyncMock())
    
    # تعيين مسار قاعدة البيانات الوهمي وملحقات CCXT
    mocker.patch('BN.DB_FILE', str(test_db_path))
    bot_data.exchange = mock_exchange
    
    # 4. إعدادات البوت اللازمة للاختبار
    bot_data.settings = {
        "trailing_sl_enabled": True,
        "trailing_sl_activation_percent": 1.5,
        "trailing_sl_callback_percent": 1.0,
        "close_retries": 3,
        "risk_reward_ratio": 2.0,
        "incremental_notifications_enabled": False,
        "real_trade_size_usdt": 100.0
    }
    
    yield test_db_path, mock_exchange


# =======================================================================================
# --- اختبار الإغلاق عند الوقف المتحرك (Trailing SL) ---
# =======================================================================================

@pytest.mark.asyncio
async def test_trailing_stop_loss_trigger(setup_test_environment):
    test_db_path, mock_exchange = setup_test_environment
    
    # 1. إعداد صفقة BTC/USDT نشطة
    ENTRY = 50000.0
    SL = 49000.0 # 2%
    TP = 52000.0 # 4%
    QTY = 0.002
    
    async with aiosqlite.connect(test_db_path) as db:
        trade_details = ('BTC/USDT', ENTRY, TP, SL, QTY, 'active', 'ORDER1', ENTRY, False, ENTRY)
        await db.execute("INSERT INTO trades (symbol, entry_price, take_profit, stop_loss, quantity, status, order_id, highest_price, trailing_sl_active, last_profit_notification_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", trade_details)
        await db.commit()
    
    mock_application = AsyncMock()
    guardian = TradeGuardian(application=mock_application)

    # 2. محاكاة ارتفاع السعر وتفعيل الوقف المتحرك (1.5% تفعيل = 50750.0)
    ACTIVATION_PRICE = 50750.0 
    await guardian.check_trade_conditions('BTC/USDT', ACTIVATION_PRICE)
    
    # 3. محاكاة وصول السعر إلى القمة (لرفع الوقف المتحرك)
    HIGH_PRICE = 51000.0
    await guardian.check_trade_conditions('BTC/USDT', HIGH_PRICE) # القمة الجديدة
    
    # 4. تحقق من قيمة الوقف المتحرك الجديدة في قاعدة البيانات
    # الوقف الجديد = 51000.0 * (1 - 0.01) = 50490.0
    async with aiosqlite.connect(test_db_path) as db:
        res = await (await db.execute("SELECT stop_loss, highest_price FROM trades WHERE order_id = 'ORDER1'")).fetchone()
        new_stop_loss, highest_price_db = res
        
        assert highest_price_db == HIGH_PRICE
        assert abs(new_stop_loss - 50490.0) < 0.01 # يجب أن يكون الوقف قد ارتفع
        
    # 5. محاكاة هبوط السعر ليلامس الوقف المتحرك (50490.0)
    TSL_TRIGGER_PRICE = 50400.0 
    await guardian.check_trade_conditions('BTC/USDT', TSL_TRIGGER_PRICE)

    # 6. التأكد من أن أمر البيع تم استدعاؤه لغرض تأمين الربح (TSL)
    mock_exchange.create_market_sell_order.assert_called_once_with('BTC/USDT', QTY)
    
    # 7. التأكد من تحديث حالة الصفقة إلى TSL
    async with aiosqlite.connect(test_db_path) as db:
        res = await (await db.execute("SELECT status, pnl_usdt FROM trades WHERE order_id = 'ORDER1'")).fetchone()
        assert 'تم تأمين الربح (TSL)' == res[0]
        assert res[1] > 0 # يجب أن يكون الربح إيجابياً

# =======================================================================================
# --- اختبار الإغلاق عند وقف الخسارة الأصلي (SL) ---
# =======================================================================================

@pytest.mark.asyncio
async def test_hard_stop_loss_trigger(setup_test_environment):
    test_db_path, mock_exchange = setup_test_environment
    
    # 1. إعداد صفقة USDT/ETH نشطة
    ENTRY = 3000.0
    SL = 2950.0
    QTY = 1.0
    
    async with aiosqlite.connect(test_db_path) as db:
        # تعطيل الـ Trailing SL لمنع تداخله في هذا الاختبار
        trade_details = ('ETH/USDT', ENTRY, 3100.0, SL, QTY, 'active', 'ORDER2', ENTRY, False, ENTRY)
        await db.execute("INSERT INTO trades (symbol, entry_price, take_profit, stop_loss, quantity, status, order_id, highest_price, trailing_sl_active, last_profit_notification_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", trade_details)
        await db.commit()
    
    mock_application = AsyncMock()
    guardian = TradeGuardian(application=mock_application)

    # 2. محاكاة هبوط السعر ليلامس الوقف (2950.0)
    SL_TRIGGER_PRICE = 2940.0
    await guardian.check_trade_conditions('ETH/USDT', SL_TRIGGER_PRICE)

    # 3. التأكد من أن أمر البيع تم استدعاؤه
    mock_exchange.create_market_sell_order.assert_called_once_with('ETH/USDT', QTY)
    
    # 4. التأكد من تحديث حالة الصفقة إلى (فاشلة SL)
    async with aiosqlite.connect(test_db_path) as db:
        res = await (await db.execute("SELECT status, pnl_usdt FROM trades WHERE order_id = 'ORDER2'")).fetchone()
        assert 'فاشلة (SL)' == res[0]
        assert res[1] < 0 # يجب أن تكون الخسارة سلبية

# =======================================================================================
