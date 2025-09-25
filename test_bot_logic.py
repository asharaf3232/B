# test_bot_logic.py

# استيراد الدالة التي نريد اختبارها من ملف البوت الرئيسي
from BN import calculate_sl_tp

def test_calculate_sl_tp_normal_case():
    # 1. الإعداد (Arrange): نجهز المدخلات
    entry_price = 100.0
    atr = 2.0
    settings = {"atr_sl_multiplier": 2.5, "risk_reward_ratio": 2.0}

    # 2. التنفيذ (Act): نستدعي الدالة
    stop_loss, take_profit = calculate_sl_tp(entry_price, atr, settings)

    # 3. التحقق (Assert): نتأكد من أن النتائج صحيحة
    expected_sl = 95.0
    expected_tp = 110.0
    assert stop_loss == expected_sl
    assert take_profit == expected_tp

def test_calculate_sl_tp_zero_atr():
    # حالة أخرى: ماذا لو كان الـ ATR صفراً؟
    entry_price = 50.0
    atr = 0.0
    settings = {"atr_sl_multiplier": 3.0, "risk_reward_ratio": 1.5}

    stop_loss, take_profit = calculate_sl_tp(entry_price, atr, settings)

    # يجب أن يكون وقف الخسارة والهدف هو نفس سعر الدخول
    assert stop_loss == 50.0
    assert take_profit == 50.0
