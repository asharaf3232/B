import os
import ccxt
import json

print("--- 🟢 بدء كود الاختبار المباشر ---")

# جلب مفاتيح API من متغيرات البيئة التي يوفرها PM2
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

# التحقق المبدئي من وجود المفاتيح
if not api_key or not api_secret:
    print("--- 🔴 خطأ فادح: مفاتيح API غير موجودة في بيئة التشغيل! ---")
    exit()

print("--- ✅ تم العثور على مفاتيح API. جارٍ محاولة الاتصال... ---")

try:
    # إعداد الاتصال بباينانس باستخدام مكتبة ccxt
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
    })

    # جلب الأرصدة
    balance = exchange.fetch_balance()

    print("\n--- 🎉 نجح الاتصال! جاري عرض الأرصدة التي لها قيمة ---")
    
    # طباعة الأرصدة التي قيمتها أكبر من صفر فقط
    for currency, data in balance['total'].items():
        if data > 0:
            print(f"- {currency}: {data}")

except ccxt.AuthenticationError:
    print("\n--- 🔴 خطأ في المصادقة (AuthenticationError)! ---")
    print("   - تأكد من صحة مفاتيح API.")
    print("   - تأكد من أن صلاحيات المفتاح على موقع باينانس تسمح بالقراءة.")
    print("   - تأكد من عدم وجود قيود IP تمنع الخادم.")

except Exception as e:
    print(f"\n--- 🔴 حدث خطأ غير متوقع: {e} ---")

print("\n--- 끄 انتهاء كود الاختبار 끄 ---")
