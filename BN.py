import os

print("--- 🧪 بدء اختبار متغيرات البيئة 🧪 ---")

# جلب المتغيرات المطلوبة
token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

# طباعة قيم المتغيرات
print(f"قيمة TELEGRAM_BOT_TOKEN: {token}")
print(f"قيمة TELEGRAM_CHAT_ID: {chat_id}")
print(f"قيمة BINANCE_API_KEY: {api_key}")
print(f"قيمة BINANCE_API_SECRET: {api_secret}")

# التحقق من وجود التوكن
if token:
    print("\n✅ نجاح: تم العثور على توكن تليجرام!")
else:
    print("\n❌ فشل: لم يتم العثور على توكن تليجرام.")

print("--- 끄 انتهاء الاختبار 끄 ---")
