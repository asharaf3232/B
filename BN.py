import asyncio
import os
import ccxt.async_support as ccxt_async

async def main():
    print("--- 🧪 Starting Binance Connection Test 🧪 ---")

    # 1. قراءة المفاتيح من متغيرات البيئة
    api_key = os.getenv('BINANCE_API_KEY')
    secret_key = os.getenv('BINANCE_API_SECRET')

    if not api_key or not secret_key:
        print("❌ FAILURE: لم يتم العثور على API Key أو Secret Key.")
        return

    print(f"✅ تم العثور على API Key، يبدأ بـ: {api_key[:5]}...")
    print(f"✅ تم العثور على Secret Key، يبدأ بـ: {secret_key[:5]}...")
    print("--------------------------------------------------")
    print("جاري محاولة الاتصال ببينانس وجلب الرصيد...")

    # 2. إنشاء اتصال ببينانس
    exchange = ccxt_async.binance({
        'apiKey': api_key,
        'secret': secret_key,
        'enableRateLimit': True,
    })

    # 3. محاولة جلب الرصيد (هذا يتطلب مفاتيح صالحة)
    try:
        balance = await exchange.fetch_balance()
        print("\n✅✅✅ SUCCESS! ✅✅✅")
        print("تم الاتصال بنجاح وجلب الرصيد.")
        if 'USDT' in balance['total']:
             print(f"رصيد USDT الإجمالي: {balance['total']['USDT']}")

    except Exception as e:
        print("\n❌❌❌ FAILURE! ❌❌❌")
        print("حدث خطأ أثناء محاولة الاتصال:")
        print(f"نوع الخطأ: {type(e).__name__}")
        print(f"تفاصيل الخطأ: {e}")

    finally:
        # 4. إغلاق الاتصال
        await exchange.close()
        print("--------------------------------------------------")
        print("--- انتهى الاختبار. ---")

if __name__ == '__main__':
    asyncio.run(main())
