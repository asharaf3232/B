# -*- coding: utf-8 -*-
# ===================================================================
# --- 🤖 بوت باينانس المبدئي - الإصدار 1.0 🤖 ---
# ===================================================================
#
#  هذا هو كود مبسط وموثوق للبدء.
#  المهمة: الاتصال بتليجرام وباينانس وعرض الرصيد.
#  يعتمد على مكتبة ccxt التي نجحت في اختبارنا السابق.
#
# ===================================================================

import os
import logging
import asyncio
import ccxt.async_support as ccxt
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.constants import ParseMode

# --- إعدادات أساسية ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- جلب المتغيرات من بيئة التشغيل (PM2) ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# --- وظائف الأوامر ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """يرسل رسالة ترحيبية عند إرسال أمر /start."""
    keyboard = [["💰 عرض الرصيد"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    welcome_message = (
        "أهلاً بك في بوت باينانس المبدئي!\n\n"
        "استخدم الزر أدناه لعرض رصيدك على منصة باينانس."
    )
    await update.message.reply_text(welcome_message, reply_markup=reply_markup)

async def show_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """يتصل بباينانس ويجلب الرصيد."""
    await update.message.reply_text("⏳ جارٍ جلب الرصيد من باينانس، يرجى الانتظار...")

    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        await update.message.reply_text("🔴 خطأ: مفاتيح API الخاصة بباينانس غير موجودة.")
        return

    try:
        # إعداد الاتصال بباينانس
        exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'enableRateLimit': True,
        })

        # جلب الأرصدة
        balance = await exchange.fetch_balance()
        
        # إغلاق الاتصال
        await exchange.close()

        message_parts = ["**💰 رصيد محفظتك على باينانس:**\n"]
        assets_found = False

        # فلترة وعرض الأرصدة التي لها قيمة فقط
        for currency, amount in balance['total'].items():
            if amount > 0:
                assets_found = True
                # تنسيق الأرقام الصغيرة جدًا بشكل أفضل
                if amount < 0.0001:
                    message_parts.append(f"- `{currency}`: `{amount:.8f}`")
                else:
                    message_parts.append(f"- `{currency}`: `{amount}`")
        
        if not assets_found:
             final_message = "لا توجد أصول ذات قيمة في محفظتك."
        else:
            final_message = "\n".join(message_parts)

    except ccxt.AuthenticationError:
        final_message = (
            "**🔴 خطأ في المصادقة!**\n\n"
            "فشل الاتصال بباينانس. يرجى التحقق من:\n"
            "1.  صحة مفاتيح API.\n"
            "2.  صلاحيات المفتاح (يجب تفعيل القراءة).\n"
            "3.  عدم وجود قيود IP تمنع الخادم."
        )
    except Exception as e:
        logger.error(f"حدث خطأ غير متوقع عند جلب الرصيد: {e}", exc_info=True)
        final_message = f"🔴 حدث خطأ غير متوقع: `{e}`"

    await update.message.reply_text(final_message, parse_mode=ParseMode.MARKDOWN)

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """يتعامل مع الرسائل النصية (الأزرار)."""
    if update.message.text == "💰 عرض الرصيد":
        await show_balance(update, context)

# --- دالة التشغيل الرئيسية ---
def main():
    """تبدأ تشغيل البوت."""
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("خطأ فادح: متغير TELEGRAM_BOT_TOKEN غير موجود!")
        return

    logger.info("Starting bot...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # إضافة معالجات الأوامر
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    
    # بدء تشغيل البوت
    application.run_polling()
    logger.info("Bot has stopped.")

if __name__ == '__main__':
    main()
