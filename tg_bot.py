import os
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton, WebAppInfo

# Бот берет токен из системы (или использует локальный, если переменная не найдена)
TOKEN = os.environ.get("BOT_TOKEN", "ТВОЙ_ТОКЕН_ДЛЯ_ЛОКАЛЬНОГО_ТЕСТА")
WEB_APP_URL = "https://dota-draftapp-3.onrender.com"

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def start(message):
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    web_app = WebAppInfo(url=WEB_APP_URL)
    btn = KeyboardButton(text="🎮 Открыть Драфт-помощника", web_app=web_app)
    markup.add(btn)

    bot.send_message(
        message.chat.id, 
        "Привет! Нажми на кнопку ниже, чтобы выбрать героев:", 
        reply_markup=markup
    )

bot.infinity_polling()