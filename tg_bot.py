import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton, WebAppInfo

# Сюда вставишь токен от BotFather
TOKEN = "8841515463:AAH4qnAvLfSzJtPWW4bcWzP6ZBhw5ETcsI4"
bot = telebot.TeleBot(TOKEN)

# Сюда мы потом вставим ссылку на твой сайт, когда закинем его в интернет
WEB_APP_URL = "https://google.com" # Пока тут гугл для примера

@bot.message_handler(commands=['start'])
def start(message):
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    # Эта кнопка скажет Телеграму открыть окошко с сайтом
    web_app = WebAppInfo(url=WEB_APP_URL)
    btn = KeyboardButton(text="🎮 Открыть Драфт-помощника", web_app=web_app)
    markup.add(btn)

    bot.send_message(
        message.chat.id, 
        "Привет! Жми на кнопку внизу, чтобы выбрать героев:", 
        reply_markup=markup
    )

bot.infinity_polling()