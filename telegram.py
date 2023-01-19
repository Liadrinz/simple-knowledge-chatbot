import telepot
import time
from telepot.loop import MessageLoop
from chatbot import Chatbot


TOKEN = open("TOKEN.txt", "r").readline().strip()


def handle(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type == "text":
        query = msg["text"]
        response = chatbot.chat(query)
        bot.sendMessage(chat_id, response)


if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.chat("test")
    bot = telepot.Bot(TOKEN)
    MessageLoop(bot, handle).run_as_thread()
    print("Telegram Bot running...")
    while True:
        time.sleep(10)
