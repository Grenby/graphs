import telebot


def send_massage(text, img=None):
    try:
        bot = telebot.TeleBot(token='6765096167:AAFKNO406kQ1ZBM-Nauf_S0y5SivOwgWyPs')

        if img is not None:
            if type(img) is str:
                img = open(img, 'rb')
            bot.send_photo(341371039, img, caption=text)
        else:
            bot.send_message(341371039, text)
    except:
        print('err on sending')

# @bot.message_handler(content_types=['text'])
# def get_text_messages(message):
#     city,id = message.text.split(' ')
#     from main import prepare_city
#     prepare_city(city, id)
#
# bot.polling(none_stop=True, interval=0)