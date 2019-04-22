import config

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

import numpy as np
from bpemb import BPEmb
from tensorflow.contrib import predictor

class Model():
    '''класс модели
        производит загрузку модели и расстановку пунктуации в предложении    
    ''' 
    def __init__(self, export_dir, vocab_size = 3000, emb_dim = 20, dict_punct = {1:2922, 2:2921, 3:2978, 4:2985, 5:2947, 6:2963}):
    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.bpemb_ru = BPEmb(lang='ru', vs=vocab_size , dim=emb_dim)

    self.export_dir = export_dir
    self.predict_fn = predictor.from_saved_model(export_dir)

    self.d = dict_punct

    def parse_fn(self, line, bpemb = bpemb_ru):
        '''
            функция кодировки строки:
            line- строка
        '''
        feature = np.array([bpemb.encode_ids(line)]).astype(np.int32)
        return feature, np.array([len(feature[0])])

    def to_capital_latter(self, sentence):
        '''фукция, переводящая прописные буквы в заглавные после точки'''
        tmp = ''
        flag = True
        for c in sentence:
            if flag and c != ' ':
                tmp += c.upper()
                flag = False
            else:
                tmp += c
            if c in '.?!':
                flag = True
        return tmp

    def predict(self, line):
        x, x_len = self.parse_fn(line)
        predict = self.predict_fn({'x':x, 'len':x_len })
        a = []
        for i in range(predict['lengths'][0]):
            a.append(predict['sequences'][0][i])
            if predict['prediction'][0][i] != 0:
                a.append(self.d[predict['prediction'][0][i]])
        return self.to_capital_latter(self.bpemb_ru.decode_ids(np.array(a)))

def startCommand(bot, update):
    print(update.effective_user)
    bot.send_message(chat_id = update.message.chat_id, 
                    text='Привет, {first_name}!'.format(first_name = update.effective_user.first_name))
    bot.send_message(chat_id = update.message.chat_id, 
                    text=' Я бот, расставляющий пунктуацию в предложении, для проверки пришли мне текст.')

def textMessage(bot, update, model):
    response = model.predict(update.message.text)
    bot.send_message(chat_id=update.message.chat_id, text=response)


def main()
    model = Model('model/1555944853')

    updater = Updater(token=config.token)
    dispatcher = updater.dispatcher

    start_command_handler = CommandHandler('start', startCommand)
    text_message_handler = MessageHandler(Filters.text, lambda x,y: textMessage(x, y, model))

    dispatcher.add_handler(start_command_handler)
    dispatcher.add_handler(text_message_handler)

    updater.start_polling(clean=True)

    updater.idle()


if __name__ == '__main__':
    main()
