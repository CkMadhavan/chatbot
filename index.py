import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from flask import Flask

from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

keras.backend.clear_session()
    
t_inp = pickle.load(open('t_inp-best' , "rb"))
t_oup = pickle.load(open('t_oup-best' , "rb"))

Xlen = 24
Ylen = 53

Xvocab = len(t_inp.word_index) + 1
Yvocab = len(t_oup.word_index) + 1

model = keras.models.load_model('chat-best.h5')

def prediction(model , inp_que , inp = ''):

  que = pad_sequences(t_inp.texts_to_sequences([inp_que]) , maxlen = Xlen , padding='pre' , truncating = 'pre')
  if inp == '':
    text = 'startseq'
  else:
    text = 'startseq ' + inp
  for i in range(50):
    ans = pad_sequences(t_oup.texts_to_sequences([text]) , maxlen = Ylen , padding='pre' , truncating = 'pre')
    y_pred = t_oup.sequences_to_texts([[np.argmax(model.predict([que.reshape(1,Xlen) , ans.reshape(1,Ylen)]))]])[0]

    text += ' ' + y_pred

    if y_pred == 'endseq':
      break

  return text

def clean_doc(docs , l=True):

  cleaned = []

  for line in docs:
    
    line = ''.join([x if x in string.printable else '' for x in line])
    line = line.lower()

    if l:
      line = line.replace('.' , ' ')

    line = re.sub(r"i'm", "i am", line)
    line = re.sub(r"'s", " is", line)
    line = re.sub(r"\'ll", " will", line)
    line = re.sub(r"\'ve", " have", line)
    line = re.sub(r"\'re", " are", line)
    line = re.sub(r"\'d", " would", line)
    line = re.sub(r"\'re", " are", line)
    line = re.sub(r"won't", "will not", line)
    line = re.sub(r"don't", "do not", line)
    line = re.sub(r"shouldn't", "should not", line)
    line = re.sub(r"mustn't", "must not", line)
    line = re.sub(r"can't", "cannot", line)
    line = re.sub(r"n't", " not", line)
    line = re.sub(r"n'", "ng", line)
    line = re.sub(r"'bout", "about", line)
    line = re.sub(r"'til", "until", line)

    for i in string.punctuation:
      line = line.replace(i,'')

    line = ' '.join([word for word in line.split() if word.isalpha()])
    line = 'startseq ' + line + ' endseq'

    cleaned.append(line)

  return cleaned
    
@app.route('/<process>')
@cross_origin()
def index(process):
    
    process = clean_doc([process])
    return prediction(process[0])

if __name__ == "__main__":
    app.run()
