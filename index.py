import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from flask import Flask

app = Flask(__name__)
    
@app.route('/<process>')
def index(process):
    keras.backend.clear_session()
    
    t_inp = pickle.load(open('t_inp' , "rb"))
    t_oup = pickle.load(open('t_oup' , "rb"))

    Xlen = 24
    Ylen = 53

    Xvocab = len(t_inp.word_index) + 1
    Yvocab = len(t_oup.word_index) + 1

    model = keras.models.load_model('nmt-updated-data.h5')

    process = process.lower()
    process = process.replace('?' , ' ')
    process = process.replace(',' , ' ')
    process = process.replace('&' , ' ')
    process = process.replace('!' , ' ')
    process = process.replace('@' , ' ')
    process = process.replace('#' , ' ')
    process = process.replace('$' , ' ')
    process = process.replace('%' , ' ')
    process = process.replace('^' , ' ')
    process = process.replace('*' , ' ')
    process = process.replace('(' , ' ')
    process = process.replace(')' , ' ')
    process = process.replace('.' , ' ')
    process = process.replace('<' , ' ')
    process = process.replace('>' , ' ')
    process = process.replace(';' , ' ')
    process = process.replace(':' , ' ')
    
    string = ["startseq " + process +" endseq"]
    pred_seq = t_inp.texts_to_sequences(string)
    pred = pad_sequences(pred_seq , maxlen=Xlen,padding="post")

    string1 = ["startseq"]

    while string1[0][-6:] != "endseq":

      pred_seq1 = t_oup.texts_to_sequences(string1)
      pred1 = pad_sequences(pred_seq1 , maxlen=Ylen,padding="post")

      prediction = model.predict([pred[0].reshape(1,Xlen) , pred1[0].reshape(1,Ylen)])
      string1[0] += ' ' + list(t_oup.word_index.keys())[list(t_oup.word_index.values()).index(np.argmax(prediction[0]))]

    return ("     " + string1[0][9:-7])

if __name__ == "__main__":
    app.run()
