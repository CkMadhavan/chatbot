import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Embedding , Dense , LSTM , TimeDistributed , RepeatVector , Dropout , Input
from tensorflow.keras.models import Sequential
import pickle
from flask import Flask

app = Flask(__name__)
    
@app.route('/<process>')
def index(process):
    
    tf.keras.backend.clear_session()
    
    t1 = pickle.load(open('t1' , "rb"))
    t2 = pickle.load(open('t2' , "rb"))

    Xlen = 15
    Ylen = 15

    Xvocab = len(t1.word_index) + 1
    Yvocab = len(t2.word_index) + 1

    n_units = 256
    
    model = tf.keras.models.Sequential()

    model.add(Embedding(Xvocab, n_units, input_length=Xlen, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(Dropout(0.2))
    model.add(RepeatVector(Ylen))
    model.add(LSTM(n_units , return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(Yvocab, activation='softmax')))

    model.compile(loss = "categorical_crossentropy" , optimizer = "adam")

    model.load_weights('chat_1700.h5')
    
    process = process.lower()

    def pred(eng):
        eng = eng.replace('?' , ' ')
        eng.strip()
        if eng != 'what_is_your_name' and eng != 'who_is_your_name' and eng != 'what_is_you' and eng != 'who_is_you' and eng != 'who_are_you':
            if eng != 'oh_ok' and eng != 'ok_then' and eng != 'okok' and eng != 'ok' and eng != 'ok_ok' and eng != 'done' and eng != 'ok_done': 
                if eng != 'hi' and eng != 'hey_you' and eng!= 'hi_there' and eng!= 'hey_there' and eng!= 'hey':
                    reverse_word_map = dict(map(reversed,  t2.word_index.items()))
                    Z = t1.texts_to_sequences([eng])
                    Z = tf.keras.preprocessing.sequence.pad_sequences(Z , maxlen = Xlen , padding = 'post')
                    Z = np.array(Z)
                    Z = Z.reshape(-1 , Xlen)

                    p = model.predict(Z)[0]

                    translated = []

                    for i in p:
                        if np.argmax(i) != 0:
                            translated.append(reverse_word_map[np.argmax(i)])

                    ger = (' ').join(translated)
                    return ger
                else:
                    return "hello buddy"
            else :
                return "good then"
        else :
            return "i am saberbot"
        
    
    return pred(process)

if __name__ == "__main__":
    app.run()
