# -*- coding: utf-8 -*-
"""
Created on Wed May 26 13:13:35 2021

@author: modon
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 26 12:37:28 2021

@author: modon
"""

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Activation
from tensorflow.keras.optimizers import RMSprop

filepath = 'ref.fna'

text = open(filepath,'rb').read().decode(encoding='utf-8').lower()

text = text[70:]

characters = sorted(set(text))

char_to_index = dict((c,i) for i, c in enumerate(characters))
index_to_char = dict((i,c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3

"""
sentences = []
next_characters = []

for i in range(0,len(text) - SEQ_LENGTH,STEP_SIZE):
    sentences.append(text[i:i+SEQ_LENGTH])
    next_characters.append(text[i+SEQ_LENGTH])
    
x = np.zeros((len(sentences),SEQ_LENGTH,len(characters)),dtype=np.bool)
y = np.zeros((len(sentences),len(characters)),dtype=np.bool)

for i,sentence in enumerate(sentences):
    for t,character in enumerate(sentence):
        x[i,t, char_to_index[character]]=1
    y[i,char_to_index[next_characters[i]]]=1
    
"""



model = tf.keras.models.load_model('seqgen.model')

# takes predicitions and takes one character
# higher the temp the more risky
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1,preds,1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH -1)
    generated = ''
    sentence = text[start_index:start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x = np.zeros((1,SEQ_LENGTH, len(characters)))
        for t , character in enumerate(sentence):
            x[0,t,char_to_index[character]] = 1
            
        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions,temperature)
        next_character = index_to_char[next_index]
        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

print("-----0.2-----")
print(generate_text(300, 0.2))
print("-----0.4-----")
print(generate_text(300, 0.4))
print("-----0.6-----")
print(generate_text(300, 0.6))
print("-----0.8-----")
print(generate_text(300, 0.8))
print("-----1.0-----")
print(generate_text(300, 1.0))   


    
    


