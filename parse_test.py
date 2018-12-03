from __future__ import unicode_literals, print_function, division
from io import open
import numpy as np
import unicodedata
import pickle

lines = open("rest_2014_lstm_test_new.txt", 'r').readlines()
test_input = []
vocab_words = pickle.load(open("train_data_word2idx.pkl", 'rb'))
vocab_aspect = pickle.load(open("train_data_aspect2idx.pkl", 'rb'))

for i in range(0, len(lines), 3):
	sentence, aspect, polarity = lines[i].strip(), lines[i + 1].strip(), lines[i + 2].strip()
	sentence = sentence.replace("$T$", aspect)
	words = sentence.split()
	aspect_words = aspect.split()
	
	for idx, word in enumerate(words):
		if word not in vocab_words.keys():
			words[idx]="<UKW>"

	for idx, word in enumerate(aspect_words):
		if word not in vocab_aspect.keys():
			aspect_words[idx]="<UKW>"		

	test_input.append([words, aspect_words, int(polarity)])

pickle.dump(test_input, open('test_input.pkl', 'wb'))