import bcolz
import string
import numpy as np
import pickle

lines = open("rest_2014_lstm_train_new.txt", 'r').readlines()
train_input = []
train_data_word2idx = {'<PAD>':0}
train_data_aspect2idx = {'<PAD>':0}
word_count = 0
aspect_count = 0

train_data_wordvec = bcolz.carray(np.zeros(1), rootdir = 'word_vec.dat', mode='w')
train_data_aspectvec = bcolz.carray(np.zeros(1), rootdir = 'aspect_vec.dat', mode='w')

vectors = bcolz.open('840B.300.dat')[:]
word2idx = pickle.load(open('840B.300_idx.pkl', 'rb'))
existing_words = 0
existing_aspect_words = 0

for i in range(0, len(lines), 3):
	sentence, aspect, polarity = lines[i].strip(), lines[i + 1].strip(), lines[i + 2].strip()
	sentence = sentence.replace("$T$", aspect)
	words = sentence.split()
	aspect_words = aspect.split()
	
	for word in words:
		if word not in train_data_word2idx.keys():
			if word in word2idx.keys():
				vec = vectors[word2idx[word]]
				existing_words+=1
			else:
				print("word: ", word)
				vec = np.random.uniform(low=-1, high = 1, size = (300,))

			word_count+=1
			train_data_word2idx[word] = word_count
			train_data_wordvec.append(vec)
			# print(word, word_count, "\n")

	for word in aspect_words:
		if word not in train_data_aspect2idx.keys():
			if word in word2idx.keys():
				vec_ = vectors[word2idx[word]]
				existing_aspect_words+=1
			else:
				print("aspect word: ", word)
				vec_ = np.random.uniform(low=-1, high = 1, size = (300,))

			aspect_count+=1
			train_data_aspect2idx[word] = aspect_count
			train_data_aspectvec.append(vec_)

	train_input.append([words, aspect_words, int(polarity)])

# print(train_input)
# print(train_data_word2idx)

print("Percentage of existing words: ",existing_words/word_count, "\nPercentage of existing aspect words: ",existing_aspect_words/aspect_count)

train_data_wordvec = bcolz.carray(train_data_wordvec[1:].reshape((word_count, 300)), rootdir = 'word_vec.dat', mode='w')
train_data_wordvec.flush()
train_data_aspectvec = bcolz.carray(train_data_aspectvec[1:].reshape((aspect_count, 300)), rootdir = 'aspect_vec.dat', mode='w')
train_data_aspectvec.flush()

# print(train_data_wordvec.shape, train_data_aspectvec.shape)

pickle.dump(train_data_word2idx, open('train_data_word2idx.pkl', 'wb'))
pickle.dump(train_data_aspect2idx, open('aspect_data_word2idx.pkl', 'wb'))


