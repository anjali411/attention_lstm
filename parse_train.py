import bcolz
import string
import numpy as np
import pickle

lines = open("rest_2014_lstm_train_new.txt", 'r').readlines()

train_input = []
train_data_word2idx = {}
train_data_aspect2idx = {}
word_count = 0
aspect_count = 0
neg = 0
pos = 0
train_data_wordvec = bcolz.carray(np.zeros(1), rootdir = 'word_vec.dat', mode='w')
train_data_aspectvec = bcolz.carray(np.zeros(1), rootdir = 'aspect_vec.dat', mode='w')


vectors = bcolz.open('6B.50.dat')[:]
word2idx = pickle.load(open('6B.50_idx.pkl', 'rb'))

existing_words = 0
existing_aspect_words = 0

non_neut_e, non_neut_a=0, 0
turn = 0

for i in range(0, len(lines), 3):
	sentence, aspect, polarity = lines[i].strip(), lines[i + 1].strip(), lines[i + 2].strip()
	sentence = sentence.replace("$T$", aspect)
	words = sentence.split()
	aspect_words = aspect.split()
	
	for word in words:
		if word not in train_data_word2idx.keys():
			if word in word2idx.keys():
				vec = vectors[word2idx[word]]
				if(int(polarity) != 0):
					existing_words+=1
			else:
				vec = np.random.uniform(low=-1, high = 1, size = (50,))

			train_data_word2idx[word] = word_count
			word_count+=1
			if(int(polarity) != 0):
				non_neut_e+=1
			#train_data_wordvec = np.concatenate((train_data_wordvec,vec))
			train_data_wordvec.append(vec)
			# print(word, word_count, "\n")

	for word in aspect_words:
		if word not in train_data_aspect2idx.keys():
			if word in word2idx.keys():
				vec_ = vectors[word2idx[word]]
				if(int(polarity) != 0):
					existing_aspect_words+=1
			else:
				print("aspect word: ", word)
				vec_ = np.random.uniform(low=-1, high = 1, size = (50,))

			train_data_aspect2idx[word] = aspect_count
			aspect_count+=1
			if(int(polarity) != 0):
				non_neut_a+=1
			#train_data_aspectvec = np.concatenate((train_data_aspectvec,vec_))
			train_data_aspectvec.append(vec_)
	
	#train_input.append([words, aspect_words, int(polarity)])
	if(int(polarity) != 0):
		if(int(polarity) == 1):
			if(turn == 1):
				train_input.append([words, aspect_words, int(polarity)])
				pos+=1
			turn = 1-turn
		elif(int(polarity) == -1):
			train_input.append([words, aspect_words, int(polarity)])
			neg+=1

print("pos: ", pos)
print("neg: ", neg)

vec_ = np.random.uniform(low=-1, high = 1, size = (50,))
train_data_word2idx['<UKW>'] = word_count
train_data_aspect2idx['<UKW>'] = aspect_count
train_data_aspectvec.append(vec_)
train_data_wordvec.append(vec_)
#train_data_wordvec = np.concatenate((train_data_wordvec,vec))
#train_data_aspectvec = np.concatenate((train_data_aspectvec,vec_))

vec_ = np.zeros(50)
train_data_aspectvec.append(vec_)
train_data_wordvec.append(vec_)
train_data_word2idx['<PAD>'] = word_count+1
train_data_aspect2idx['<PAD>'] = aspect_count+1
#train_data_wordvec = np.concatenate((train_data_wordvec,vec))
#train_data_aspectvec = np.concatenate((train_data_aspectvec,vec_))

word_count+=2
aspect_count+=2

print("Percentage of existing words: ",existing_words/non_neut_e, "\nPercentage of existing aspect words: ",existing_aspect_words/non_neut_a)

train_data_wordvec = bcolz.carray(train_data_wordvec[1:].reshape((word_count, 50)), rootdir = '50_word_vec.dat', mode='w')
train_data_wordvec.flush()
train_data_aspectvec = bcolz.carray(train_data_aspectvec[1:].reshape((aspect_count, 50)), rootdir = '50_aspect_vec.dat', mode='w')
train_data_aspectvec.flush()

# train_data_wordvec = np.reshape(train_data_wordvec[1:],(idx, 300))
# pickle.dump(train_data_wordvec, open("word_vec.pkl", 'wb'))
# train_data_aspectvec = np.reshape(train_data_aspectvec[1:],(idx, 300))
# pickle.dump(train_data_aspectvec, open("aspect_vec.pkl", 'wb'))

pickle.dump(train_data_word2idx, open('50_train_data_word2idx.pkl', 'wb'))
pickle.dump(train_data_aspect2idx, open('50_train_data_aspect2idx.pkl', 'wb'))
pickle.dump(train_input, open('without_neut_train_input.pkl', 'wb'))


#73%, 71%...
#89%, ...