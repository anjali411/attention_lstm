import bcolz
import string
import numpy as np
import pickle

idx=0 
word2idx = {}
word_embedding_file = 'glove.840B.300d.txt'
count=0
vectors = bcolz.carray(np.zeros(1), rootdir = '840B.300.dat', mode='w')

with open(word_embedding_file, 'rb') as file:
	for word_vec in file:
		line = word_vec.decode().split()
		#print(line)
		word = line[0]
		start_idx = len(line)-300
		
		if(start_idx>=1):
			print(idx, word, len(line))
			word2idx[word] = idx
			idx+=1
			vec = np.array(line[start_idx:]).astype(np.float)
			vectors.append(vec)
		else:
			count+=1

print("count: ", count)
vectors = bcolz.carray(vectors[1:].reshape((idx, 300)), rootdir = '840B.300.dat', mode='w')
vectors.flush()
pickle.dump(word2idx, open('840B.300_idx.pkl', 'wb'))
