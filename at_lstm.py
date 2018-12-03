from __future__ import unicode_literals, print_function, division
from io import open
import numpy as np
import unicodedata
import string
import random
import bcolz
import pickle

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.optim as optim		

def pad_array(X, filename):
	vocab = pickle.load(open(filename, 'rb'))
	pad_token = vocab['<PAD>']	
	#print(len(vocab))
	X = [[vocab[word] for word in sent] for sent in X]

	inp_size = len(X)
	X_lengths = np.ones(inp_size)
	for i in range(inp_size):
		X_lengths[i] = len(X[i])

	longest_entry = int(max(X_lengths))
	padded_X = np.ones((inp_size, longest_entry))*pad_token

	for i, x_len in enumerate(X_lengths):
		sequence = X[i]
		idx = int(x_len)
		padded_X[i, 0:idx] = sequence[:int(x_len)]

	return padded_X, X_lengths, pad_token

def form_input_data():
	input = pickle.load(open('train_input.pkl', 'rb'))
	
	input_size = len(input)
	Y = np.zeros((input_size, 3))
	X, aspect = [], []

	for i, item in enumerate(input):
		X.append(item[0])
		aspect.append(item[1])
		Y[i][item[2]]=1 # yi = [neutral, pos, neg]

	X, X_lengths, p_idx_sent = pad_array(X, "train_data_word2idx.pkl")
	aspect, aspect_lengths, p_idx_aspect = pad_array(aspect, "train_data_aspect2idx.pkl")

	arridx = X_lengths.argsort()
	X = X[arridx[::-1]]
	X_lengths=X_lengths[arridx[::-1]]
	aspect = aspect[arridx[::-1]]
	aspect_lengths = aspect_lengths[arridx[::-1]]
	Y = Y[arridx[::-1]]

	return X, X_lengths, p_idx_sent, aspect, aspect_lengths, p_idx_aspect, Y
		
class atae_LSTM(nn.Module):
	# num_embeddings - train data word dict len
	# taking aspect_embedding_dim = word_embedding_dim
	def __init__(self, w2vfile, a2vfile, num_class, max_sent_len, dropout_prob, p_idx_sent, p_idx_aspect, batch_size):
		super(atae_LSTM, self).__init__()
		self.batch_size = batch_size
		self.p_idx_sent = p_idx_sent
		self.p_idx_aspect = p_idx_aspect		
		self.max_sent_len = max_sent_len
		self.w2vfile = w2vfile
		self.a2vfile = a2vfile
		self.dropout_prob = dropout_prob
		self.num_class = num_class

		self.create_embed_layer()

		self.lstm = nn.LSTM(input_size = 2*self.embedding_dim, hidden_size = self.embedding_dim, num_layers=1, bias=True, batch_first=True)
		self.dropout = nn.Dropout(self.dropout_prob)
		self.attn1 = nn.Linear(self.embedding_dim, self.embedding_dim)
		self.attn2 = nn.Linear(self.embedding_dim, self.embedding_dim)
		self.attn = nn.Linear(2*self.embedding_dim, 1)
		self.softmax = nn.Softmax(dim=0)
		
		self.sent_rep1 = nn.Linear(self.embedding_dim, self.embedding_dim)		
		self.sent_rep2 = nn.Linear(self.embedding_dim, self.embedding_dim)
		self.lin_out = nn.Linear(self.embedding_dim, self.num_class, bias=True)
		self.softmax_ = nn.Softmax(dim=2)

	def create_embed_layer(self, trainable = True):
		w2v = torch.from_numpy(bcolz.open(self.w2vfile)[:])
		a2v = torch.from_numpy(bcolz.open(self.a2vfile)[:])

		self.num_embeddings = w2v.shape[0]
		self.num_aspect_embed = a2v.shape[0]
		self.embedding_dim = w2v.shape[1]

		self.embedding = nn.Embedding(num_embeddings = self.num_embeddings, embedding_dim = self.embedding_dim, padding_idx=self.p_idx_sent)
		self.aspect_embedding = nn.Embedding(num_embeddings = self.num_aspect_embed, embedding_dim = self.embedding_dim, padding_idx=self.p_idx_aspect)
		
		self.embedding.load_state_dict({'weight': w2v})
		self.aspect_embedding.load_state_dict({'weight': a2v})

		if not trainable:
			self.embedding.weight.requires_grad = False
			self.embedding.weight.requires_grad = False

	def forward(self, X, X_lengths, aspect, aspect_lengths):
		word_emb = self.embedding(X) #bxNxd
		asp_emb = self.embedding(aspect) #b x asp_len x d
		asp_emb = torch.sum(asp_emb, dim=1)
		
		asp_lens = aspect_lengths.tolist()

		for i in range(self.batch_size):
			asp_emb[i] = asp_emb[i]/asp_lens[i]

		embedded = torch.zeros(self.batch_size, self.max_sent_len, 2*self.embedding_dim)		
		for i in range(self.batch_size):
			for j in range(X_lengths[i]):
				embedded[i][j] = torch.cat([word_emb[i][j],asp_emb[i]])
		
		embedded = nn.utils.rnn.pack_padded_sequence(embedded, X_lengths, batch_first=True)
		# embedded = self.dropout(embedded)

		output, (hidden, memory) = self.lstm(embedded)	
		output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True) # output: (batch, seq_len, hidden_size)

		return output, hidden, asp_emb

	#aspect - aspect embeddings
	def output_gen(self, H, aspect, hidden, X_lengths):
		print(H.shape, self.max_sent_len * self.batch_size * self.embedding_dim)
		M_1 = self.attn1(H).view(self.max_sent_len, self.batch_size, self.embedding_dim)	# M_1: (N, batch, hidden_size)
		# print(M_1.shape)
		aspect = aspect.view(self.batch_size,1,-1)	# aspect: (batch, 1, hidden_size)

		M_2_cell = self.attn2(aspect)	# M_2_cell: (batch, 1, hidden_size)
		M_2 = M_2_cell.view(self.batch_size, self.embedding_dim) # M_2_cell: (batch, hidden_size)
		M_2 = M_2.expand(self.max_sent_len, self.batch_size, self.embedding_dim) # M_2: (N, batch, hidden_size)

		for i in range(self.batch_size):
			for j in range(X_lengths[i],self.max_sent_len):
				M_2[j][i] = torch.zeros(self.embedding_dim)					
		
		#M_1 ->Nxbxd, M_2->Nxbxd
		M = torch.tanh(torch.cat([M_1, M_2], dim=2))	#Nxbx2d
		#print(M.shape)
		attn_weights = self.softmax(self.attn(M)).permute(1,0,2)	#bxNx1
		#print(attn_weights.shape)
		H = H.permute(0,2,1)	#bxdxN
		#print(H.shape)
		sent_rep = torch.zeros(self.batch_size, self.embedding_dim, 1)	#bxdx1
		#print(sent_rep.shape)
		for i in range(self.batch_size):
			sent_rep[i] = torch.mm(H[i],attn_weights[i])
			#print(sent_rep[i].shape)
		#print(sent_rep.shape)
		sent_rep = sent_rep.permute(0,2,1)	#bx1xd
		#print(sent_rep.shape)
		#print(hidden.permute(1,0,2).shape)
		sent_rep = torch.tanh(torch.add(self.sent_rep1(sent_rep), self.sent_rep2(hidden.permute(1,0,2)))) 
		#print(sent_rep.shape)
		output = self.softmax_(self.lin_out(sent_rep))	#bx1x3
		#print(output.shape)
		output = output.view(self.batch_size, self.num_class)	#bx3 
		return output

#inp - bxNx2d, target - bx1x3
def train(X, X_lengths, aspect, aspect_lengths, target_tensor, lstm_, optimizer, criterion):#=MAX_LENGTH):	
	optimizer.zero_grad()
	loss = 0

	output, hidden, asp_emb = lstm_.forward(X, X_lengths, aspect, aspect_lengths)
	output_ = lstm_.output_gen(output, asp_emb, hidden, X_lengths)

	# loss += criterion(output_, target_tensor)
	# loss.backward()
	# optimizer.step()
	return loss

def trainIters(n_iters, batch_size, print_every=100, learning_rate=0.01):
    
	X, X_lengths, p_idx_sent, aspect, aspect_lengths, p_idx_aspect, Y = form_input_data()
	lstm_ = atae_LSTM(w2vfile="word_vec.dat", a2vfile="aspect_vec.dat", num_class=3, max_sent_len=int(X_lengths[0]), dropout_prob=0.01, p_idx_sent=p_idx_sent, p_idx_aspect=p_idx_aspect, batch_size=25)
	X, X_lengths, aspect, aspect_lengths = np.asarray(X, dtype=int), np.asarray(X_lengths, dtype=int), np.asarray(aspect, dtype=int), np.asarray(aspect_lengths, dtype=int)
	X, X_lengths, aspect, aspect_lengths = torch.from_numpy(X), torch.from_numpy(X_lengths), torch.from_numpy(aspect), torch.from_numpy(aspect_lengths)
	Y = torch.from_numpy(Y).float()
	
	loss_total = 0  # Reset every print_every
	optimizer = optim.Adagrad(lstm_.parameters(), lr=learning_rate, weight_decay=0.001)
	criterion = nn.CrossEntropyLoss()

	for i in range(n_iters):
		for j in range(0, 26, batch_size): #X.shape[0]
			if(j+25<X.shape[0]):
				loss_total += train(X[j:j+batch_size], X_lengths[j:j+batch_size], aspect[j:j+25], aspect_lengths[j:j+25], Y[j:j+25], lstm_, optimizer, criterion)

if __name__ == "__main__":
	trainIters(n_iters=5, batch_size=25)