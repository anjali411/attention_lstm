from __future__ import unicode_literals, print_function, division
from io import open
import numpy as np
import unicodedata
import string
import re
import random
import bcolz

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.optim as optim		

def pad_array(X, filename):
	vocab = pickle.load(open(filename, 'rb'))
	pad_token = vocab['<PAD>']

	X = [[vocab[word] for word in sent] for sent in X]
	X_lengths = [len(sent) for sent in X]
	pad_token = 0 # entry in dict corresponding to <PAD>
	longest_sent = max(X_lengths)
	batch_size = len(X)
	padded_X = np.ones((batch_size, longest_sent))*pad_token

	for i, x_len in enumerate(X_lengths):
		sequence = X[i]
		padded_X[i, 0:x_len] = sequence[:x_len]

	return padded_X

def form_input_data():
	input = pickle.load(open('train_input.pkl', 'rb'))
	for item in input:
		X.append(input[0])
		aspect.append(input[1])
		Y.append(input[2])

	X = pad_array(X, "train_data_word2idx.pkl")
	aspect = pad_array(aspect, "train_data_aspect2idx.pkl")

	return X, aspect, batch

		
class atae_LSTM(nn.Module):
	# num_embeddings - train data word dict len
	# taking aspect_embedding_dim = word_embedding_dim
	def __init__(self, w2vfile, a2vfile, num_class, max_sent_len, dropout_prob, p_idx_sent, p_idx_aspect, batch_size):
		super(attention_LSTM, self).__init__()
		create_embed_layer(w2vfile, a2vfile, p_idx_sent, p_idx_aspect)

		self.batch_size = batch_size
		self.p_idx_sent = p_idx_sent
		self.p_idx_aspect = p_idx_aspect		
		self.max_sent_len = max_sent_len

		self.lstm = nn.LSTM(2*self.embedding_dim, self.embedding_dim, bias=True, batch_first=True)
		self.dropout = nn.Dropout(self.dropout_prob)
		self.attn1 = nn.Linear(self.word_embedding_dim, self.word_embedding_dim)
		self.attn2 = nn.Linear(self.word_embedding_dim, self.word_embedding_dim)
		self.attn = nn.Linear(2*self.embedding_dim, 1)
		self.softmax = torch.nn.Softmax(dim=0)
		
		self.sent_rep1 = nn.Linear(self.word_embedding_dim, self.word_embedding_dim)		
		self.sent_rep2 = nn.Linear(self.word_embedding_dim, self.word_embedding_dim)
		self.lin_out = nn.Linear(self.word_embedding_dim, self.num_class, bias=True)
		self.softmax_ = torch.nn.Softmax(dim=2)

	def create_embed_layer(self, w2v_file, a2v_file, trainable = True):
		w2v = bcolz.open(w2v_file)[:]
		a2v = bcolz.open(a2v_file)[:]

		self.num_embed = w2v.shape[0]
		self.num_aspect_embed = a2v.shape[0]
		self.embedding_dim = w2v.shape[1]

		self.aspect_embedding = nn.Embedding(num_embeddings = self.num_aspect_embed, embedding_dim = self.embedding_dim, padding_idx=self.p_idx_aspect)
		self.embedding = nn.Embedding(num_embeddings = self.num_embeddings, embedding_dim = self.embedding_dim, padding_idx=p_idx_sent)
		
		self.embedding.load_state_dict({'weight': w2v})
		self.aspect_embedding.load_state_dict({'weight': a2v})

		if not trainable:
			self.embedding.weight.requires_grad = False
			self.embedding.weight.requires_grad = False

	def forward(self, X, X_lengths, aspect):
		word_emb = self.embedding(X)
		asp_emb = self.embedding(aspect)
		asp_emb = torch.sum(asp_emb)
		embedded = np.ones(self.batch_size, self.max_sent_len, 2*self.embedding_dim)		
		#find mean by taking care of the original size of every aspect
		
		for i in range(self.batch_size):
			for j in range(self.max_sent_len):
				embedded[i][j] = torch.cat([word_emb[i][j],asp_emb[0]])
			embedded[i][X_lengths[i]:] = torch.zeros(2*self.embedding_dim)

		embedded = nn.utils.rnn.pack_padded_sequence(embedded, X_lengths, batch_first=True)

		embedded = self.dropout(embedded)
		# output: (seq_len, batch, hidden_size)
		output, (hidden, memory) = self.LSTM(embedded, (hidden, memory))

		output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
		
		return output, hidden

	def output_gen(self, H, aspect, hidden):
		M_1 = self.attn1(H)
		aspect = aspect.view(1,self.batch_size,-1)

		M_2_cell = self.attn2(aspect)
		M_2 = M_2_cell.view(self.batch_size, self.embedding_dim)
		M_2 = M_2.expand(H.shape[0],self.batch_size,-1)
		#M_1 ->Nxbxd, M_2->Nxbxd
		M = torch.tanh(torch.cat([M_1, M_2], dim=2)) #Nxbx2d
		attn_weights = nn.softmax(self.attn(M)).permute(1,0,2) #bxNx1
		H = H.permute(1,2,0) #bxdxN

		# sent_rep = torch.matmul(H, torch.t(attn_weights))
		sent_rep = torch.zeros(self.batch_size, self.embedding_dim, 1) #bxNx1
		for i in range(self.batch_size):
			sent_rep[i] = torch.mm(H[i],attn_weights[i])

		sent_rep.permute(0,2,1) #bx1xd

		sent_rep = self.tanh(self.sent_rep1(sent_rep) + self.sent_rep2(hidden)) 
		output = self.softmax_(self.lin_out(sent_rep)) #bx1x3

		return output

#inp - bxNx2d, target - bx1x3
def train(input_tensor, aspect, target_tensor, lstm, optimizer, criterion):#=MAX_LENGTH):	
	optimizer.zero_grad()
	loss = 0

	output, hidden = lstm.forward(X, X_lengths, aspect)
	output_ = lstm.output_gen(output, aspect, hidden)
	
	loss += criterion(output_, target_tensor)
	loss.backward()

    optimizer.step()
    
    return loss

def trainIters(lstm, n_iters, print_every=100, learning_rate=0.01):
    print_loss_total = 0  # Reset every print_every

    optimizer = optim.Adagrad(lstm.parameters(), lr=learning_rate, weight_decay=0.001)
  
    criterion = nn.CrossEntropyLoss()

    # for iter in range(1, n_iters + 1):
        # training_pair = training_pairs[iter - 1]
        # input_tensor = training_pair[0]
        # target_tensor = training_pair[1]

        # loss = train(input_tensor, aspect, target_tensor, lstm, optimizer, criterion)
        # print_loss_total += loss
        # plot_loss_total += loss

        # if iter % print_every == 0:
        #     print_loss_avg = print_loss_total / print_every
        #     print_loss_total = 0

if __name__ == "__main__":

















# class LSTM:
# 	def __init__(self, word_embedding_dim, batch_size, num_hidden, lr, num_class, num_iter):
# 		self.word_embedding_dim = word_embedding_dim
# 		self.batch_size = batch_size
# 		self.num_hidden = num_hidden
# 		self.lr = lr
# 		self.num_class = num_class
# 		self.num_iter = num_iter
# 		self.word_to_index = {}
# 		self.word_to_vector= {}
# 		self.aspec_to_id = {}