from __future__ import unicode_literals, print_function, division
from io import open
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

class atae_LSTM(nn.Module):
	# num_embeddings - train data word dict len
	# taking aspect_embedding_dim = word_embedding_dim
	def __init__(self, num_embeddings, word_embed_dim, num_aspect_embeddings, num_class, max_max_sent_len, dropout_prob):
		super(attention_LSTM, self).__init__()
		self.num_embeddings = num_embeddings	#word to vec dict size
		self.word_embedding_dim = word_embedding_dim
		#self.aspect_embedding_dim = aspect_embedding_dim
		self.num_class = num_class
		self.dropout_prob = dropout_prob
		self.max_sent_len = max_sent_len

		self.embedding = nn.Embedding(num_embeddings = self.num_embeddings, embedding_dim = self.word_embedding_dim,  padding_idx=padding_idx)
		self.aspect_embedding = nn.Embedding(self.num_aspect_embeddings, self.word_embedding_dim)

		self.lstm = nn.LSTM(2*self.word_embedding_dim, self.word_embedding_dim, bias=True, batch_first=True)
		self.dropout = nn.Dropout(self.dropout_prob)
		self.attn1 = nn.Linear(self.word_embedding_dim, self.max_sent_len)
		self.attn1 = nn.Linear(self.aspect_embedding_dim, self.max_sent_len)
		self.attn = nn.Linear( ,self.max_sent_len)
		self.sent_rep1 = nn.Linear(self.word_embedding_dim, self.max_sent_len)		
		self.sent_rep2 = nn.Linear(self.word_embedding_dim, self.max_sent_len)		
		self.lin_out = nn.Linear(self.word_embedding_dim, self.num_class, bias=True)

	def create_embed_layer(self, weights_matrix, trainable = False):
		self.embedding.load_state_dict({'weight': weights_matrix})
		if not trainable:
        	self.embedding.weight.requires_grad = False

	def forward(self, input, hidden, aspect, memory):
		word_emb = self.embedding(input).view(1,1,-1)
		asp_emb = self.embedding(aspect).view(1,1,-1)
		embedded = torch.cat([word_emb,asp_emb], dim=2)
		embedded = self.dropout(embedded)
		output, (hidden, memory) = self.lstm(embedded, (hidden, memory))

		return output, hidden, memory

	def output_gen(self, H, aspect):
		M_1 = self.attn(H)
		M_2_cell = self.attn(aspect)
		M_2 = M_2_cell
		for i in range(self.max_sent_len-1):
			M_2 = torch.cat([M, M_2_cell], dim=1)

		M = torch.tanh(torch.cat([M_1, M_2], dim=0)
		
		attn_weights = F.softmax(self.attn(M))
		sent_rep = torch.matmul(H, torch.t(attn_weights))
		sent_rep = self.tanh(self.sent_rep1(sent_rep) + self.sent_rep2(H[self.max_sent_len])) 
		output = F.softmax(self.lin_out(sent_rep))







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