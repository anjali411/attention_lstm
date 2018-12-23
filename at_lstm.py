#python3 at_lstm.py <train/test>
#Requires one argument- train/test

from __future__ import unicode_literals, print_function, division
from io import open
import numpy as np
import unicodedata
import string
import random
# import bcolz
import pickle
import sys

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.optim as optim		

use_cuda = torch.cuda.is_available()

def pad_array(X, filename):
	vocab = pickle.load(open(filename, 'rb'))
	pad_token = vocab['<PAD>']
	#print(pad_token)	
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

def form_input(filename):
	input = pickle.load(open(filename, 'rb'))
	
	input_size = len(input)
	Y = np.zeros(input_size)
	X, aspect = [], []

	for i, item in enumerate(input):
		X.append(item[0])
		aspect.append(item[1])
		Y[i]=item[2]+1 # yi = [neg,neutral, pos]

	X, X_lengths, p_idx_sent = pad_array(X, "50_train_data_word2idx.pkl")
	aspect, aspect_lengths, p_idx_aspect = pad_array(aspect, "50_train_data_aspect2idx.pkl")

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
	def __init__(self,p_idx_sent, p_idx_aspect, max_sent_len=79, dropout_prob = 0.01, batch_size=16, w2vfile="50_word_vec.pkl", a2vfile="50_aspect_vec.pkl", num_class=3):
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

		if use_cuda:
			self.lstm.cuda()
			self.attn1.cuda()
			self.attn2.cuda()
			self.attn.cuda()
			self.sent_rep1.cuda()
			self.sent_rep2.cuda()
			self.lin_out.cuda()

	def create_embed_layer(self, trainable = True):
		# w2v = torch.from_numpy(bcolz.open(self.w2vfile)[:])
		# a2v = torch.from_numpy(bcolz.open(self.a2vfile)[:])
		w2v = torch.from_numpy(pickle.load(open(self.w2vfile, 'rb'))).cuda()
		a2v = torch.from_numpy(pickle.load(open(self.a2vfile, 'rb'))).cuda()

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
		#print(asp_emb.shape, self.batch_size)
		for i in range(self.batch_size):
			asp_emb[i] = asp_emb[i]/asp_lens[i]

		embedded = torch.zeros(self.batch_size, self.max_sent_len, 2*self.embedding_dim).cuda()		
		for i in range(self.batch_size):
			for j in range(X_lengths[i]):
				embedded[i][j] = torch.cat([word_emb[i][j],asp_emb[i]])
		
		embedded = self.dropout(embedded)
		embedded = nn.utils.rnn.pack_padded_sequence(embedded, X_lengths, batch_first=True)

		output, (hidden, memory) = self.lstm(embedded)	
		output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True) # output: (batch, seq_len, hidden_size)
		#print("output.shape, hidden.shape, _.shape", output.shape, hidden.shape, _.shape)
	
		return output, hidden, asp_emb

	#aspect - aspect embeddings
	def output_gen(self, H, aspect, hidden, X_lengths):
		batch_max_sent_len = H.shape[1]

		# print(H.shape, self.max_sent_len * self.batch_size * self.embedding_dim)
		
		M_1 = self.attn1(H).view(batch_max_sent_len, self.batch_size, self.embedding_dim)	# M_1: (N, batch, hidden_size)
		# print(M_1.shape)
		aspect = aspect.view(self.batch_size,1,-1)	# aspect: (batch, 1, hidden_size)

		M_2_cell = self.attn2(aspect)	# M_2_cell: (batch, 1, hidden_size)
		M_2 = M_2_cell.view(self.batch_size, self.embedding_dim) # M_2_cell: (batch, hidden_size)
		M_2 = M_2.expand(batch_max_sent_len, self.batch_size, self.embedding_dim) # M_2: (N, batch, hidden_size)

		for i in range(self.batch_size):
			for j in range(X_lengths[i],batch_max_sent_len):
				M_2[j][i] = torch.zeros(self.embedding_dim)					
		
		#M_1 ->Nxbxd, M_2->Nxbxd
		M = torch.tanh(torch.cat([M_1, M_2], dim=2))	#Nxbx2d
		#print(M.shape)
		attn_weights = self.softmax(self.attn(M)).permute(1,0,2)	#bxNx1
		#print(attn_weights.shape)
		H = H.permute(0,2,1)	#bxdxN
		#print(H.shape)
		sent_rep = torch.zeros(self.batch_size, self.embedding_dim, 1).cuda()	#bxdx1
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

	loss += criterion(output_, target_tensor)
	loss.backward()
	optimizer.step()
	return loss

def trainIters(n_iters, batch_size, learning_rate=0.01,exists=False):
	X, X_lengths, p_idx_sent, aspect, aspect_lengths, p_idx_aspect, Y = form_input('neut_neg_pos_train_input.pkl')
	lstm_ = atae_LSTM(w2vfile="50_word_vec.pkl", a2vfile="50_aspect_vec.pkl", num_class=3, max_sent_len=int(X_lengths[0]), dropout_prob=0.01, p_idx_sent=p_idx_sent, p_idx_aspect=p_idx_aspect, batch_size=batch_size)
	if(exists):
		lstm_.load_state_dict(torch.load("_model_checkpoint.pth"))
	X, X_lengths, aspect, aspect_lengths = np.asarray(X, dtype=int), np.asarray(X_lengths, dtype=int), np.asarray(aspect, dtype=int), np.asarray(aspect_lengths, dtype=int)
	X, X_lengths, aspect, aspect_lengths = torch.from_numpy(X), torch.from_numpy(X_lengths), torch.from_numpy(aspect), torch.from_numpy(aspect_lengths)
	Y = torch.from_numpy(Y).long()	
	
	lstm_.cuda()
	loss_total = 0  # Reset every print_every
	optimizer = optim.Adagrad(lstm_.parameters(), lr=learning_rate, weight_decay=0.001)
	criterion = nn.CrossEntropyLoss()

#	if use_cuda:
#		lstm_.cuda()
#		X.cuda()
#		X_lengths.cuda()
#		aspect.cuda()
#		aspect_lengths.cuda()
#		Y.cuda()

	for i in range(n_iters):
		for j in range(0, X.shape[0], batch_size):
			print(i,j)
			if(j+batch_size<X.shape[0]):
				loss_total += train(X[j:j+batch_size].cuda(), X_lengths[j:j+batch_size].cuda(), aspect[j:j+batch_size].cuda(), aspect_lengths[j:j+batch_size].cuda(), Y[j:j+batch_size].cuda(), lstm_, optimizer, criterion.cuda())
#print(i,loss_total)
		print(i,loss_total)	
		if(i%50 == 0):
			print(i)	
			torch.save(lstm_.state_dict(), str(i)+"_model_checkpoint.pth")	
	torch.save(lstm_.state_dict(), "_model_checkpoint.pth")

def eval(batch_size):
	print("Batch size: ", batch_size)
	X, X_lengths, p_idx_sent, aspect, aspect_lengths, p_idx_aspect, Y = form_input('test_input.pkl')
	print("X.shape: ", X.shape,"padding index sent: ", p_idx_sent, "padding index aspect: ", p_idx_aspect)
	model = atae_LSTM(p_idx_sent=p_idx_sent, p_idx_aspect=p_idx_aspect, batch_size=batch_size).cuda()
	model.load_state_dict(torch.load("_model_checkpoint.pth"))
	X, X_lengths, aspect, aspect_lengths = np.asarray(X, dtype=int), np.asarray(X_lengths, dtype=int), np.asarray(aspect, dtype=int), np.asarray(aspect_lengths, dtype=int)
	X, X_lengths, aspect, aspect_lengths = torch.from_numpy(X), torch.from_numpy(X_lengths), torch.from_numpy(aspect), torch.from_numpy(aspect_lengths)
	Y=torch.from_numpy(Y)
	confusion_matrix=np.zeros((3,3))
	with torch.no_grad():
		for j in range(0, X.shape[0], batch_size):
			if(j+batch_size<X.shape[0]):
				output, hidden, asp_emb = model.forward(X[j:j+batch_size].cuda(), X_lengths[j:j+batch_size].cuda(), aspect[j:j+batch_size].cuda(), aspect_lengths[j:j+batch_size].cuda())
				output_ = model.output_gen(output, asp_emb, hidden, X_lengths[j:j+batch_size].cuda())				
				
				for i in range(batch_size):
					out_ = int(output_[i].max(0)[1])
					target_out = int(Y[j+i])
					#print(type(out_),type(target_out))
					#print(out_,Y[i])
					confusion_matrix[target_out][out_]+=1
		print(confusion_matrix)
		num = confusion_matrix[0][0]+confusion_matrix[1][1]+confusion_matrix[2][2]
		denom = num + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][2]+confusion_matrix[2][1]
		print("accuracy: ",num/denom)
		neg, neutral, pos=0,0,0
		for i in range(X.shape[0]):
			val = int(Y[i])
			if(val == 0):
				neg+=1
			elif(val ==1):
				neutral+=1
			else:
			 	pos+=1
		print("Neg examples:", neg,"Neutral examples:", neutral, "Positive Examples:", pos)

#150 epochs, batch size 16

if __name__ == "__main__":
	if(sys.argv[1] =="train"):
		for i in range(3):
			exists=True
			print("ith run of 50 iterations:", i)
			if(i==0):
				exists=False
			trainIters(n_iters=50, batch_size=16,exists=exists)
	elif(sys.argv[1] =="test"):
		eval(batch_size=16)
