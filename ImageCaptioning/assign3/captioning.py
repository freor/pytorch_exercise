# Write your own image captiong code
# You can modify the class structure
# and add additional function needed for image captionging

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

import numpy as np

import option

class Captioning(nn.Module):

    def __init__(self):
        super(Captioning, self).__init__()

        # input_size, hidden_size, num_layers
        self.lstm = nn.LSTM(option.input_size, option.hidden_size, option.num_layers, dropout=0.7).cuda()

        #self.embeddings = Variable(torch.randn(option.n_words, option.embedding_size)).float()
        #init.uniform(self.embeddings, -0.08, 0.08)
        self.embed = nn.Embedding(option.n_words, option.embedding_size).cuda()

        '''
        # initial state
        self.h0 = torch.randn(option.num_layers, option.batch_size, option.hidden_size).long()#Variable(torch.randn(option.num_layers, option.batch_size, option.hidden_size)).long()
        init.constant(self.h0, 0)
        #self.h0 = self.h0.cuda()
        self.c0 = torch.randn(option.num_layers, option.batch_size, option.hidden_size).long()#Variable(torch.randn(option.num_layers, option.batch_size, option.hidden_size)).long()
        init.constant(self.c0, 0)
        #self.c0 = self.c0.cuda()
        self.hidden = Variable()
        '''

        self.fc = nn.Linear(option.hidden_size, option.n_words)
        init.xavier_uniform(self.fc.weight)
        init.constant(self.fc.bias, 0)

        self.softmax = nn.Softmax()


    def forward(self, x):
        image_features = x[0] # (1, 512, 512)
        captions = x[1]
        size = captions.shape[0]

        lengths = list(map(len, captions)) # take the length of captions
        seq_len = torch.cuda.LongTensor(lengths)
        seq_tensor = Variable(torch.zeros((option.batch_size, 17))).long().cuda() # .long() is error

        for idx, (seq, seqlen) in enumerate(zip(captions, seq_len)):
            seq_tensor[idx, :seqlen] = torch.cuda.LongTensor(seq)

        seq_len, perm_idx = seq_len.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        seq_tensor = seq_tensor.transpose(0, 1) # to (L, B, D)
        seq_tensor = self.embed(seq_tensor)

        captions = nn.utils.rnn.pack_padded_sequence(seq_tensor, seq_len.cpu().numpy())
        # input: (seq_len, batch_size, input_size)
        _, initial_state = self.lstm(image_features)
        packed_out, _ = self.lstm(captions, initial_state)

        out = nn.utils.rnn.pad_packed_sequence(packed_out)
        out = out[0]
        fcs = []
        for i in range(17):
            fcs.append(self.fc(out[i]))

        logits = torch.stack(fcs)

        return logits


    def make_optimizer(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=option.learning_rate, eps=option.epsilon, weight_decay=1e-4)

    def prediction(self, x, word_to_idx, idx_to_word): # x : (10000, 512)
        self.eval()

        captions = Variable(torch.zeros(x.shape[0], 17).float()) # results
        print(idx_to_word[0])
        print(word_to_idx["<END>"])

        for b in range(x.shape[0] // option.test_batch_size):
            s, e = b * option.test_batch_size, (b+1) * option.test_batch_size
            features = Variable(torch.cuda.FloatTensor(x[s: e])).view(1, option.test_batch_size, x.shape[1]) # (1, 500, 512)

            start_mark = np.array([np.full((option.test_batch_size), word_to_idx["<START>"])])
            start_mark = Variable(torch.cuda.LongTensor(start_mark))
            inputs = self.embed(start_mark)

            _, initial_state = self.lstm(features)

            state = initial_state
            captions[s: e, 0] = start_mark
            for i in range(16):
                out, state = self.lstm(inputs, state)
                inputs = out

                out = self.fc(out[0])
                _, prob = torch.max(self.softmax(out), 1)

                prob = prob.view(-1, 1).float()
                captions[s: e, i+1] = prob

        self.train()

        #captions = captions.data.cpu().numpy().astype(int)
        captions = captions.cpu().data.numpy().astype(int)
        print(captions[1])

        return captions

    # for "torch.nn.utils.rn.pck_padded_sequence(inputs, seq_lengths, batch_first=True)
    def get_lengths(self, x): # (data_size, 17)
        return np.argwhere(x==2)[:, 1]







