import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
from copy import copy
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, dropout_ratio, num_hidden_layers, device):
        # arguments passed:
        '''
        embedding_matrix: obtained from preprocessing = np matrix of shape vocab_size x embedding dim
        hidden_dim: 200 (Indicated in paper)
        dropout_ratio: fraction neurons dropped as regularization (TBD)
        '''
        super(Encoder, self).__init__()
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #if self.device == torch.device("cuda:0"):
         #   torch.set_default_tensor_type(torch.cuda.FloatTensor)
        #else:
        #    torch.set_default_tensor_type(torch.FloatTensor)
        self.device = device
        print(self.device)
        self.num_hidden_layers = num_hidden_layers
        self.embedding_matrix = embedding_matrix
        self.hidden_dim = hidden_dim
        # nr pretrained words
        self.vocab_size = self.embedding_matrix.shape[0]
        # dimension of the embedding (300)
        self.embedding_dim = self.embedding_matrix.shape[1]
        # We want to use pre-trained word embeddings as initial weight vectors for embedding layer in a encoder model
        # We already have pretrained embeddings so we have to turn it to a tensor
        self.weights = torch.from_numpy(embedding_matrix)
        self.embedding = nn.Embedding.from_pretrained(self.weights)

        # In case we need dropout
        dropout_ratio = 0.0001
        self.dropout = nn.Dropout(dropout_ratio)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.encoder_lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_hidden_layers, dropout=dropout_ratio)
        '''
        input size: self.embedding_dim  
        hidden_size: self.hidden_dim 
        num_layers: self.num_hidden_layers
        dropout: self.dropout
        '''
        # linear layers that must be trained along with the LSTM
        # Applies a linear transformation to the incoming data: y=Ax+b
        # It transforms the hidden state representation (lstm_output) into a vector of size embedding_dim
        # Not sure about output dimensions (what is l in paper, i assume here it is self.embedding_dim?)
        # if l = hidden_dim we do not need linear1
        self.linear1 = nn.Linear(hidden_dim, self.embedding_dim)
        self.linear2 = nn.Linear(self.embedding_dim, self.embedding_dim)
        # Non-linear projection layer to allow for variation between the question encoding
        # and the document encoding space
        self.tanh = nn.Tanh()

        # https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html

    def forward(self, input_data, input_lengths, training, question):
        '''
        - input: batch questions and/or contexts
        batch input is a list with the indices of the words in a question/context
        e.g. [[5, 18, 29,0], [32, 100,0,0], [699, 6, 9, 17]]. Length list = batchsize
        - input_lengths = [3,2,4]
        - training: boolean indicating whether we are training or not
        - question: boolean whether it is a question or not
        '''
        batch_size = len(input_data)
        # For packing (see further) we require sorted sequences in the input batch (in descending order of sequence length)
        #sorted_input_lenghts, indices = torch.sort(torch.tensor(input_lengths), descending=True)
        input_data = torch.cat(input_data, dim=0).view( batch_size, len(input_data[0]))
        #sorted_input = input_data[indices]
        sorted_input = input_data 
        # get the corresponding embedding (use the instance of nn.Embedding)
        # batch_input = B x T, where B is the batch size and T the length of the longest sequence
        # With the embedding it becomes a B x T x E matrix
        
        input_embeddings = self.embedding(torch.tensor(sorted_input, device=self.device).long())


        # if we are training use dropout
        if training:
            input_embeddings = self.dropout(input_embeddings)

        # before we feed the embeddings into the LSTm we want to pack them.
        # This are formats that pytorchs rnns can read and ignore the padded inputs
        # calculating gradients for backpropagation
        '''
        When training LSTM it is difficult to batch the variable length sequences. 
        E.g. if length of sequences in a batch size of 8 is: [4,6,8,5,4,3,7,8], 
        you will pad all the sequences and that will results in 8 sequences of length 8
        You do 64 computations instead of 45. 
        Packing allows the RNN not to continue processing the padding
        '''

        #packed_input_embeddings = pack_padded_sequence(input_embeddings, sorted_input_lenghts, batch_first=True)

        '''
        padding does not work 
        '''
        packed_input_embeddings = input_embeddings
        # Run the input = sentence through the lstm
        # I think we do not hve to specify the hidden states (will be set to zero automatically)

        lstm_output_packed, hidden_state = self.encoder_lstm(packed_input_embeddings.float())

        # unpack the output
        # lstm_output will be of size B * T * hidden_dim (not sure?)
        # lstm_output = pad_packed_sequence(lstm_output_packed,batch_first=True)
        '''
        padding does not work 
        '''
        lstm_output = lstm_output_packed

        # Use the linear layer to get the right dimensions
        encoding = self.linear1(lstm_output)

        # add sentinel vector
        # sentinel vector must be trained
        sentinel = torch.randn(( batch_size, 1, encoding.shape[2]), requires_grad=True)
        encoding = torch.cat((encoding, sentinel), 1)

        if training:
            encoding = self.dropout(encoding)

        if question:
            encoding = self.tanh(self.linear2(encoding))

        return encoding
