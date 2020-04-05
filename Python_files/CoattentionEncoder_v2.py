import torch
import torch.nn as nn
import torch.cuda
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import random 
class CoattentionEncoder(nn.Module):

  def __init__(self, input_size, hidden_size, dropout_ratio, num_hidden_layers,device):
    super(CoattentionEncoder, self).__init__()
    self.device = device
    self.input_size = input_size # 3 x L 
    self.hidden_dim = hidden_size
    self.dropout_ratio = dropout_ratio
    self.num_hidden_layers = num_hidden_layers
    #do not hard code. Linear layer makes sure that the outup size of the coattention encoder is the same as the input for the Dynamic Coatt Point Decoder
    #Apparently if we se bilinear lstm the output is 2 times the hidden size
    self.lstm_output_dimension = self.input_size * 2 / 3


    self.linear = nn.Linear(hidden_size * 2, int(self.lstm_output_dimension))
    self.lstm = torch.nn.LSTM(input_size, hidden_size, self.num_hidden_layers, bidirectional=True)
      
    #In case we need dropout 
    self.dropout = nn.Dropout(self.dropout_ratio)

  def forward(self, docEncodMatrixBatch, questEncodMatrixBatch, training, padding):
    #Input is a list with E x T tensors. 
    #If padding = True, T is the same for every tensor. Else T is different for every tensor
    batch_size = len(docEncodMatrixBatch)
    # If there is padding we want to put everything at once through the lstm 
    if padding: 
      lstm_inputs = [] 
      for i in range(batch_size):
        #returns a list with  T x 900 tensors 
        lstm_inputs.append(self.forward_single_batch(docEncodMatrixBatch[i], questEncodMatrixBatch[i], training,padding))
      #tensor of dimension B x T x 900
      lstm_input_batch = torch.stack(lstm_inputs)
      #through lstm: returns lstm_output which is a BxTx (2*hidden_dim) tensor 
      #add pack and unpack stuff!! 
      lstm_output, hidden_state = self.lstm(lstm_input_batch)
      #through linear layer: returns a tensor of shape B x T x 600 tensor 
      output = self.linear(lstm_output) 
      if training:
        output = self.dropout(output)
      #remove the sentinel vector before giving it to the deocder 
      #returns a list with T x 600 tensors. Note that T is different for every sample 
      return list(output[:,:-1,:])
    # If there is no padding we want to put every sample one by one throught the lstm 
    else: 
      lstm_outputs = [] #list of results from forward pass applied to every matrix (one for each batch)
      for i in range(batch_size): 
        lstm_outputs.append(self.forward_single_batch(docEncodMatrixBatch[i], questEncodMatrixBatch[i], training,padding))
      #returns a list with T x 600 tensors. Note that T is different for every sample 
      return lstm_outputs

  def forward_single_batch(self, docEncodMatrix, questEncodMatrix, training,padding):
    transpose_docEncodMatrix = torch.t(docEncodMatrix)
    L = torch.mm( transpose_docEncodMatrix, questEncodMatrix)  
    rowL, columnL = L.size()
    A_Q = nn.functional.softmax(L, dim=1, _stacklevel=3, dtype=None) #Row wise normalization 
    rowAQ, columnAQ = A_Q.size()
    A_D = nn.functional.softmax(torch.t(L), dim=0, _stacklevel=3, dtype=None) #Column wise normalization
    rowAD, columnAD = A_D.size()
    C_Q = torch.mm( docEncodMatrix, A_Q )
    QC_Q = torch.cat((questEncodMatrix, C_Q), dim=0) #concatenate horizontally (but works actually only vertically)
    C_D = torch.mm( QC_Q ,A_D )
    rowCD, columnCD = C_D.size()
    D_CD = torch.cat((docEncodMatrix, C_D), dim=0)
    input_lstm = D_CD
    #change the dimensions: 1 x T x E
    input_lstm = input_lstm.view(1, input_lstm.shape[1], input_lstm.shape[0])
    if padding: 
      #if padding we want to let a batch go through the lstm 
      #input_lstm is a T x 900 tensor
      return input_lstm[0]
    #through lstm: returns lstm_output which is a 1xTx (2*hidden_dim) tensor 
    lstm_output, hidden_state = self.lstm(input_lstm)
    # through linear layer: returns a 1xTx 600 tensor 
    output = self.linear(lstm_output) 
    if training:
      output = self.dropout(output)
    #returns the output of the LSTM which is a T x 600 tensor 
    return output[0][:-1,:]
