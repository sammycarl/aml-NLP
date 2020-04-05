
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
from copy import copy
import numpy as np
import time

MAXITERATIONS = 4 #From paper
MAXOUT_LAYER_POOLSIZE = 16


class Maxout(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)


    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m

class HighWayMaxoutNetwork(nn.Module):
    def __init__(self, embedding_dim, output_dimension, poolsize, dropout_ratio):
        super(HighWayMaxoutNetwork, self).__init__()
        #"Maxout poolsize of 16"
        self.linear1 = nn.Linear(5*embedding_dim,embedding_dim, bias=False)
        self.tanh = nn.Tanh()
        self.maxout1 = Maxout(3*embedding_dim, embedding_dim, poolsize)
        self.maxout2 = Maxout(embedding_dim, embedding_dim, poolsize)
        self.maxout3 = Maxout(2*embedding_dim, output_dimension, poolsize)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, u_i,h_i,u_s_im1,u_e_im1):
        #Where do we deal with bias and weights?????
        timeBefore = time.time()

        a = torch.cat([h_i,u_s_im1, u_e_im1],1) 
        a = self.dropout(a)
        x = self.linear1(a)
        r = self.tanh(x)
        m1 = self.maxout1(torch.cat([u_i,r],1))
        m2 = self.maxout2(m1)
        m3 = self.maxout3(torch.cat([m1,m2],1))
        timeAfter = time.time()
        return m3
        
class Dynamic_Pointing_Decoder(nn.Module):

    def __init__(self, u_embedding_dim, hidden_dim, num_hidden_layers, dropout_ratio,device):
      #arguments passed: 
      '''
      coattention_encoding_U from the coattention encoder matrix dimension with m (number of words in document) by 2l dimension i.e. a 2l vector for each word in the document
      hidden_dim: 200 (Indicated in paper)
      dropout_ratio: fraction neurons dropped as regularization (TBD)

      '''
      super(Dynamic_Pointing_Decoder, self).__init__()
      self.device = device
      self.u_embedding_dim = u_embedding_dim
      self.embedding_dim = int(u_embedding_dim/2)
      self.hidden_dim = hidden_dim
      self.num_hidden_layers = num_hidden_layers
      self.dropout = nn.Dropout(dropout_ratio)
      self.dropout_ratio = dropout_ratio
      input_dimension = 3*u_embedding_dim + self.hidden_dim
      output_dimension =  1 #We just want a score

      self.startHMN = HighWayMaxoutNetwork(self.embedding_dim, output_dimension, MAXOUT_LAYER_POOLSIZE,self.dropout_ratio) 
      self.endHMN = HighWayMaxoutNetwork(self.embedding_dim, output_dimension, MAXOUT_LAYER_POOLSIZE,self.dropout_ratio)

      #need to work out details


      #need to work out details
      #self.lstm = nn.LSTM(2*self.u_embedding_dim, self.embedding_dim, num_layers=self.num_hidden_layers, dropout=self.dropout_ratio)#, self.num_hidden_layers, dropout=self.dropout_ratio)
      self.lstm = nn.LSTM(2*self.u_embedding_dim, self.embedding_dim)#, self.num_hidden_layers, dropout=self.dropout_ratio)
      #Still need to be done 
      #Note that this is part of a forward pass 
    def forward(self, batch_coattention_encoding_U, training, padding, doc_length_vector):
      timeForward = time.time()
      device = self.device
      #initialise h_i, s_i, e_i
      #h_i the hidden state of the LSTM at iteration i
      #s_i, e_i estimate of start and end postion at time i
      #u_s_im1 is the previous estimate of start word encoding
      #u_e_im1 is the previous estimate of end word encoding
      batch_start_indices = []
      batch_end_indices = []
      batch_start_scores_tensor = []
      batch_end_scores_tensor = []
      batch_index = 0
      #for coattention_encoding_U in batch_coattention_encoding_U:
      for coattention_encoding_U_unopened in batch_coattention_encoding_U:

        coattention_encoding_U = coattention_encoding_U_unopened#[0]
      
        # initialize the hidden state as zeros (from paper) might not have to as might default to zero...
        (h_i,c_i) = (torch.zeros(1,1,self.embedding_dim),torch.zeros(1,1,self.embedding_dim))
        
        start_indices = np.array([])
        end_indices = np.array([])

        start_scores = []
        end_scores = []
        #initialise all to zero/ the first word in the document??
        s_i = 0
        e_i = 0 
        u_s_im1 = coattention_encoding_U[0] #is this right? wrong way around...
        u_e_im1 = coattention_encoding_U[0]
        u_s_i = coattention_encoding_U[0]
        u_e_i = coattention_encoding_U[0]

        best_estimate_found = False
        i=0
        
        if(padding):
          docLength = doc_length_vector[batch_index]
        else:
          docLength = coattention_encoding_U.shape[0]

        coattention_encoding_U = coattention_encoding_U[:docLength]
        while not best_estimate_found and i < MAXITERATIONS:
          iterTime = time.time()
          #Do one step
          i += 1
          _, (h_i,c_i) = self.lstm(torch.cat([u_s_im1,u_e_im1],0).view(1,1,-1),(h_i,c_i))
          t = 0

          oneLoop = time.time()
          h_projected = h_i[0].repeat(docLength,1)
          u_s_im1_projected = u_s_im1.repeat(docLength,1)
          u_e_im1_projected = u_e_im1.repeat(docLength,1)

          neural_net = time.time()
          alphas_i = self.startHMN(coattention_encoding_U,h_projected, u_s_im1_projected,u_e_im1_projected) #Score for candidate word u_i being start word
          betas_i = self.endHMN(coattention_encoding_U,h_projected ,u_s_im1_projected ,u_e_im1_projected) #Score for candidate word u_i being end word

          
          alpha_max_i = max(alphas_i) #find largest score
          s_i = (alphas_i == alpha_max_i).nonzero()[0][0].item() #position of start estimate at i'th round
          start_indices = np.append(start_indices,s_i)
          start_scores.append(alphas_i.view(1,-1))

          beta_max_i = max(betas_i)
          e_i = (betas_i == beta_max_i).nonzero()[0][0].item()
          end_indices = np.append(end_indices,e_i)
          end_scores.append(betas_i.view(1,-1))

          u_s_im1 = u_s_i
          u_e_im1 = u_e_i

          u_s_i = coattention_encoding_U[s_i] #new best guess start encoding
          u_e_i = coattention_encoding_U[e_i] #new best guess end encoding
        
          #end condition, positions aren't changing
          if len(start_indices)>1 and (start_indices[-1] == start_indices[-2]) and (end_indices[-1] == end_indices[-2]):
            best_estimate_found = True
        
        start_scores_tensor = torch.cat(start_scores,0)
        end_scores_tensor = torch.cat(end_scores,0)
        batch_start_indices.append(start_indices[-1])
        batch_end_indices.append(end_indices[-1])
        batch_start_scores_tensor.append(start_scores_tensor)
        batch_end_scores_tensor.append(end_scores_tensor)
        batch_index += 1
        
      return batch_start_indices, batch_end_indices, batch_start_scores_tensor, batch_end_scores_tensor
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
from copy import copy
import numpy as np
import time

MAXITERATIONS = 4 #From paper
MAXOUT_LAYER_POOLSIZE = 16


class Maxout(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)


    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m

class HighWayMaxoutNetwork(nn.Module):
    def __init__(self, embedding_dim, output_dimension, poolsize):
        super(HighWayMaxoutNetwork, self).__init__()
        #"Maxout poolsize of 16"
        self.linear1 = nn.Linear(5*embedding_dim,embedding_dim, bias=False)
        self.tanh = nn.Tanh()
        self.maxout1 = Maxout(3*embedding_dim, embedding_dim, poolsize)
        self.maxout2 = Maxout(embedding_dim, embedding_dim, poolsize)
        self.maxout3 = Maxout(2*embedding_dim, output_dimension, poolsize)

    def forward(self, u_i,h_i,u_s_im1,u_e_im1):
        #Where do we deal with bias and weights?????
        timeBefore = time.time()

        a = torch.cat([h_i,u_s_im1, u_e_im1],1) 
        x = self.linear1(a)
        r = self.tanh(x)
        m1 = self.maxout1(torch.cat([u_i,r],1))
        m2 = self.maxout2(m1)
        m3 = self.maxout3(torch.cat([m1,m2],1))
        timeAfter = time.time()
        return m3
        
class Dynamic_Pointing_Decoder(nn.Module):

    def __init__(self, u_embedding_dim, hidden_dim, num_hidden_layers, dropout_ratio,device):
      #arguments passed: 
      '''
      coattention_encoding_U from the coattention encoder matrix dimension with m (number of words in document) by 2l dimension i.e. a 2l vector for each word in the document
      hidden_dim: 200 (Indicated in paper)
      dropout_ratio: fraction neurons dropped as regularization (TBD)

      '''
      super(Dynamic_Pointing_Decoder, self).__init__()
      self.device = device
      self.u_embedding_dim = u_embedding_dim
      self.embedding_dim = int(u_embedding_dim/2)
      self.hidden_dim = hidden_dim
      self.num_hidden_layers = num_hidden_layers
      self.dropout = nn.Dropout(dropout_ratio)
      self.dropout_ratio = dropout_ratio
      input_dimension = 3*u_embedding_dim + self.hidden_dim
      output_dimension =  1 #We just want a score
      self.startHMN = HighWayMaxoutNetwork(self.embedding_dim, output_dimension, MAXOUT_LAYER_POOLSIZE) 
      self.endHMN = HighWayMaxoutNetwork(self.embedding_dim, output_dimension, MAXOUT_LAYER_POOLSIZE)

      #need to work out details
      self.lstm = nn.LSTM(2*self.u_embedding_dim, self.embedding_dim)#, self.num_hidden_layers, dropout=self.dropout_ratio)
      #Still need to be done 
      #Note that this is part of a forward pass 
    def forward(self, batch_coattention_encoding_U, training):
      timeForward = time.time()
      device = self.device
      #initialise h_i, s_i, e_i
      #h_i the hidden state of the LSTM at iteration i
      #s_i, e_i estimate of start and end postion at time i
      #u_s_im1 is the previous estimate of start word encoding
      #u_e_im1 is the previous estimate of end word encoding
      batch_start_indices = []
      batch_end_indices = []
      batch_start_scores_tensor = []
      batch_end_scores_tensor = []
      #for coattention_encoding_U in batch_coattention_encoding_U:
      for coattention_encoding_U_unopened in batch_coattention_encoding_U:

        coattention_encoding_U = coattention_encoding_U_unopened[0]
      
        # initialize the hidden state as zeros (from paper) might not have to as might default to zero...
        (h_i,c_i) = (torch.zeros(1,1,self.embedding_dim),torch.zeros(1,1,self.embedding_dim))
        
        start_indices = np.array([])
        end_indices = np.array([])

        start_scores = []
        end_scores = []
        #initialise all to zero/ the first word in the document??
        s_i = 0
        e_i = 0 
        u_s_im1 = coattention_encoding_U[0] #is this right? wrong way around...
        u_e_im1 = coattention_encoding_U[0]
        u_s_i = coattention_encoding_U[0]
        u_e_i = coattention_encoding_U[0]

        best_estimate_found = False
        i=0
        docLength = coattention_encoding_U.shape[0]
        while not best_estimate_found and i < MAXITERATIONS:
          iterTime = time.time()
          #Do one step
          i += 1
          _, (h_i,c_i) = self.lstm(torch.cat([u_s_im1,u_e_im1],0).view(1,1,-1),(h_i,c_i))
          t = 0

          oneLoop = time.time()
          h_projected = h_i[0].repeat(docLength,1)
          u_s_im1_projected = u_s_im1.repeat(docLength,1)
          u_e_im1_projected = u_e_im1.repeat(docLength,1)

          neural_net = time.time()
          alphas_i = self.startHMN(coattention_encoding_U,h_projected, u_s_im1_projected,u_e_im1_projected) #Score for candidate word u_i being start word
          betas_i = self.endHMN(coattention_encoding_U,h_projected ,u_s_im1_projected ,u_e_im1_projected) #Score for candidate word u_i being end word
          alpha_max_i = max(alphas_i) #find largest score
          s_i = (alphas_i == alpha_max_i).nonzero()[0][0].item() #position of start estimate at i'th round
          start_indices = np.append(start_indices,s_i)
          start_scores.append(alphas_i.view(1,-1))

          beta_max_i = max(betas_i)
          e_i = (betas_i == beta_max_i).nonzero()[0][0].item()
          end_indices = np.append(end_indices,e_i)
          end_scores.append(betas_i.view(1,-1))

          u_s_im1 = u_s_i
          u_e_im1 = u_e_i

          u_s_i = coattention_encoding_U[s_i] #new best guess start encoding
          u_e_i = coattention_encoding_U[e_i] #new best guess end encoding
        
          #end condition, positions aren't changing
          if len(start_indices)>1 and (start_indices[-1] == start_indices[-2]) and (end_indices[-1] == end_indices[-2]):
            best_estimate_found = True

        start_scores_tensor = torch.cat(start_scores,0)
        end_scores_tensor = torch.cat(end_scores,0)
        batch_start_indices.append(start_indices[-1])
        batch_end_indices.append(end_indices[-1])
        batch_start_scores_tensor.append(start_scores_tensor)
        batch_end_scores_tensor.append(end_scores_tensor)
        
      return batch_start_indices, batch_end_indices, batch_start_scores_tensor, batch_end_scores_tensor
"""
