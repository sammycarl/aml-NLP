import sys
import Python_files.Encoder_v2 as Encoder_v2
import Python_files.CoattentionEncoder_v2 as CoattentionEncoder_v2
import Python_files.Dynamic_Pointing_Decoder as Dynamic_Pointing_Decoder
import Python_files.Config_file as Config_file
import os 
from pathlib import Path
import torch
import torch.nn as nn
import torch.cuda
import numpy as np
import pandas as pd
import csv as csv
import torch.optim as optimizer

import random 
import time
import pickle
from statistics import mean
sys.path.append('')


save = False
cwd = os.getcwd()

#check if cuda is available
#if it is available, set the default tensor to cuda 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == torch.device("cuda:0"):
  torch.set_default_tensor_type(torch.cuda.FloatTensor)
else: 
  torch.set_default_tensor_type(torch.FloatTensor)
#torch.randn((1,2)).long().is_cuda

with open( os.path.dirname(cwd)+'/glove_words_used_pd.txt' , 'r') as glove:
  glove_data_file = glove
  words = pd.read_table(glove_data_file, sep=",", index_col=0, header=None, quoting=csv.QUOTE_ALL)
  
#create the embedding matrix: np matrix of shape vocab_size x embedding dim
embedding_matrix = words.to_numpy()
#add a zero vector that represents words that are not in glove 
zero_vector = np.zeros(300)
embedding_matrix = np.vstack([embedding_matrix, zero_vector])

#create a word_to_index dictionary: key is word and value is index in embedding matrix
#create a index_to_word dictionary: key is index in embeddingmatrix  and value is word
word_to_index = dict()
index = 0 
for word in words.index.values:
  word_to_index[word]=index
  index = index+1

index_to_word = dict((v,k) for k,v in word_to_index.items())

def to_batches(data):
  return [data[i * Config_file.batch_size:(i + 1) * Config_file.batch_size] for i in range((len(data) + Config_file.batch_size - 1) // Config_file.batch_size )]

class Model(nn.Module):
  def __init__(self, embedding_matrix, hidden_dim, dropout_ratio, num_hidden_layers, input_size, hidden_size, u_embedding_dim,device):
    super(Model, self).__init__()
    self.Encoder = Encoder_v2.Encoder(embedding_matrix, hidden_dim, dropout_ratio, num_hidden_layers, device)
    self.CoattentionEncoder = CoattentionEncoder_v2.CoattentionEncoder(input_size, hidden_size, dropout_ratio, num_hidden_layers,device)
    self.Decoder = Dynamic_Pointing_Decoder.Dynamic_Pointing_Decoder(u_embedding_dim, hidden_dim, num_hidden_layers, dropout_ratio,device)

  def forward(self, batch_input_context, batch_context_lengths, batch_input_question, batch_question_lengths, training, padding): 
    start_DEncoding = time.time()
    context_encoding = self.Encoder.forward(batch_input_context, training, padding, question=False)
    end_DEncoding = time.time()
    #print("Time D encoding", end_DEncoding-start_DEncoding)

    start_QEncoding = time.time()
    question_encoding = self.Encoder.forward(batch_input_question, training, padding, question=True)
    end_QEncoding = time.time()
    #print("Time Q encoding", end_QEncoding-start_QEncoding)

    
    start_CoaEnc = time.time()
    U = self.CoattentionEncoder.forward( context_encoding,question_encoding, training, padding )
    end_CoaEnc = time.time()
    #print("Time  CoaEncoder", end_CoaEnc-start_CoaEnc)

    
    start_answer = time.time()
    #@sam add padding argument 
    start_indices, end_indices, start_scores_tensor, end_scores_tensor = self.Decoder.forward(U, training, padding, batch_context_lengths)
    end_answer = time.time()
    #print("Time dynamic coattention decoder", end_answer-start_answer)
    return start_indices, end_indices, start_scores_tensor, end_scores_tensor


Config_file.epochs = 200
Config_file.batch_size = 32
Config_file.dropout_ratio = 0
Config_file.learning_rate = 0.002
Config_file.num_hidden_layers = 1
Config_file.hidden_dim = 200

saveEveryXEpochs = 50 
epoch_left_off = 0
epoch_losses = []

number_to_train = 32
start_from = 0

experimenter = "Fleur" #Change this to your name when you run an experiment
experiment = "dr0.0" #
hyperparameters = {'batch_size':Config_file.batch_size,
                   'learning_rate':Config_file.learning_rate,
                   'dropout_ratio': Config_file.dropout_ratio,
                   'Number_to_train': number_to_train,
                   'Number_hidden_layers': Config_file.num_hidden_layers,
                   'hidden_dim': Config_file.hidden_dim,
                   'MAXITERATIONS':Config_file.MAXITERATIONS,
                   'MAXOUT_LAYER_POOLSIZE':Config_file.MAXOUT_LAYER_POOLSIZE,
                   'epochs' : Config_file.epochs}
          
 

u_embedding_dim = 2*embedding_matrix.shape[1]
#Also "Hidden_size" for now I have used hidden_dim
input_size = 3* embedding_matrix.shape[1]

model = Model(embedding_matrix, Config_file.hidden_dim, Config_file.dropout_ratio, Config_file.num_hidden_layers, input_size, Config_file.hidden_dim, u_embedding_dim,device).to(device)
loss_function = nn.CrossEntropyLoss()

filtered_params = filter(lambda p: p.requires_grad, model.parameters())
adam_optimizer = optimizer.Adam(filtered_params, Config_file.learning_rate)


#print("Starting, config = ...")
padding = False
training = True 

model.train() #Make sure it's in training mode

from tqdm import tqdm

with open(cwd +'/data/train.span', 'r') as span:
  allspans = span.readlines()

spans = allspans[start_from:start_from+number_to_train]
print(hyperparameters)

with open(cwd + '/data/Saved_Files_Test/traincontexts_lengths_1to5', 'rb') as traincontexts_lengths_file:
    contexts_lengths = pickle.load(traincontexts_lengths_file)[start_from:start_from+number_to_train]

with open(cwd +'/data/Saved_Files_Test/trainquestions_lengths_1to5', 'rb') as trainquestions_lengths_file:
    questions_lengths = pickle.load(trainquestions_lengths_file)[start_from:start_from+number_to_train]
                                                                      
with open(cwd +'/data/Saved_Files_Test/traincontexts_indices_1to5', 'rb') as traincontexts_indices_file:
    contexts_indices = pickle.load(traincontexts_indices_file)[start_from:start_from+number_to_train]
with open(cwd +'/data/Saved_Files_Test/trainquestions_indices_1to5', 'rb') as trainquestions_indices_file:
    questions_indices = pickle.load(trainquestions_indices_file)[start_from:start_from+number_to_train]

batch_losses = []
epoch_loss = 0 if len(epoch_losses)==0 else epoch_losses[-1]

for epoch in range(epoch_left_off, Config_file.epochs): 
  print("starting epoch: " + str(epoch+1) + " of: " + str(Config_file.epochs))
  
  if save and epoch%saveEveryXEpochs==0:
    FILE_PATH = cwd + "/Models/" + experimenter+experiment + "epochs" +str(epoch)+"sizedata"+str(number_to_train)
    torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': adam_optimizer.state_dict(),
              'loss': epoch_loss,
              'epoch_losses': epoch_losses,
              'hyperparameters': hyperparameters
              }, FILE_PATH)
    
  if save and epoch%5==0:
    FILE_PATH = cwd + "/Models/" + experimenter+experiment+"sizedata"+str(number_to_train)
    torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': adam_optimizer.state_dict(),
              'loss': epoch_loss,
              'epoch_losses': epoch_losses,
              'hyperparameters': hyperparameters
            }, FILE_PATH)

  timeEpochStuff = time.time()
  temp = list(zip( contexts_indices, questions_indices,contexts_lengths,questions_lengths,spans))
  random.shuffle(temp)
  contexts_indices, questions_indices,contexts_lengths,questions_lengths,spans = zip(*temp)

  #divide the training data in batches of batch_size
  batches_contexts_indices = to_batches(contexts_indices)
  batches_contexts_lengths = to_batches(contexts_lengths)
  batches_questions_indices = to_batches(questions_indices)
  batches_questions_lengths = to_batches(questions_lengths)
  batches_spans = to_batches(spans)
  #print("epoch stuff", time.time()- timeEpochStuff)
  #loop through the batches
  for i in tqdm(range(len(batches_contexts_indices ))):
    if padding:
      batch_context_indices = list(add_padding(batches_contexts_indices[i]))
      batch_question_indices = list(add_padding(batches_questions_indices[i]))
    else: 
      batch_context_indices = [torch.Tensor(batches_contexts_indices[i][k]) for k in range (len(batches_contexts_indices[i])) ]
      batch_question_indices = [torch.Tensor(batches_questions_indices[i][k]) for k in range (len(batches_questions_indices[i])) ]
    #backward() function accumulates gradients and we dont want to mix up gradients between minibatches. 
    #thus we want to set the to zero at the start of a new minibatch
    adam_optimizer.zero_grad()
  

    timeForwardPass= time.time()
    batch_start_indices, batch_end_indices, batch_start_scores_tensor, batch_end_scores_tensor = model(batch_context_indices, list(batches_contexts_lengths[i]), batch_question_indices, list(batches_questions_lengths[i]), training, padding)
    
    #print("timeFor1ForwardPass = ", time.time()- timeForwardPass)
    timeLossCalc = time.time()

    true_start_and_ends = batches_spans[i]
    iter_through_batch = 0 #Something to iterate through loop
    loss_start_indices = 0
    loss_end_indices = 0
    batch_score = 0
    for true_start_unopened in true_start_and_ends:
      true_start, true_end = true_start_unopened.split()
      true_start = int(true_start)
      true_end = int(true_end)
      true_start_tensor = torch.Tensor([true_start]*batch_start_scores_tensor[iter_through_batch].shape[0]).long()
      true_end_tensor = torch.Tensor([true_end]*batch_end_scores_tensor[iter_through_batch].shape[0]).long()
      #print("true_end  ", true_end)
      #print("true_end_tensor ",true_end_tensor)
      #print("true_end_tensor ",true_end_tensor.shape)
      #print("batch_end_scores_tensor[iter_through_batch] ",batch_end_scores_tensor[iter_through_batch])
      #print("batch_end_scores_tensor[iter_through_batch] ",batch_end_scores_tensor[iter_through_batch].shape)

      if true_start == batch_start_indices[iter_through_batch] and true_end == batch_end_indices[iter_through_batch]:
        batch_score += 1
      loss_start_indices += loss_function(batch_start_scores_tensor[iter_through_batch],true_start_tensor)
      loss_end_indices += loss_function(batch_end_scores_tensor[iter_through_batch],true_end_tensor)

      iter_through_batch+=1

    
    #add the losses of the start indices and end indices 
    loss = loss_start_indices + loss_end_indices
    batch_losses.append(loss.item())
    
    #loss.requires_grad = True
    #calculate the gradients using the backward() method of the lossfunction 
    timeLossBack = time.time()
    loss.backward()
    #print("loss.backward ", time.time()- timeLossBack)
    #update the parameters using the step() method of the optimizer 
    adam_optimizer.step()
    #print("loss calculations", time.time()- timeLossCalc)
  
  epoch_loss = mean(batch_losses)
  epoch_losses.append(epoch_loss)
  print("Mean epoch loss ", epoch_loss)
  batch_losses = []

FILE_PATH = cwd + "/Models/" + experimenter + experiment + "epochs" +str(epoch)+"sizedata"+str(number_to_train) + "End"
torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': adam_optimizer.state_dict(),
          'loss': epoch_loss,
          'epoch_losses': epoch_losses,
          'hyperparameters': hyperparameters
          }, FILE_PATH)