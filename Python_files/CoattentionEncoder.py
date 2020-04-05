import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

'''
1. Get the doument and question encoding (check for the matrix dimensions): 
D (l x m+1) and Q (l x n+1)
2. Compute affinity matrix L = D^T Q (m+1 x n+1)
3. Aq = softmax(L)   (m+1 x n+1)
4. Ad = softmax(L^T) (n+1 x m+1)
5. Cq = DAq          (l x n+1)
6. Cd = [Q;Cq]Ad     (2l x m+1)
 - CqAd              (l x m+1)
 - QAd               (l x m+1)
7. Bidirectional LSTM: fusion of temporal information 

'''
'''
l = 1
m = 3
n = 2

docEncodMatrix = torch.randn(1, l, m+1)
questEncodMatrix = torch.randn(1, l, n+1)

nbatches1, row1, column1 = docEncodMatrix.size()
nbatches2, row2, column2 = questEncodMatrix.size()

if (row1!=row2):
  print("!The matrices are not compatible for matrix multiplication as they don't have the same number of rows!")
else:
  print("Compatible for matrix multiplication")
'''

class coattentionEncoder(nn.Module):
  def __init__(self, input_size, hidden_size, dropout_ratio, num_hidden_layers,device):
    super(coattentionEncoder, self).__init__()
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

  def forward(self, docEncodMatrixBatch, questEncodMatrixBatch, training):
    device = self.device
    lstm_inputs = [] #list of results from forward pass applied to every matrix (one for each batch)

    for doc_single_batch, quest_single_batch in zip(docEncodMatrixBatch, questEncodMatrixBatch):
      lstm_inputs.append(self.forward_single_batch(doc_single_batch, quest_single_batch, training))
      
    return(lstm_inputs)

  def forward_single_batch(self, docEncodMatrix, questEncodMatrix, training):
    
    '''  
    if(training):
        print("Training")
    else:
        print("Not training")
    '''
    transpose_docEncodMatrix = torch.t(docEncodMatrix)
    

    #AFFINITY MATRIX
    #Contains affinity scores corresponding to all pairs of document words and question words
    #Matrix multiplication (mm) between the transpose of the document encoding matrix and the question encoding matrix

    
    L = torch.mm( transpose_docEncodMatrix, questEncodMatrix)  #affinity matrix
    rowL, columnL = L.size()
    #print(rowL, columnL)


    #NORMALISE AFFINITY MATRIX
    #Row-wise to produce the attention weights accross the document for each word in the question  
    
    # Normalize dim will be 1 for row wise
    A_Q = nn.functional.softmax(L, dim=1, _stacklevel=3, dtype=None) #Row wise normalization 
    #print(A_Q[0].mean())
    #print(A_Q[0].std())
    rowAQ, columnAQ = A_Q.size()
    #print(rowAQ, columnAQ)


    #Column-wise to produce the attention weights accross the question for each word in the document
    A_D = nn.functional.softmax(torch.t(L), dim=0, _stacklevel=3, dtype=None) #Column wise normalization
    #print(A_D[:,0].mean())
    #print(A_D[:,0].std())
    rowAD, columnAD = A_D.size()
    #print(rowAD, columnAD)


    #SUMMARIES
    #Attention contexts, of the document in light of each word of the question
    C_Q = torch.mm( docEncodMatrix, A_Q )


    #Cd is a co-dependent representation of the question and document, as the coattention context
    QC_Q = torch.cat((questEncodMatrix, C_Q), dim=0) #concatenate horizontally (but works actually only vertically)

    C_D = torch.mm( QC_Q ,A_D )
    rowCD, columnCD = C_D.size()
    #print(rowCD, columnCD)
    

    D_CD = torch.cat((docEncodMatrix, C_D), dim=0)
    input_lstm = D_CD

    if training:
      input_lstm = self.dropout(input_lstm)
    
    row_input_lstm, column_input_lstm = input_lstm.size()
    #print("row input lstm: ", row_input_lstm)
    #print("column input lstm: ", column_input_lstm)

    #Config_file.batch_size = 1 #for now
    #packed_input = pack_padded_sequence(input_lstm, torch.tensor(row_input_lstm*[column_input_lstm]),batch_first=True)

    packed_input = input_lstm.view(1, column_input_lstm, row_input_lstm)
    #print("lstm input dim", packed_input.shape)
    lstm_output_packed, hidden_state = self.lstm(packed_input)
    #lstm_output = pad_packed_sequence(lstm_output_packed,batch_first=True)

    lstm_output = lstm_output_packed
    #linear layer to get the right output dimension
    #print("lstm input dim", lstm_output.shape)
    output = self.linear(lstm_output) 
    
    return output
