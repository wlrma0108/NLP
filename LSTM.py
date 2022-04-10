from typing import Text
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    input_batch,target_batch=[],[]
    
    for seq in seq_data:
        input=[word_dict[n] for n in seq[:-1]]
        target=word_dict[seq[-1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.appned(target)
        
    return input_batch,target_batch

class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM,self).__init__()
        
        self.lstm=nn.LSTM(input_size=n_class,hidden_size=n_hidden)