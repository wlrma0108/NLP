from turtle import forward
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN,self).__init__()
        self.num_filters_total= num_filters*len(filter_sizes)
        self.W=nn.Embedding(vocab_size,embedding_size)
        self.Weight=nn.Linear(self.num_filters_total,num_classes,bias=False)
        self.Bias=nn.Parameter(torch.ones([num_classes]))
        self.filter_list=nn.ModuleList([nn.Conv2d(1,num_filters,(size,embedding_size)) for size in filter_sizes])
    def forward(self,X):
        embedded_chars=self.W(X)
        embedded_chars=embedded_chars.unsqueeze(1)
        
        pooled_outputs=[]
        
        for i , conv in enumerate(self.filter_list):
            h=F.relu(conv(embedded_chars))
            mp=nn.MaxPool2d((sequence_length-filter_sizes[i]+1,1))
            pooled=mp(h).permute(0,3,2,1)
            pooled_outputs.append(pooled)
            
        h_pool=torch.cat(pooled_outputs,len(filter_sizes))
        h_pool_flat=torch.reshape(h_pool,[-1,self.num_filters_total])
        model=self.Weight(h_pool_flat)+self.Bias
        return model
    
    
if __name__=='__main__':
    embedding_size=2
    sequence_length=3
    num_classes=2
    filter_sizes=[2,2,2]
    num_filters=3
    
    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels=[1,1,1,0,0,0]
    
    word_list=" ".join(sentences).split()
    word_list=list(set(word_list))
    word_dict={w:i for i, w in enumerate(word_list)}
    vocab_size=len(word_dict)
    
    model=TextCNN()
    
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    
    inputs=torch.LongTensor([np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences])
    targets=torch.LongTensor([out for out in  labels])
    
    for epoch in range(5000):
        optimizer.zero_grad()
        output=model(inputs)
        
        loss=criterion(output,targets)
        if (epoch +1)%1000 ==0:
            print('Epoch:','%04d'%(epoch+1),'cost=','{:0.6f}'.format(loss))
            
        loss.backward()
        optimizer.step()
        
    test_text='sorry hate you'
    tests=[np.asarray[(word_dict[n] for n in test_text.split())]]
    test_batch=torch.LongTensor(tests)
    
    
    predict=model(test_batch).data.max(1,keepdim=True)[1]
    if predict[0][0]==0:
        print(test_text,"is bad")
    else:
        print(test_text,"is good ")        