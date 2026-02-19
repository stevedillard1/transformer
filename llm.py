from torch import nn
import torch
from data_types import Config

class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.activation = nn.ReLU()
        self.linear_up: nn.Linear = nn.Linear(config.d_model, config.d_hidden)
        self.hidden_layer: nn.Linear = nn.Linear(config.d_hidden, config.d_hidden)
        self.linear_down: nn.Linear = nn.Linear(config.d_hidden, config.d_model)

    def forward(self, input):
        a = self.activation(self.linear_up(input))
        a = self.activation(self.hidden_layer(a))
        return self.activation(self.linear_down(a))
    



class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.qk = torch.rand(self.d_model, self.d_model)
        self.ov = torch.rand(self.d_model, self.d_model)
        self.softmax = nn.Softmax()

    def forward(self, embedding):
        n_context = embedding.shape[0]
        
        attention = embedding@self.qk@embedding.T
        negatives = torch.ones(n_context, n_context)*torch.inf*-1
        negatives = torch.triu(negatives, diagonal=1)
        attention += negatives

        A = self.softmax(attention)@embedding@self.ov
        return A
    

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.attention_head = AttentionHead(config)
        self.mlp = MLP(config)


    def forward(self, x):
        return x + self.attention_head(x) + self.mlp(x)


class LLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = torch.nn.Embedding(config.d_vocab, config.d_model)
        self.TB1 = TransformerBlock(config)
        self.TB2 = TransformerBlock(config)
        self.embedding_matrix = torch.nn.Parameter(torch.rand(config.d_vocab, config.d_model))
        self.config = config


    def forward(self, x):
        return self.unembed(self.TB2(self.TB1(self.embed(x))))
    
    def embed(self, data):
        x = torch.zeros(len(data), self.config.d_model)
        for i in range(len(data)):
            x[i,:] = self.embedding_matrix[data[i]]
        return x

    def unembed(self, data):
        return data@self.embedding_matrix.T
    
    def map_token(self, token:int):
        return self.embedding_matrix[token]
        