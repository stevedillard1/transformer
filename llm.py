from torch import nn
import torch
from data_types import Config
from tokenizer import Tokenizer

class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.device = config.device
        self.linear_up: nn.Linear = nn.Linear(config.d_model, config.d_hidden).to(self.device)
        self.hidden_layer: nn.Linear = nn.Linear(config.d_hidden, config.d_hidden).to(self.device)
        self.linear_down: nn.Linear = nn.Linear(config.d_hidden, config.d_model).to(self.device)

    def forward(self, input):
        a = self.activation(self.linear_up(input))
        a = self.activation(self.hidden_layer(a))
        return self.activation(self.linear_down(a))
    



class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.device = config.device
        self.scale = self.d_model ** 0.5
        # Use nn.Parameter so these are trainable, and initialize with smaller values
        self.qk = nn.Parameter(torch.randn(self.d_model, self.d_model) / self.scale).to(self.device)
        self.ov = nn.Parameter(torch.randn(self.d_model, self.d_model) / self.scale).to(self.device)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_data):
        # embedding shape: [batch, n_context, d_model] or [n_context, d_model]
        n_context = input_data.shape[1]
        
        # Matrix multiplications with batch support using transpose on last two dims
        # Scale attention scores to prevent overflow
        attention = (input_data @ self.qk @ input_data.transpose(-2, -1)) / self.scale
        negatives = torch.ones(n_context, n_context) * torch.inf * -1
        negatives = torch.triu(negatives, diagonal=1).to(self.device)
        attention += negatives

        A = self.softmax(attention) @ input_data @ self.ov
        return A
    

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.d_model = config.d_model
        self.attention_head = AttentionHead(config).to(self.device)
        self.mlp = MLP(config).to(self.device)
        self.ln1 = nn.LayerNorm(config.d_model).to(self.device)
        self.ln2 = nn.LayerNorm(config.d_model).to(self.device)


    def forward(self, x):
        x = x + self.attention_head(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LLM(nn.Module):
    def __init__(self, config, tokenizer:Tokenizer):
        super().__init__()
        self.device = config.device
        num_transformer_blocks = config.num_transformer_blocks
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config).to(self.device) for _ in range(num_transformer_blocks)])
        # self.TB1 = TransformerBlock(config).to(self.device)
        # self.TB2 = TransformerBlock(config).to(self.device)
        # Initialize embeddings with normal distribution scaled appropriately
        self.embedding_matrix = torch.nn.Parameter(torch.randn(config.d_vocab, config.d_model)*.02).to(self.device)
        self.positional_embedding = torch.nn.Embedding(config.d_vocab, config.d_model).to(self.device)  # Positional embeddings for up to 1000 tokens
        self.config = config
        self.tokenizer = tokenizer


    def forward(self, x):
        x = self.embed(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.unembed(x)
    
    def embed(self, data):
        # Use direct indexing to preserve gradients
        return self.embedding_matrix[data] #+ self.positional_embedding(torch.arange(data.shape[1]).to(self.device))
    

    def unembed(self, data):
        return data@self.embedding_matrix.T
    
    def map_token(self, token:int):
        return self.embedding_matrix[token]
    
    def generate(self, prompt:str, max_length:int = 50):
            tokens = torch.tensor(self.tokenizer.process_tokenize_encode(prompt)).to(self.device)
            input_size = tokens.shape[0]
            for _ in range(max_length):
                output = self.forward(tokens[-input_size:].reshape(1,-1))
                next_token = torch.argmax(output[0,-1])
                tokens = torch.cat((tokens, next_token.unsqueeze(0)))
            return self.tokenizer.decode(tokens.tolist())