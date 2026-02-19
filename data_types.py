from dataclasses import dataclass

@dataclass
class Config:
    d_model:int
    d_vocab:int
    d_hidden:int