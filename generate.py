import torch
from llm import LLM
from data_types import Config
from tokenizer import Tokenizer
import pickle
import json
import sys

def main():
    model_name = sys.argv[1]
    config_name = sys.argv[2]
    with open(config_name+".pkl", 'rb') as f:
         config = pickle.load(f)
    seinfeld_episodes = json.load(open('seinfeld_scripts.json', 'r'))
    episode_list = []
    for season in seinfeld_episodes.keys():
        for episode in seinfeld_episodes[season].keys():
            episode_list.append((" ".join(seinfeld_episodes[season][episode].split()))[1:])
    all_episodes =" ".join(episode_list)
    tokenizer = Tokenizer()
    all_tokens = tokenizer.process_set_tokenize_text(all_episodes)

    model = LLM(config,tokenizer)

    model.load_state_dict(torch.load(model_name+".pt"))
    user_string = input("Enter prompt to generate from: ")

    print(model.generate(user_string, max_length=100))


if __name__ == "__main__":
    main()