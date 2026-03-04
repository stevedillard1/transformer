from llm import LLM
from tokenizer import Tokenizer
import torch.nn as nn
import torch
import json
from data_types import Config
import pickle
import os
from pull_seinfeld_scripts import main as pull_seinfeld_scripts_main


def main():
    print("Hello from transformer!")
    print("======Enter Config Values======")
    device = 'cpu'

    dmodel = int(input("dmodel: "))
    num_episodes = int(input("number of episodes: "))
    dhidden = int(input("dhidden: "))
    num_transformer_blocks = int(input("number of transformer blocks: "))
    window_size = int(input("window size: "))
    batch_size = int(input("number to batch by: "))
    PATH = input("name your model (no spaces): ")
    if not os.path.exists('seinfeld_scripts.json'):
        print("Pulling Seinfeld scripts from IMSDb...")
        pull_seinfeld_scripts_main()
    print("\nTraining....")
    seinfeld_episodes = json.load(open('seinfeld_scripts.json', 'r'))
    episode_list = []
    for season in seinfeld_episodes.keys():
        for episode in seinfeld_episodes[season].keys():
            episode_list.append((" ".join(seinfeld_episodes[season][episode].split()))[1:])

    training_episodes = episode_list[:num_episodes]
    all_episodes =" ".join(episode_list)
    tokenizer = Tokenizer()
    all_tokens = tokenizer.process_set_tokenize_text(all_episodes)

    config = Config(dmodel, len(tokenizer.vocab_arr), dhidden,device,num_transformer_blocks)
    model = LLM(config,tokenizer)

    
    data = torch.zeros((1, window_size+1), dtype=torch.long).to(device)
    for episode in training_episodes:
        tokenized_episode = torch.tensor(tokenizer.process_tokenize_encode(episode))
        data = torch.cat([data, tokenized_episode.unfold(0, window_size+1, 1).to(device)], dim=0)

    data = data[1:]
    print(batch_size, data.shape)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)  

    num_epochs = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(num_epochs):
            total_loss = 0
            total_n = 0
            for i, batch in enumerate(data_loader):
                optimizer.zero_grad()
                input_data = batch[:, :-1].to(device)
                output_data = batch[:, 1:].to(device)
                predicted_output = model(input_data)
                # CrossEntropyLoss expects: input (N, C), target (N,) with class indices
                loss = criterion(
                    predicted_output.reshape(-1, config.d_vocab),  # (batch*seq, vocab_size)
                    output_data.reshape(-1)                        # (batch*seq,) class indices
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()*batch.shape[0]
                total_n += batch.shape[0]
                print(f'Progress: {i*batch_size}/{data.shape[0]}, loss= {loss.item()}', end='\r')
            print(f'Epoch: {epoch+1}/{num_epochs}, Average Loss: {(total_loss/total_n)}')
    base_path = "./saved_models/" + PATH + "/"
    model_path = base_path + "model.pt" 
    os.makedirs(base_path, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    with open(base_path + 'config.pkl', 'wb') as f:
        pickle.dump(config, f)
  



if __name__ == "__main__":
    main()
