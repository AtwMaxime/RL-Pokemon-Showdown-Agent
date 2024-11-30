import asyncio
import random


import numpy as np

from poke_env.player import RandomPlayer
from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env import LocalhostServerConfiguration

from poke_env.environment.pokemon import Pokemon
from poke_env.environment.move import Move
from poke_env.player import Player, RandomPlayer

import poke_env
import asyncio
import time
from poke_env.environment import AbstractBattle
from poke_env.player.battle_order import BattleOrder,ForfeitBattleOrder, DrawBattleOrder,DefaultBattleOrder
from poke_env.data import GenData, to_id_str
from poke_env.environment.pokemon_type import PokemonType

import glob
import json
import re
import numpy as np
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.status import Status
from poke_env.environment.effect import Effect
from poke_env.environment.pokemon_type import PokemonType

import poke_env.player.battle_order
from poke_env.environment.move import Move


from scripts.Script_prediction_randombattle9g import set_to_prediction
from scripts.Fonction_information import info


json_files = glob.glob('data/sets_random_battle_9g/*.json')
data_list = []

for filename in json_files:
    with open(filename, 'r') as file:
        data = json.load(file)
        data_list.append(data)

import torch
import torch.nn as nn
import torch.optim as optim
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
import numpy as np
from collections import deque
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time


file = open('data/sets_random_battle_9g/data_for_training.json') #File that contains all th
Teams = json.load(file)

t1=''
t2=''
for i in Teams[0][0]:
    t1+=i

for i in Teams[1][0]:
    t2+=i



def reward(battle):
    score = 0
    t2=list(battle.opponent_team.values())

    for pokemon in t2:
        score_status = 10 if str(pokemon.status).split()[0] in ['BRN','FRZ','PAR','PSN','TOX','SLP'] else 0
        score += score_status

    for pokemon in t2:

        score_degats = 0
        if str(pokemon.status).split()[0] =='FNT':
            score_degats = 100
        else:
            score_degats = 50*(1-pokemon.current_hp/100)
        score += score_degats

    return score



### Test d'une classe sans modèle d'IA

#
#
# temps1=time.time()
#
#
#
#
#
#
#
# class DrawAI(Player):
#     def __init__(self, battle_format, team,*args,max_turns=10, **kwargs):
#         super().__init__(battle_format=battle_format, team=team, *args ,**kwargs)
#         self.memory = []
#         self.past_turns = []
#         self.past_vectors = []
#         self.past_rewards = []
#         self.past_actions = []
#         self.playing_turns = []
#         self.past_probas = []
#         self.match_watched = [False]*10
#
#     def teampreview(self, battle):
#         return "/team " + "".join([str(i + 1) for i in range(6)])
#
#     def choose_move(self,battle):
#         old_matchs = list(self.battles.values())
#         if len(old_matchs)>=2 and self.match_watched[len(old_matchs)-2]==False:
#             if (list(self.battles.values())[-2].finished):
#                 self.match_watched[len(old_matchs)-2]=True
#                 reward_final=reward(list(self.battles.values())[-1])
#                 if list(self.battles.values())[-1].won:
#                     reward_final+=500
#                 elif list(p1.battles.values())[-1].lost:
#                     reward_final-=500
#
#                 self.past_rewards.append(reward_final-np.sum(self.past_rewards))
#
#                 self.memory.append((self.past_rewards,self.past_actions,self.past_turns,self.past_vectors,self.playing_turns))
#                 self.past_turns = []
#                 self.past_vectors = []
#                 self.past_rewards = []
#                 self.past_actions = []
#                 self.playing_turns = []
#                 self.past_probas = []
#
#
#
#         score=0
#         turn = battle.turn
#
#         if turn == 2:
#             score = reward(battle)
#
#         if turn >=3:
#             score = reward(battle)-np.sum(self.past_rewards)
#
#
#
#         espace,actions = info(battle)
#
#         k=actions[0]
#
#         self.past_vectors.append(espace)
#         self.playing_turns.append(battle.turn)
#         self.past_actions.append(k)
#         self.past_turns.append(battle)
#         if turn != 1:
#             self.past_rewards.append(score)
#
#
#
#
#         return self.choose_random_move(battle)
#
#
#
# p1=DrawAI(battle_format="gen9anythinggoes",team = t1,max_turns=10)
# p2=DrawAI(battle_format="gen9anythinggoes",team = t2,max_turns=10)
#
#
#
# async def main():
#
#     n_challenge = 10
#     await p1.battle_against(p2, n_battles=n_challenge)
#     # await p1.send_challenges("afafefgqefgzgzara", n_challenges=n_challenge)
#
#
#     return p1,p2
#
# asyncio.run(main())
#
# if (list(p1.battles.values())[-1].finished):
#     print("COMBAT TERMINE")
#     reward_final=reward(list(p1.battles.values())[-1])
#     if list(p1.battles.values())[-1].won:
#         reward_final+=500
#     elif list(p1.battles.values())[-1].lost:
#         reward_final-=500
#
#     p1.past_rewards.append(reward_final-np.sum(p1.past_rewards))
#
#     p1.memory.append((p1.past_rewards,p1.past_actions,p1.past_turns,p1.past_vectors,p1.playing_turns,p1.past_probas))
#
# temps2=time.time()
#
#
# # p1=DrawAI(battle_format="gen9ubers",team=t1)
# # p1=DrawAI(battle_format="gen9randombattle")
#
# t=(temps2-temps1)/60
# print(f"temps necessaire pour 10 matchs : {t}")
# print(f"p1 à :{p1.n_won_battles/p1.n_finished_battles} de wr")

# for j in range(len(p1.memory)):
#     print(f"COMBAT NUMERO {j}")
#     longueur = len(p1.memory[j][0])
#     for i in range(longueur):
#         print(f"{i+1} tour",p1.memory[j][4][i])
#         print(f"{i+1} recompense",p1.memory[j][0][i])
#         print(f"{i+1} action",p1.memory[j][1][i])
#         # print(f"{i+1} past_vectors",p1.memory[j][3][i])
#         print(f"{i+1} battle object",p1.memory[j][2][i])
#         print("\n")



### Definition du model

import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, emb_dim, nhead, num_layers, hidden_dim, output_dim, max_seq_len, dropout_prob=0.1):
        """
        Initialize the Transformer model.

        :param input_dim: Dimension of the input features.
        :param emb_dim: Dimension of the embedding space.
        :param nhead: Number of attention heads in the multihead attention mechanism.
        :param num_layers: Number of layers in the Transformer encoder.
        :param hidden_dim: Dimension of the feedforward network inside the Transformer.
        :param output_dim: Dimension of the output (e.g., number of classes).
        :param max_seq_len: Maximum sequence length to handle.
        :param dropout_prob: Dropout probability for regularization.
        """
        super(TransformerEncoderModel, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len

        # Linear layer for embedding the input
        self.embedding = nn.Linear(input_dim, emb_dim)

        # Generate positional encodings (sinusoidal)
        self.positional_encoding = self._generate_positional_encoding(max_seq_len, emb_dim)

        # Define Transformer encoder layer and the encoder itself
        encoder_layers = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Dropout and additional fully connected layers for final output
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def _generate_positional_encoding(self, max_len, emb_dim):
        """
        Generate sinusoidal positional encodings.

        :param max_len: Maximum length of the sequences.
        :param emb_dim: Dimension of the embeddings.
        :return: Positional encodings tensor.
        """
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim))
        pos_enc = torch.zeros(max_len, emb_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def _create_padding_mask(self, seq_lengths, max_len, device):
        """
        Create a padding mask for the sequences.

        :param seq_lengths: Lengths of sequences in the batch.
        :param max_len: Maximum sequence length in the batch.
        :param device: Device (CPU/GPU) to which tensors will be moved.
        :return: Padding mask tensor.
        """
        mask = torch.arange(max_len, device=device).unsqueeze(0) >= seq_lengths.unsqueeze(1)
        return mask

    def forward(self, x, seq_lengths):
        """
        Forward pass of the model.

        :param x: Input tensor with shape (batch_size, seq_len, input_dim).
        :param seq_lengths: Lengths of sequences in the batch.
        :return: Output tensor with shape (batch_size, output_dim).
        """
        # Apply embedding layer
        x = self.embedding(x)

        # Add sinusoidal positional encoding
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.positional_encoding[positions]

        # Pad sequences to the maximum length
        x_padded = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)

        # Create padding mask
        src_key_padding_mask = self._create_padding_mask(seq_lengths, x_padded.size(1), x.device)

        # Apply Transformer encoder
        x_padded = x_padded.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, emb_dim)
        x = self.transformer_encoder(x_padded, src_key_padding_mask=src_key_padding_mask)

        # Apply dropout and layer normalization
        x = x[-1, :, :]  # Take the output from the last time step
        x = self.dropout(x)
        x = self.layer_norm(x)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# # Example usage
# input_dim = 2412
# emb_dim = 512
# nhead = 8
# num_layers = 2
# hidden_dim = 2048
# output_dim = 14
# max_seq_len = 200
#
# model = TransformerEncoderModel(input_dim, emb_dim, nhead, num_layers, hidden_dim, output_dim, max_seq_len)
#
# # Create example data
# x_data = torch.randn(11, 2412)  # Example input
# x_data = x_data.unsqueeze(0)  # Add batch dimension (batch_size, seq_len, input_dim)
# seq_lengths = torch.tensor([11])  # Length of the sequence
#
# # Forward pass
# try:
#     output = model(x_data, seq_lengths)
#     print(output)
# except Exception as e:
#     print(f"Error during forward pass: {e}")

### Creation du player

import torch
import torch.optim as optim

class CynthAI(Player):
    def __init__(self, battle_format, team,model,nbr_match,temperature=0.95,lr=0.01,*args, **kwargs):
        # Heriter des variables de la classe Player
        super().__init__(battle_format=battle_format, team=team, *args ,**kwargs)

        # On initialise le model, l'optimiser, la fonction de perte et la température
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.temperature = temperature


        # On initialise les autres variables de classe, nécessaire pour récuperer les données importantes pour l'entrainement
        self.memory = []
        self.past_turns = []
        self.past_vectors = []
        self.past_rewards = []
        self.past_actions = []
        self.playing_turns = []
        self.past_probas = []
        self.match_watched = [False]*nbr_match

    def teampreview(self, battle):
        return "/team " + "".join([str(i + 1) for i in range(6)])

    def choose_move(self,battle):
        #Cette partie gère l'enregistrement d'une partie précédente et le nettoyage des variables de classe pour le prochain matchs
        old_matchs = list(self.battles.values())
        if len(old_matchs)>=2:
            if self.match_watched[len(old_matchs)-2]==False:
                if (list(self.battles.values())[-2].finished):
                    self.match_watched[len(old_matchs)-2]=True
                    reward_final=reward(list(self.battles.values())[-1])
                    if list(self.battles.values())[-1].won:
                        reward_final+=500
                    elif list(self.battles.values())[-1].lost:
                        reward_final-=500

                    self.past_rewards.append(reward_final-np.sum(self.past_rewards))

                    self.memory.append((self.past_probas,self.past_rewards,self.past_actions,self.past_turns,self.past_vectors,self.playing_turns))
                    self.past_turns = []
                    self.past_vectors = []
                    self.past_rewards = []
                    self.past_actions = []
                    self.playing_turns = []
                    self.past_probas = []



        #On ajoute le score uniquement à partir du tour 2, car le score d'un tour t est calculé au tour t+1

        score=0
        turn = battle.turn

        if turn == 2:
            score = reward(battle)

        if turn >=3:
            score = reward(battle)-np.sum(self.past_rewards)




        if turn != 1:
            self.past_rewards.append(score)


        if turn ==150:
            return DrawBattleOrder()

        #On utilise la fonction info qui transforme un battle object en un vecteur de taille 2412, et renvoie par la même occasion l'espace des actions possibles qui est une liste dans {0,1}^14 indiquant si chaque action est possible
        espace,actions_possibles = info(battle)


        #On rajoute ces données dans les variables de classe

        self.past_vectors.append(torch.tensor(espace, dtype=torch.float))
        self.playing_turns.append(battle.turn)

        self.past_turns.append(battle)


        #On passe le modèle en mode évaluation pour calculer la meilleur action à choisir

        self.model.eval()
        with torch.no_grad():
            vect = torch.stack(self.past_vectors)
            seq_lengths = torch.tensor([len(self.past_vectors)])
            output = self.model(vect.unsqueeze(0), seq_lengths)
            logits = output.squeeze()

            exp_logits = torch.exp(logits / self.temperature)
            boltzmann_probs = exp_logits / exp_logits.sum()


        actions_possibles_gen9 = actions_possibles[0:4] + actions_possibles[16:20] + actions_possibles[20:26]
        actions_possibles_gen9 = [bool(i) for i in actions_possibles_gen9]





        valid_probs = boltzmann_probs[actions_possibles_gen9]
        if valid_probs.sum() == 0:
            print("Aucune action possible")


            return DefaultBattleOrder()


        # Normalize valid probabilities
        valid_probs = valid_probs / valid_probs.sum()


        action = torch.multinomial(valid_probs, 1).item()
        action_index = torch.nonzero(torch.tensor(actions_possibles_gen9))[action].item()



        self.past_probas.append(logits)
        self.past_actions.append(action_index)


        attaques = list(battle.active_pokemon.moves.values())
        tera_move = False

        if action_index in [0,1,2,3]:
            return self.create_order(attaques[action_index])
        elif action_index in [4,5,6,7]:
            return self.create_order(attaques[action_index-4],terastallize=True)
        elif action_index in [8,9,10,11,12,13]:
            return self.create_order(list(battle.team.values())[action_index-8])

        print("error, random move choisi")
        return self.choose_random_move(battle)



# p1 = CynthAI(battle_format="gen9anythinggoes", team=t1, model=model, nbr_match=1, temperature=0.95,lr=0.01)
#
#
#
# async def main():
#     n_challenge = 1
#     await p1.send_challenges("CompteDeTest07500", n_challenges=1)
#     return p1
#
# asyncio.run(main())
#
#
# if (list(p1.battles.values())[-1].finished):
#     print("COMBAT TERMINE")
#     reward_final=reward(list(p1.battles.values())[-1])
#     if list(p1.battles.values())[-1].won:
#         reward_final+=500
#     elif list(p1.battles.values())[-1].lost:
#         reward_final-=500
#
#     p1.past_rewards.append(reward_final-np.sum(p1.past_rewards))
#
#     p1.memory.append((p1.past_probas,p1.past_rewards,p1.past_actions,p1.past_turns,p1.past_vectors,p1.playing_turns))
#
#
#

# Definir les teams:
import random
import torch

Teams_str=[]
Teams_str_clean = []
Teams_str_clean2 = []


for k in range(0,len(Teams)):
    t =""
    for i in Teams[k][0]:
        t+=i
    Teams_str.append(t)

for i in Teams_str:
    if not('none' in i or 'None' in i):
        Teams_str_clean.append(i)


for i in Teams_str_clean:
    if len(i.split()[0]) <= 17 and len(i.split()[1*54])<= 17 and len(i.split()[2*54])<= 17 and len(i.split()[3*54])<= 17 and len(i.split()[4*54])<= 17 and len(i.split()[5*54])<= 17:
        Teams_str_clean2.append(i)



def initialize_players(i,model):
    team1 = random.choice(Teams_str_clean2)
    team2 = random.choice(Teams_str_clean2)
    print(team1)
    print(team2)
    player1 = CynthAI(battle_format="gen9anythinggoes", team=team1, model=model, nbr_match=100, temperature=0.95,lr=0.01)
    player2 = CynthAI(battle_format="gen9anythinggoes", team=team2, model=model, nbr_match=100, temperature=0.95,lr=0.01)
    return player1, player2

def update_players_postgame(player1,player2):
    if (list(player1.battles.values())[-1].finished):
        print("EPOQUE TERMINE PLAYER 1")
        reward_final=reward(list(player1.battles.values())[-1])
        if list(player1.battles.values())[-1].won:
            reward_final+=500
        elif list(player1.battles.values())[-1].lost:
            reward_final-=500

        player1.past_rewards.append(reward_final-np.sum(player1.past_rewards))

        player1.memory.append((player1.past_probas,player1.past_rewards,player1.past_actions,player1.past_turns,player1.past_vectors,player1.playing_turns))

    if (list(player2.battles.values())[-1].finished):
        print("EPOQUE TERMINE PLAYER 2")
        reward_final=reward(list(player2.battles.values())[-1])
        if list(player2.battles.values())[-1].won:
            reward_final+=500
        elif list(player2.battles.values())[-1].lost:
            reward_final-=500

        player2.past_rewards.append(reward_final-np.sum(player2.past_rewards))

        player2.memory.append((player2.past_probas,player2.past_rewards,player2.past_actions,player2.past_turns,player2.past_vectors,player2.playing_turns))




async def main(player1,player2,game_per_epoch):
    # Play a batch of games
    await player1.battle_against(player2, n_battles=game_per_epoch)

    return player1,player2




# Function to run a batch of games and update the model
def train_batch(player1, player2,model,optimizer,game_per_epoch):
    # Make players play a batch of games
    asyncio.run(main(player1,player2,game_per_epoch))
    update_players_postgame(player1, player2)

    # List of players
    players = [player1, player2]

    for player in players:
        print("Changement de joueur")
        for match in player.memory:
            # Iterate through each 6-tuple in memory
            probas, rewards, actions, turns, vectors, numero_tours = match
            # Process data for each subset of turns
            for end_idx in range(1, len(turns) + 1):
                # Extract the data up to the current end index
                probas_subset = torch.stack(probas[:end_idx])  # Logits up to current turn
                rewards_subset = torch.tensor(rewards[:end_idx], dtype=torch.float32)
                actions_subset = torch.tensor(actions[:end_idx], dtype=torch.long)
                vectors_subset = torch.stack(vectors[:end_idx])  # Assuming vectors are already tensors

                seq_lengths = torch.tensor([len(vectors_subset)])

                vectors_subset = vectors_subset.unsqueeze(0)
                # Apply the model to get logits for the current batch
                model.train()
                optimizer.zero_grad()

                output = model(vectors_subset,seq_lengths)
                logits = output.squeeze()  # Shape: (batch_size, num_actions)

                # Convert logits to probabilities
                probabilities = F.softmax(logits, dim=0)


                # Prepare target probabilities based on actions
                target_probs = torch.zeros_like(probabilities)
                for action in (actions_subset):
                    target_probs[action] = 1.0  # One-hot encoding


                # Compute loss: Negative log likelihood
                loss = -torch.sum(target_probs * torch.log(probabilities + 1e-10), dim=0)  # Add epsilon to avoid log(0)

                # Aggregate loss with rewards
                loss = torch.sum(loss * rewards_subset)  # Apply rewards as weight

                # Update model




                loss.backward()
                player.optimizer.step()

        # Clear memory after processing
        player.memory = []



def load_et_entrainement(k,first_training,batch_size = 100):
    temps1=time.time()

    # Initialize model and players
    model = TransformerEncoderModel(input_dim=2412, emb_dim=128, nhead=8, num_layers=6, hidden_dim=2048, output_dim=14, max_seq_len=150)

    # Number of games per batch

    temperature = 0.95  # Initial temperature

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if not(first_training):
        checkpoint = torch.load('models/model_CynthAI_30epochs.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print("Chargement du modèle")

    # Main training loop
    for epoch in range(k):  # Number of epochs
        player1, player2 = initialize_players(epoch,model)
        train_batch(player1, player2,model,optimizer,batch_size)
        print(f"Epoch {epoch + 1} completed")

        torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,  # L'époque actuelle
        }, f'models/model_CynthAI_{30+epoch+1}epochs.pth')

    temps2=time.time()




    return model

##

from poke_env import AccountConfiguration, ShowdownServerConfiguration

Model_AI = load_et_entrainement(0,False)

##




async def match_en_ligne(player,nbr_matchs):
    await player.ladder(nbr_matchs)

async def match_against(player,username,nbr_matchs):
    await player.send_challenges(username, n_challenges=nbr_matchs)




def play_on_ladder(username,password,nbr_matchs,team_str=None,format_="gen9randombattle"):
    player_test  = CynthAI(battle_format=format_,account_configuration=AccountConfiguration(username, password),server_configuration=ShowdownServerConfiguration , team = team_str , model=Model_AI, nbr_match=nbr_matchs, temperature=0.5,lr=0.01)

    asyncio.run(match_en_ligne(player_test,nbr_matchs))

    print(player_test.n_won_battles)
    print(player_test.n_finished_battles)
    print("winrate:",player_test.n_won_battles/player_test.n_finished_battles)
    player_test.reset_battles()



def play_against(username,password,username_opponent,nbr_matchs=1,team_str=None,format_="gen9randombattle"):
    player_test  = CynthAI(battle_format=format_,account_configuration=AccountConfiguration(username, password),server_configuration=ShowdownServerConfiguration , team = team_str , model=Model_AI, nbr_match=nbr_matchs, temperature=0.5,lr=0.01)

    asyncio.run(match_against(player_test,username_opponent,nbr_matchs))
    player_test.reset_battles()




    # Fix : ogerponwellspring|Embody Aspect (Wellspring)