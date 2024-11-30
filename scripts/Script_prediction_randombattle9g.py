import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import time

# # Fonction pour charger et normaliser les données depuis un fichier unique
# def load_and_normalize_data(csv_file):
#     # Chargement des données depuis le fichier CSV
#     df = pd.read_csv(csv_file)
    
#     # Extraire les vecteurs d'entrée et de sortie
#     inputs = df['input_vector'].apply(eval).tolist()
#     outputs = df['output_vector'].apply(eval).tolist()
    
#     # Convertir en numpy arrays
#     input_vectors = np.array(inputs)
#     output_vectors = np.array(outputs)
    
#     # Normalisation des données (entre 0 et 1)
#     scaler = MinMaxScaler()
#     output_vectors = scaler.fit_transform(output_vectors)  # Utilise le même scaler pour la sortie
    
#     return input_vectors, output_vectors, scaler

# t0=time.time()

# # Chemin du fichier CSV unique
# csv_file = "D:/PokeIA/RandomData/Random_Battle_9g_vectors_combined2.csv"


# # Charger et normaliser les données
# X, Y, scaler = load_and_normalize_data(csv_file)

# # Suppression des 6 dernières colonnes
# # X = X[:, :-6]


# t1=time.time()
# print(t1-t0)

# # Sauvegarder le scaler pour une utilisation ultérieure

# joblib.dump(scaler, "scaler.pkl")


# # Convertir les données en tenseurs PyTorch
# X = torch.tensor(X).float()
# Y = torch.tensor(Y).float()

# t2=time.time()
# print(t2-t1)

# # Diviser les données en jeux d'entraînement et de validation
# train_size = int(0.8 * len(X))
# val_size = len(X) - train_size
# train_dataset, val_dataset = random_split(TensorDataset(X, Y), [train_size, val_size])

# t3=time.time()
# print(t3-t2)

#%%

# # Préparer les DataLoaders
# batch_size = 256
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCELoss()  # BCE pour les 1193 premiers neurones
        self.mse_loss = nn.MSELoss()  # MSE pour les 6 derniers neurones
    
    def forward(self, outputs, targets):
        bce_loss = self.bce_loss(outputs[:, :1193], targets[:, :1193])
        mse_loss = self.mse_loss(outputs[:, 1193:], targets[:, 1193:])

            
        return bce_loss + 5*mse_loss

# Utilisation de la perte combinée



# Architecture de l'autoencodeur avec Dropout et normalisation L2
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)  # Dropout
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout
            nn.Linear(512, input_dim+6),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


#%%

# # Initialiser le modèle, la perte, l'optimiseur, et la régularisation L2
# input_dim = X.shape[1]
# print(input_dim)
# model = Autoencoder(input_dim)
# criterion = CombinedLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 Regularization avec weight_decay

# # Early Stopping
# patience = 5
# best_loss = float('inf')
# trigger_times = 0

# # Entraînement du modèle avec validation et early stopping
# epochs = 5000
# for epoch in range(epochs):
#     model.train()
#     train_loss = 0.0
#     for data in train_loader:
#         inputs, targets = data
        
#         # Forward pass
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
        
#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         train_loss += loss.item()
    
#     # Calculer la loss de validation
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for data in val_loader:
#             inputs, targets = data
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             val_loss += loss.item()
    
#     train_loss /= len(train_loader)
#     val_loss /= len(val_loader)
    
#     print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
#     # Early stopping
#     if val_loss < best_loss:
#         best_loss = val_loss
#         trigger_times = 0
#         torch.save(model.state_dict(), "best_autoencoder_model.pth")
#     else:
#         trigger_times += 1
#         print(f'Early stopping trigger times: {trigger_times}')
        
#         if trigger_times >= patience:
#             print('Early stopping!')
#             break


#%%
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import time

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCELoss()  # BCE pour les 1193 premiers neurones
        self.mse_loss = nn.MSELoss()  # MSE pour les 6 derniers neurones
    
    def forward(self, outputs, targets):
        bce_loss = self.bce_loss(outputs[:, :1193], targets[:, :1193])
        mse_loss = self.mse_loss(outputs[:, 1193:], targets[:, 1193:])

            
        return bce_loss + 5*mse_loss

# Utilisation de la perte combinée



# Architecture de l'autoencodeur avec Dropout et normalisation L2
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)  # Dropout
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout
            nn.Linear(512, input_dim+6),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



result = {}

with open("data/random_data/Dictionnaire_encoding_prediction_randombattle.json", 'r') as json_file:
    result = json.load(json_file)


def set_to_prediction(Liste_sets):

    result = {}

    with open("data/random_data/Dictionnaire_encoding_prediction_randombattle.json", 'r') as json_file:
        result = json.load(json_file)
    
 
    # Création de la liste de vecteurs
    vector_list = []

    Poke = result["pokemon"]
    Objet = result["objet"]
    Ability = result["talent"]
    Tera = result["terratype"]
    Moves = result["moves"]

    for pokemon_set in Liste_sets:
        pokemon, objet, talent, terratype, moves, stats = pokemon_set
        
        # Création du vecteur one-hot pour chaque composant
        pokemon_vector = [1 if p == pokemon else 0 for p in Poke]
        objet_vector = [1 if o == objet else 0 for o in Objet]
        talent_vector = [1 if t == talent else 0 for t in Ability]
        terratype_vector = [1 if tt == terratype else 0 for tt in Tera]
        moves_vector = [1 if m in moves else 0 for m in Moves]

        # Combine les vecteurs one-hot avec les valeurs numériques
        combined_vector = pokemon_vector + objet_vector + talent_vector + terratype_vector + moves_vector 
        
        # Ajoute le vecteur combiné à la liste des vecteurs
        vector_list.append(combined_vector)
    
    #Chargement du modèle + prédiction
    model = Autoencoder(1193)
    model.load_state_dict(torch.load("models/prediction/prediction_model_randombattle9g.pth"))
    model.eval()
    
    input_tensor=torch.tensor(vector_list).float()

    # Effectuer les prédictions
    with torch.no_grad():
        predictions = model(input_tensor)
    
    data=[]
    for i in range(len(Liste_sets)):
        resultat=[]
        a=predictions[i]
    
        # Chemin du fichier où le scaler est sauvegardé
        scaler_file = 'models/prediction/scaler prediction randombattle9g.pkl'
        scaler = joblib.load(scaler_file)
    
    
        # Appliquer un argmax pour les 561 premiers neurones (Espèce)
        species_pred = torch.argmax(a[:561]).tolist()
    
        # Appliquer un argmax pour les 63 neurones suivants (Objet)
        item_pred = torch.argmax(a[561:624]).tolist()
    
        # Appliquer un argmax pour les 202 neurones suivants (Talent)
        ability_pred = torch.argmax(a[624:826]).tolist()
    
        # Appliquer un argmax pour les 19 neurones suivants (Tera)
        tera_pred = torch.argmax(a[826:845]).tolist()
    
        # Appliquer un top-k(4) pour les 348 neurones suivants (Mouvements)
        moves_pred = torch.topk(a[845:1193], k=4).indices
    
        # Les 6 derniers neurones sont pour les statistiques
        stats_pred = a[1193:1199].tolist()  # Aucune transformation n'est spécifiée, donc on les garde tels quels
    
        Vect=np.zeros(1199)
        Vect[species_pred]=1
        Vect[item_pred]=1
        Vect[ability_pred]=1
        Vect[tera_pred]=1
        for i in moves_pred:    
            Vect[i]=1
        Vect[1193:1199]=stats_pred
    
        Vect_clean = scaler.inverse_transform(Vect.reshape(1, -1))
        Vect_clean=np.rint(Vect_clean[0])
    
    
        resultat.append(result["pokemon"][species_pred])
        resultat.append(result["objet"][item_pred])
        resultat.append(result["talent"][ability_pred])
        resultat.append(result["terratype"][tera_pred])
        resultat_moves=[]
        for i in moves_pred:
            resultat_moves.append(result["moves"][i])
        
        resultat.append(resultat_moves)
    
        resultat.append(Vect_clean[-6:])
        data.append(resultat)

    return data






