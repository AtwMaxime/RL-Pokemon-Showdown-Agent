###Modules
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
from poke_env.player.battle_order import BattleOrder,ForfeitBattleOrder, DrawBattleOrder
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

import sys

from scripts.Script_prediction_randombattle9g import set_to_prediction



###Embeddings et Mappings

json_files = glob.glob('data/embeddings/*.json')
data_list = []

for filename in json_files:
    with open(filename, 'r') as file:
        data = json.load(file)
        data_list.append(data)

embedding_attaques = data_list[0]
embedding_meteo = data_list[1]
embedding_objet = data_list[2]
embedding_talent = data_list[3]
embedding_types = data_list[4]

type_mapping = {
    "Normal": "Normal",
    "Fire": "Feu",
    "Water": "Eau",
    "Electric": "Électrique",
    "Grass": "Plante",
    "Ice": "Glace",
    "Fighting": "Combat",
    "Poison": "Poison",
    "Ground": "Sol",
    "Flying": "Vol",
    "Psychic": "Psy",
    "Bug": "Insecte",
    "Rock": "Roche",
    "Ghost": "Spectre",
    "Dragon": "Dragon",
    "Dark": "Ténèbres",
    "Steel": "Acier",
    "Fairy": "Fée",
    "Stellar": "Stellar"
}

type_mapping2 = {
    "NORMAL": "Normal",
    "FIRE": "Feu",
    "WATER": "Eau",
    "ELECTRIC": "Électrique",
    "GRASS": "Plante",
    "ICE": "Glace",
    "FIGHTING": "Combat",
    "POISON": "Poison",
    "GROUND": "Sol",
    "FLYING": "Vol",
    "PSYCHIC": "Psy",
    "BUG": "Insecte",
    "ROCK": "Roche",
    "GHOST": "Spectre",
    "DRAGON": "Dragon",
    "DARK": "Ténèbres",
    "STEEL": "Acier",
    "FAIRY": "Fée",
    "STELLAR": "Stellar"
}

def Capitale(string):
    if len(string) > 0:
        return string[0].upper() + string[1:]
    else:
        return string

def dic_nbr(dic):
    valeurs=[]
    L=[i for i in dic.items()]
    t=0
    if L!=[]:
        for i in range(len(L)):
            valeur=0
            t=L[i]
            Temp=str(t)
            valeur=(Temp.split(":")[1].strip().strip("}>"))[0]
            valeurs.append(float(valeur))
        return valeurs
    else :
        return [0]

def dic_nbr2(D):
    valeurs = []
    for key in D:
        value = key.value  # Obtenir la valeur associée à la clé
        valeurs.append(float(value))
    return valeurs

def dic_val(dic):
    Liste=[]
    for cle in dic.keys():
        Liste.append(cle)
    return Liste


def get_embedding_for_double_type(type1, type2):
    # Convertir les types en tuples possibles
    tuple1 = str((type1, type2))
    tuple2 = str((type2, type1))

    # Vérifier les deux ordres dans le dictionnaire
    if tuple1 in embedding_types:
        return embedding_types[tuple1]
    elif tuple2 in embedding_types:
        return embedding_types[tuple2]
    else:
        print("Double type non trouvé dans le dictionnaire.")
        return None

# Fonction pour formater la clé de Showdown
def format_key(key):
    # Convertir en minuscule, retirer les espaces et apostrophes
    formatted_key = key.lower().replace(" ", "").replace("'", "").replace("-","")
    return formatted_key

# Créer des dictionnaires de correspondance pour chaque type d'embedding
mapping_objet = {format_key(k): k for k in embedding_objet.keys()}
mapping_talent = {format_key(k): k for k in embedding_talent.keys()}
mapping_attaque = {format_key(k): k for k in embedding_attaques.keys()}

# Fonction générique pour obtenir l'embedding à partir de la clé de Showdown
def get_embedding_from_showdown_key(showdown_key, mapping, embedding_data):
    formatted_key = format_key(showdown_key)
    # Trouver la clé dans le mapping
    if formatted_key in mapping:
        correct_key = mapping[formatted_key]
        return embedding_data[correct_key]
    else:
        print("Clé non trouvée dans le dictionnaire.")
        print(showdown_key)
        return None

# Fonctions spécifiques pour chaque type d'embedding
def get_embedding_objet(showdown_key):
    return get_embedding_from_showdown_key(showdown_key, mapping_objet, embedding_objet)

def get_embedding_talent(showdown_key):
    return get_embedding_from_showdown_key(showdown_key, mapping_talent, embedding_talent)

def get_embedding_attaque(showdown_key):
    return get_embedding_from_showdown_key(showdown_key, mapping_attaque, embedding_attaques)


def extract_and_format_type(pokemon_type_str):
    # Extraire la partie TYPE avant l'espace
    type_part = pokemon_type_str.split(' ')[0]

    # Mettre en majuscule la première lettre seulement
    formatted_type = type_part.capitalize()

    return formatted_type




### Fonction Information


def info(battle):
    # print("---------------------")
    Informations = []
    """
    Index DIM 2412 :
        - Weather (dim 2)
        - Fields (dim 12)
        - Force switch / Trapped (dim 2)
        - Side conditions (dim 11 * 2)
        - Team du joueur (dim 984)
            Par pokémon (6*dim 164):
                - Vecteur Type, Vecteur Tera Type, Tera on/off (dim 7)
                - Vecteur Item (dim 16)
                - Vecteur Talent (dim 16)
                - Stats, Bases Stats, Current hp, Boosts (dim 6 + 6 + 1 + 7 = dim 20)
                - Level (dim 1)
                - Active on/off (dim 1)
                - Status (dim 7)
                - Moves (dim 92):
                    Par Moves (4*dim 24):
                        - Vecteur Moves (dim 16)
                        - Vecteur Type (dim 3)
                        - Nombre pp restants (dim1)
                        - Nombre pp max (dim1)
                        - Base Power (dim 1)
                        - Priorité (dim 1)
                        - Nombre coups (dim 1)
        - Team de l'adversaire (dim 984)
        - Pokemon découvets (dim 6)
        - Effets sur les pokémons actifs (dim 187 *2 )
        - Actions possibles (dim 26)
    """

    #Weather
    Weather=battle.weather
    Current_weather = embedding_meteo["UNKNOWN"]

    if Weather:
        Current_weather = embedding_meteo[str(list(Weather.keys())[0]).split(' ')[0]]

    Informations += Current_weather
    # print("Meteo" , len(Informations))

    #Field
    Field=battle.fields

    ELECTRIC_TERRAIN= 0
    GRASSY_TERRAIN= 0
    GRAVITY= 0
    HEAL_BLOCK= 0
    MAGIC_ROOM= 0
    MISTY_TERRAIN= 0
    MUD_SPORT= 0
    MUD_SPOT= 0
    PSYCHIC_TERRAIN= 0
    TRICK_ROOM= 0
    WATER_SPORT= 0
    WONDER_ROOM= 0

    Terrains=[ELECTRIC_TERRAIN,GRASSY_TERRAIN,GRAVITY,HEAL_BLOCK,MAGIC_ROOM,MISTY_TERRAIN,MUD_SPORT,MUD_SPOT,PSYCHIC_TERRAIN,TRICK_ROOM,WATER_SPORT,WONDER_ROOM]
    Terrains_str=["ELECTRIC_TERRAIN","GRASSY_TERRAIN","GRAVITY","HEAL_BLOCK","MAGIC_ROOM","MISTY_TERRAIN","MUD_SPORT","MUD_SPOT","PSYCHIC_TERRAIN","TRICK_ROOM","WATER_SPORT","WONDER_ROOM"]

    if Field:
        Terrains[Terrains_str.index(str(list(Field.keys())[0]).split(' ')[0])]=1

    Informations += Terrains
    # print("Terrains" , len(Informations))

    #Force switch / Trapped :
    Force_switch1=0
    Force_switch0=battle.force_switch

    if Force_switch0==[True] or Force_switch0==True:
        Force_switch1=1

    Trapped0=battle.trapped
    Trapped1=0
    if Trapped0==[True] or Trapped0==True:
        Trapped1=1


    Informations.append(Force_switch1)
    Informations.append(Trapped1)
    # print("Switch" , len(Informations))

    #Sides (Joueur1 1 & 2)

    Opponent_side=battle.opponent_side_conditions
    OS=dic_nbr2(Opponent_side)
    O_Aurora=0
    O_Light_screen=0
    O_lucky_chant=0
    O_mist=0
    O_reflect=0
    O_safeguard=0
    O_spikes=0
    O_stealth_rock=0
    O_sticky_web=0
    O_tail_wind=0
    O_toxic_spike=0

    key1 = SideCondition.SPIKES
    key2 = SideCondition.TOXIC_SPIKES

    if 2 in OS:
        O_Aurora=1
    if 10 in OS:
        O_Light_screen=1
    if 11 in OS:
        O_lucky_chant=1
    if 12 in OS:
        O_mist=1
    if 13 in OS:
        O_reflect=1
    if 14 in OS:
        O_safeguard=1
    if 15 in OS:
        value1=Opponent_side[key1]  # A verif
        O_spikes=value1
    if 16 in OS:
        O_stealth_rock=1
    if 17 in OS:
        O_sticky_web=1
    if 18 in OS:
        O_tail_wind=1
    if 19 in OS:
        value2=Opponent_side[key2]
        O_toxic_spike=value2

    Informations.append(O_Aurora)
    Informations.append(O_Light_screen)
    Informations.append(O_lucky_chant)
    Informations.append(O_mist)
    Informations.append(O_reflect)
    Informations.append(O_safeguard)
    Informations.append(O_spikes)
    Informations.append(O_stealth_rock)
    Informations.append(O_sticky_web)
    Informations.append(O_tail_wind)
    Informations.append(O_toxic_spike)

    Side=battle.side_conditions
    S=dic_nbr2(Side)
    Aurora=0
    Light_screen=0
    lucky_chant=0
    mist=0
    reflect=0
    safeguard=0
    spikes=0
    stealth_rock=0
    sticky_web=0
    tail_wind=0
    toxic_spike=0

    key1 = SideCondition.SPIKES
    key2 = SideCondition.TOXIC_SPIKES

    if 2 in S:
        Aurora=1
    if 10 in S:
        Light_screen=1
    if 11 in S:
        lucky_chant=1
    if 12 in S:
        mist=1
    if 13 in S:
        reflect=1
    if 14 in S:
        safeguard=1
    if 15 in S:
        value1=Side[key1]
        spikes=value1
    if 16 in S:
        stealth_rock=1
    if 17 in S:
        sticky_web=1
    if 18 in S:
        tail_wind=1
    if 19 in S:
        value2=Side[key2]
        toxic_spike=value2

    Informations.append(Aurora)
    Informations.append(Light_screen)
    Informations.append(lucky_chant)
    Informations.append(mist)
    Informations.append(reflect)
    Informations.append(safeguard)
    Informations.append(spikes)
    Informations.append(stealth_rock)
    Informations.append(sticky_web)
    Informations.append(tail_wind)
    Informations.append(toxic_spike)
    # print("Sides", len(Informations))


    #Team1:
    Team1=list(battle.team.values())


    for pokemon in Team1:

        #Type, Tera on, Type tera : 3+3+1, total 7
        #Type
        species = to_id_str(pokemon.species)
        temp_len = len(Informations)
        # print(species)
        dex_entry = pokemon._data.pokedex[species]
        type1_dex = (PokemonType.from_name(dex_entry["types"][0])).__str__().split()[0]
        if len(dex_entry["types"]) == 1:
            type2_dex = None
        else:
            type2_dex = (PokemonType.from_name(dex_entry["types"][1])).__str__().split()[0]



        if (pokemon._terastallized) and (pokemon.tera_type_2 == "Stellar") and not(pokemon.species == 'terapagosstellar'):
            if type2_dex :
                type_couple = (type_mapping2[type1_dex],type_mapping2[type2_dex])
                # print('cas tera stellar + monotype')
            else :
                type_couple = (type_mapping2[type1_dex],type_mapping2[type1_dex])
                # print('cas tera stellar + doubletype')

        elif pokemon._terastallized and not(pokemon.tera_type_2 == "Stellar") :
            type_couple = (type_mapping2[pokemon.type_1.__str__().split()[0]],type_mapping2[pokemon.type_1.__str__().split()[0]])
            # print('cas tera classique')

        elif (pokemon._terastallized) and (pokemon.tera_type_2 == "Stellar") and (pokemon.species == 'terapagosstellar'):
            type_couple = ("Stellar","Stellar")
            # print('cas terapagos Stellar')

        else :
            if pokemon.type_2:
                type_couple = (type_mapping2[pokemon.type_1.__str__().split()[0]],type_mapping2[pokemon.type_2.__str__().split()[0]])
                # print('cas DoubleType')
            else :
                type_couple = (type_mapping2[pokemon.type_1.__str__().split()[0]],type_mapping2[pokemon.type_1.__str__().split()[0]])
                # print('cas Monotype')

        


        a=type_mapping.get(pokemon.tera_type_2)
        
        if a == None:
            print(pokemon)

        # print(type_couple)
        # print((a,a))
        # print(pokemon._terastallized)


        vector_type = get_embedding_for_double_type(type_couple[0],type_couple[1])
        vector_tera = embedding_types[str((a,a))]
        tera_on = int(pokemon._terastallized)

        Informations += vector_type
        Informations += vector_tera
        Informations.append(tera_on)


        #Item and ability : +16*2, total 39
        # print(pokemon.item,pokemon.item == None )

        if pokemon.item != None and pokemon.item != "" :

            Informations += get_embedding_objet(pokemon.item)
        else :
            Informations += embedding_objet['None']


        if pokemon.ability != None and pokemon.ability != "" :

            Informations += get_embedding_talent(pokemon.ability)
        else :
            Informations += embedding_talent['None']






        #Stats, bases stats, current hp, boosts : +6 + 6 + 1 + 7 , total 59
        BS = list(pokemon.base_stats.values())
        Informations += [BS[2],BS[0],BS[1],BS[3],BS[4],BS[5],pokemon.max_hp]
        Informations += list(pokemon.stats.values())
        Informations += list(pokemon.boosts.values())
        Informations.append(pokemon.current_hp)





        #level : +1, total 60


        Informations.append(pokemon.level)




        #active : +1, total 61


        Informations.append(int(pokemon.active))




        #statuts : + 7, total 68


        BRN = 0
        FNT = 0
        FRZ = 0
        PAR = 0
        PSN = 0
        TOX = 0
        SLP = 0
        if str(pokemon.status).split()[0] == "BRN":
            BRN = 1
        if str(pokemon.status).split()[0] == "FNT":
            FNT = 1
        if str(pokemon.status).split()[0] == "FRZ":
            FRZ = 1
        if str(pokemon.status).split()[0] == "PAR":
            PAR = 1
        if str(pokemon.status).split()[0] == "PSN":
            PSN = 1
        if str(pokemon.status).split()[0] == "TOX":
            TOX = 1
        if str(pokemon.status).split()[0] == "SLP":
            SLP = 1


        Informations += [BRN,FNT,FRZ,PAR,PSN,TOX,SLP]




        #moves : +(16 +3 +1 +1 +1 +1+1)*4, total 164

        for move in list(pokemon.moves.values()):
            a=type_mapping2[str(move.type).split()[0]]
            move_representation = get_embedding_attaque(move.id) + embedding_types[str((a,a))] + [move.base_power,move.priority,move.current_pp,move.max_pp,move.expected_hits]
            Informations += move_representation


        for i in range(len(list(pokemon.moves.values())),4):
            Informations += get_embedding_attaque('splash') + embedding_types[str(('Normal','Normal'))] + [0,0,0,0]
            # print('Moins de 4 attaques')

    # print("Team1",len(Informations))


    #Team2:
    Team2=list(battle.opponent_team.values())





    #PREDICTIONS:

    Liste=[]

    t2=list(battle.opponent_team.values())


    for pokemon in t2:
        objet = 'a'
        talent = 'a'
        tera = 'a'


        espece=pokemon.species
        if (pokemon.item != 'unknown_item') and (pokemon.item != "") and (pokemon.item != None):
            objet=pokemon.item
        if pokemon.ability != None:
            talent=pokemon.ability
        if pokemon._terastallized_type != None:
            tera=extract_and_format_type(str(pokemon._terastallized_type))
        moves_temp=list(pokemon.moves.values())
        moves=[i.id for i in moves_temp]

        Liste.append([espece,objet,talent,tera,moves,[]])


        # species = to_id_str(espece)
        # dex_entry = pokemon._data.pokedex[species]
        # type1 = PokemonType.from_name(dex_entry["types"][0])
        # if len(dex_entry["types"]) == 1:
        #     type2 = None
        # else:
        #     type2 = PokemonType.from_name(dex_entry["types"][1])
        # print(type1,type2)

    Predictions = set_to_prediction(Liste)
    # print(Predictions)
    # print("\n")
    # print(Liste)



    #Remplir Vecteur :
    Team2 = list(battle.opponent_team.values())

    # print(Team2)


    for i in range(0,len(Team2)):

        pokemon=Team2[i]

        #Type, Tera on, Type tera : 3+3+1, total 7
        #Type

        species = to_id_str(pokemon.species)
        temp_len = len(Informations)

        dex_entry = pokemon._data.pokedex[species]
        type1_dex = (PokemonType.from_name(dex_entry["types"][0])).__str__().split()[0]
        if len(dex_entry["types"]) == 1:
            type2_dex = None
        else:
            type2_dex = (PokemonType.from_name(dex_entry["types"][1])).__str__().split()[0]



        if (pokemon._terastallized) and (pokemon._terastallized_type == "Stellar") and not(pokemon.species == 'terapagosstellar'):
            if type2_dex :
                type_couple = (type_mapping2[type1_dex],type_mapping2[type2_dex])
                # print('cas tera stellar + monotype')
            else :
                type_couple = (type_mapping2[type1_dex],type_mapping2[type1_dex])
                # print('cas tera stellar + doubletype')

        elif pokemon._terastallized and not(pokemon._terastallized_type == "Stellar") :
            type_couple = (type_mapping2[pokemon.type_1.__str__().split()[0]],type_mapping2[pokemon.type_1.__str__().split()[0]])
            # print('cas tera classique')

        elif (pokemon._terastallized) and (pokemon._terastallized_type == "Stellar") and (pokemon.species == 'terapagosstellar'):
            type_couple = ("Stellar","Stellar")
            # print('cas terapagos Stellar')

        else :
            if pokemon.type_2:
                type_couple = (type_mapping2[pokemon.type_1.__str__().split()[0]],type_mapping2[pokemon.type_2.__str__().split()[0]])
                # print('cas Doubletype')
            else :
                type_couple = (type_mapping2[pokemon.type_1.__str__().split()[0]],type_mapping2[pokemon.type_1.__str__().split()[0]])
                # print('cas Monotype')




        a=type_mapping[Predictions[i][3]]
        
        if a == None:
            print(pokemon)

        # print(type_couple)
        # print((a,a))
        # print(pokemon._terastallized)


        vector_type = get_embedding_for_double_type(type_couple[0],type_couple[1])
        vector_tera = embedding_types[str((a,a))]
        tera_on = int(pokemon._terastallized)

        # print(type_couple,(a,a),tera_on)

        Informations += vector_type
        Informations += vector_tera
        Informations.append(tera_on)



        #Item and ability : +16*2, total 39



        if pokemon.item == None or pokemon.item == "":
            Informations += embedding_objet['None']
            # print("No item")
        elif pokemon.item == 'unknown_item':
            Informations += get_embedding_objet(Predictions[i][1])
            # print("Item inconnu -> prediction",Predictions[i][1])
        else :
            Informations += get_embedding_objet(pokemon.item)
            # print("Item connu",pokemon.item)

        if pokemon.ability == None:
            if get_embedding_talent(Predictions[i][2]) == None:
                print("ERROR NO ABILITY PREDICTED")
            else :
                Informations += get_embedding_talent(Predictions[i][2])

            # print("Ability inconnue -> prediction",Predictions[i][2])
        else :
            Informations += get_embedding_talent(pokemon.ability)
            # print("Ability connue",pokemon.ability)



        #Stats, bases stats, current hp, boosts : +6 + 6 + 7 + 1 , total 59
        BS = list(pokemon.base_stats.values())
        Informations += [BS[2],BS[0],BS[1],BS[3],BS[4],BS[5]]
        Informations += list(Predictions[i][5])
        Informations += list(pokemon.boosts.values())
        Informations.append(int(pokemon.current_hp*Predictions[i][5][0]))






        #level : +1, total 60


        Informations.append(pokemon.level)




        #active : +1, total 61


        Informations.append(int(pokemon.active))




        #statuts : + 7, total 68


        BRN = 0
        FNT = 0
        FRZ = 0
        PAR = 0
        PSN = 0
        TOX = 0
        SLP = 0
        if str(pokemon.status).split()[0] == "BRN":
            BRN = 1
        if str(pokemon.status).split()[0] == "FNT":
            FNT = 1
        if str(pokemon.status).split()[0] == "FRZ":
            FRZ = 1
        if str(pokemon.status).split()[0] == "PAR":
            PAR = 1
        if str(pokemon.status).split()[0] == "PSN":
            PSN = 1
        if str(pokemon.status).split()[0] == "TOX":
            TOX = 1
        if str(pokemon.status).split()[0] == "SLP":
            SLP = 1


        Informations += [BRN,FNT,FRZ,PAR,PSN,TOX,SLP]




        #moves : +(16 +3 +1 +1 +1 +1 +1 )*4, total 164

        for move in list(pokemon.moves.values()):
            a=type_mapping2[str(move.type).split()[0]]
            move_representation = get_embedding_attaque(move.id) + embedding_types[str((a,a))] + [move.base_power,move.priority,move.current_pp,move.max_pp,move.expected_hits]
            Informations += move_representation

        k=0
        j=0
        Liste_moves_id = [m.id for m in list(pokemon.moves.values())]
        while k<4-len(list(pokemon.moves.values())):
            temp_move = Move(Predictions[i][4][j],9)
            # print(temp_move)



            if not (Predictions[i][4][j] in Liste_moves_id):
                move_representation = get_embedding_attaque(temp_move.id) + embedding_types[str((a,a))] + [temp_move.base_power,temp_move.priority,temp_move.current_pp,temp_move.max_pp,temp_move.expected_hits]
                Informations += move_representation
                # print('less than 4 moves, pred:',Predictions[i][4][j])
                Liste_moves_id.append(temp_move.id)
                k+=1
                j+=1

            else :
                # print('less than 4 moves, pred:',Predictions[i][4][j],"Move déjà dans la liste !!")
                j+=1

            # print(Liste_moves_id)



    # print("Team2 Incomplete",len(Informations))

    #Création des vecteurs complémentaires pour l'équipe 2
    for i in range(len(Team2),6):
        Vecteur_remplissage_type = [-99,-99,-99,-99,-99,-99,0]              #Taille 7
        Vecteur_remplissage_objet_talent = [-99]*32                         #Taille 32
        Vecteur_remplissage_stats = [-99]*6 + [-99]*6 + [0]*7 + [-99]       #Taille 20
        Vecteur_remplissage_level = [-99]                                   #Taille 1
        Vecteur_remplissage_active = [0]                                    #Taille 1
        Vecteur_remplissage_Status = [0]*7                                  #Taille 7
        Vecteur_remplissage_Moves = ([-99]*16 + [-99]*3 + [-99]*5)*4        #Taille 164

        Vecteur_remplissage = Vecteur_remplissage_type + Vecteur_remplissage_objet_talent + Vecteur_remplissage_stats + Vecteur_remplissage_level + Vecteur_remplissage_active + Vecteur_remplissage_Status + Vecteur_remplissage_Moves
        Informations += Vecteur_remplissage


    # print("Team2 Complete",len(Informations))

    is_info_masked = [1]*len(Team2) + [0]*(6-len(Team2))


    Informations+= is_info_masked

    # print("Team2 Complete + mask",len(Informations))



    # Effets sur les pokémons actifs de chaque équipe

    for i in range(0,6):
        if Team1[i].active==True:
            numero=i

    for i in range(len(Team2)):
        if Team2[i].active==True:
            opponent_numero=i

    Poke1=Team1[numero]
    Poke2=Team2[opponent_numero]


    Liste_effets1 = list((Poke1.effects.keys()))
    Liste_effets1_value = [m.value for m in Liste_effets1]

    Liste_effets2 = list((Poke2.effects.keys()))
    Liste_effets2_value = [m.value for m in Liste_effets2]

    Effets_Team1=[0]*185
    Effets_Team2=[0]*185

    for k in Liste_effets1_value:
        i=k-1
        if k<36:
            Effets_Team1[i]=1
        elif 36<=k<=41:
            Effets_Team1[36]=k-35
        elif 42<=k<=105:
            Effets_Team1[i-5]=1
        elif 106<=k<=109:
            if k == 106:
                Effets_Team1[101]=4
            elif k == 107:
                Effets_Team1[101]=3
            elif k == 108:
                Effets_Team1[101]=2
            elif k == 109:
                Effets_Team1[101]=1
        elif 110<=k<=155:
            Effets_Team1[i-8]=1
        elif 156<=k<=159:
            Effets_Team1[147]=k-155
        elif 160<=k<=196:
            Effets_Team1[i-11]=1

    for k in Liste_effets2_value:
        i=k-1
        if k<36:
            Effets_Team2[i]=1
        elif 36<=k<=41:
            Effets_Team2[36]=k-35
        elif 42<=k<=105:
            Effets_Team2[i-5]=1
        elif 106<=k<=109:
            if k == 106:
                Effets_Team2[101]=4
            elif k == 107:
                Effets_Team2[101]=3
            elif k == 108:
                Effets_Team2[101]=2
            elif k == 109:
                Effets_Team2[101]=1
        elif 110<=k<=155:
            Effets_Team2[i-8]=1
        elif 156<=k<=159:
            Effets_Team2[147]=k-155
        elif 160<=k<=196:
            Effets_Team2[i-11]=1

    Informations += Effets_Team1
    Informations += Effets_Team2

    # print("Effets classiques",len(Informations))


    Informations.append(int(Poke1.must_recharge))
    Informations.append(1 if Poke1.preparing_move else 0)



    Informations.append(int(Poke2.must_recharge))
    Informations.append(1 if Poke2.preparing_move else 0)

    # print("Effets supplémentaires",len(Informations))

    #Actions possibles (dim 14):
    switch_disponibles = [1]*6
    for i in range(len(Team1)):
        pokemon=Team1[i]
        if not(pokemon in battle.available_switches):
            switch_disponibles[i]=0

    attaques_disponibles = [0]*4
    Moves = list(Poke1.moves.values())

    for i in range(0,len(Moves)):
        move = Moves[i]
        if move in battle.available_moves:
            attaques_disponibles[i]=1

    tera_available = int(battle.can_tera!=None)

    attaques_disponibles_tera = list(np.array(attaques_disponibles)*tera_available)

    #le [0]*12 correspond à attaque + mega/Zmove/dynamax, qui ne sont pas implémentés ici
    actions_disponibles = attaques_disponibles + [0]*12 + attaques_disponibles_tera + switch_disponibles


    Informations += actions_disponibles


    return Informations, actions_disponibles


























