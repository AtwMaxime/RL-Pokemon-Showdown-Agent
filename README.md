# RL Pokémon Showdown Agent

This project is an AI agent designed to play Pokémon Showdown battles using reinforcement learning techniques. It utilizes embeddings, pre-trained models, and various scripts to enhance gameplay and decision-making.

## Project Structure

## Project Structure

- **RL_Pokemon_Showdown_Agent/**
  - **data/**                             # Data related to the project
    - **embeddings/**                   # Embedding files
      - `embedding_attaques_dict.json`
      - `embedding_meteo_dict.json`
      - `embedding_object_dict.json`
      - `embedding_talent_dict.json`
      - `embedding_type_dict.json`
      - **random_data/**
        - `Dictionnaire_encoding_prediction_randombattle.json`
    - **sets_random_battle_9/**         # Battle set files for training
      - `battle_data.json`
      - `battle_data2.json`
      - `battle_data3.json`
      - `data_for_training.json`
  - **models/**                           # Pre-trained models and related files
    - **prediction/**
      - `prediction_model_randombattle9g.pth`
      - `scaler_prediction_randombattle9g.pkl`
      - `model_CynthAI_30epochs.pth`
  - **scripts/**                          # Source code for the project
    - `fonction_information.py`        # Information functions
    - `script_entrainement.py`        # Training script
    - `script_prediction_randombattle9g.py`  # Prediction script
    - `play.py`                       # Main play script
  - `README.md`                         # Main project documentation
    
### Data Directory
- **embeddings/**: Contains JSON files that define various embeddings used for attacks, weather conditions, objects, talents, and Pokémon types.
  - `embedding_attaques_dict.json`: Dictionary for attack embeddings.
  - `embedding_meteo_dict.json`: Dictionary for weather condition embeddings.
  - `embedding_object_dict.json`: Dictionary for object embeddings.
  - `embedding_talent_dict.json`: Dictionary for talent embeddings.
  - `embedding_type_dict.json`: Dictionary for type embeddings.

- **random_data/**: Contains random battle data and a dictionary for encoding predictions.
  - `Dictionnaire_encoding_prediction_randombattle.json`: Contains encoding for predictions in random battles.
  - `sets_random_battle_9/`: Folder with JSON files containing data from battles, used for training and testing.
    - `battle_data.json`: Battle data set 1.
    - `battle_data2.json`: Battle data set 2.
    - `battle_data3.json`: Battle data set 3.
    - `data_for_training.json`: Data specifically formatted for training.

- **models/**: Contains pre-trained models and scalers used for predictions in battles.
  - **prediction/**: Contains files related to model predictions.
    - `prediction_model_randombattle9g.pth`: Pre-trained prediction model for random battles.
    - `scaler_prediction_randombattle9g.pkl`: Scaler used for normalizing prediction data.
    - `model_CynthAI_30epochs.pth`: Ppre-trained model of an Agent

### Scripts Directory
- **Fonction_information.py**: Contains utility functions and helper methods to manage and process information related to the Pokémon battles.
- **Script_entrainement.py**: Script responsible for training the AI agent using reinforcement learning techniques.
- **Script_prediction_randombattle9g.py**: Script used for making predictions based on battle data and the trained model.
- **Play.py**: Main script to initiate the game, allowing users to interact with the AI agent, including playing on the ladder or against opponents.

## Usage

To use the RL Pokémon Showdown Agent, all you need to do is run the `Play.py` file. This script will prompt you for your username, password, and whether you want to play on the ladder or against an opponent.

```bash
python scripts/Play.py
```

**Note**: The `play_on_ladder` feature is currently being debugged. Please use the "play against an opponent" option for now.

## Contributing

Contributions are welcome! Please create a pull request or open an issue if you have suggestions or improvements.

Special thanks to the creators of [poke-env](https://github.com/simonm83/poke-env), a powerful library for building Pokémon battle environments, which greatly aids in the development of this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



