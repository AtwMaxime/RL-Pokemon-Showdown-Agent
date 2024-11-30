# Importing necessary functions from the script module
from scripts.Script_entrainement import play_against, play_on_ladder

# Define constants for the match settings
TEAM_STRING = None  # Required only for formats that need a team (e.g., OU, UU, etc.)
FORMAT = "gen9randombattle"

# Prompt the user for input
username = input("Enter your username: ")
password = input("Enter your password: ")

# Ask the player to choose between playing on the ladder or playing against an opponent
game_mode = input("Choose your game mode: \n1. Play on ladder\n2. Play against an opponent\nEnter 1 or 2: ")

# Ask for the number of matches to play
number_of_matches = input("Enter the number of matches to play: ")

# Convert the number of matches to an integer
try:
    number_of_matches = int(number_of_matches)
    if number_of_matches <= 0:
        raise ValueError("Number of matches must be greater than 0.")
except ValueError as e:
    print(f"Invalid input for number of matches: {e}")
    number_of_matches = 1  # Default to 1 match if input is invalid

if game_mode == '1':
    # If the player chooses to play on ladder
    play_on_ladder(username, password, number_of_matches, TEAM_STRING, FORMAT)
elif game_mode == '2':
    # If the player chooses to play against an opponent
    username_opponent = input("Enter your opponent's username: ")
    play_against(username, password, username_opponent, number_of_matches, TEAM_STRING, FORMAT)
else:
    print("Invalid choice! Please enter 1 or 2.")
