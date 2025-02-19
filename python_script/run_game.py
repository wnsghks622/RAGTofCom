from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from env_config import load_api_key
from player import Player
from agentic_game import AgenticGame, GameState

# Load API key from .env file
load_api_key()

# Define your prompt template
player_prompt_template = '''
    ... your prompt template ...
'''

player_prompt = PromptTemplate(
    input_variables=["num_players", "profile", "game_pot", "stages_left", "previous_decisions"],
    template=player_prompt_template,
)

def run_game():
    # Create LLMs
    llm1 = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    # Create players
    players = [
        Player(
            id=1,
            name="Conservative",
            profile="...",
            llm=llm1,
            prompt_template=player_prompt
        ),
        # ... other players ...
    ]

    # Game parameters
    test_input = {
        'messages': [],
        'players_pot': 0,
        'game_pot': 100,
        'iterations': 8,
        'stage_number': 1,
        'player_decisions': {}
    }

    # Create and run game
    game = AgenticGame(
        game_state=GameState,
        players=players,
        state_parameters=test_input
    )
    game.create_graph()
    game.run_game()
    
    return game

if __name__ == "__main__":
    game = run_game()
    # Access statistics or plot results
    game.plot_game_statistics() 