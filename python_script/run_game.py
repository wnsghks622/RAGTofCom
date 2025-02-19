from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from env_config import load_api_key
from player import Player
from agentic_game import AgenticGame, GameState

# Load API key from .env file
load_api_key()

player_prompt_template = '''
    You are one out of {num_players} players in an economic game with the following behavioral profile:
    {profile}
    -----
    You must choose how much of the {game_pot} dollars pot to claim for yourself.
    Other players will make their decisions simultaneously.
    Your goal is to maximize your profits (i.e., how much you are left with at the end of the game).
    If the total pot is depleted then the game ends and no player receives revenue.
    The remaining money in the pot doubles after each round.
    
    There are {stages_left} rounds left to play. 

    Your response must only contain the single integer between 0 and {game_pot} representing your decision.

    These are the decisions other players made so far through the game.
    {previous_decisions}
'''

player_prompt = PromptTemplate(
    input_variables=["num_players", "profile", "game_pot", "stages_left", "previous_decisions"],
    template=player_prompt_template,
)

def run_game():
        # Create LLMs
        # Create different LLMs for different players
    llm1 = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    llm2 = ChatGroq(model_name="llama3-8b-8192", temperature=0)
    llm3 = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0)

    # Create players with different configurations
    players = [
            Player(
                id=1,
                name="Conservative",
                profile="You are a conservative agent. Try to maximize your profit without depleting the pot by carefully observing other players' decisions.", # should be replaced by RAG profile
                llm=llm1,
                prompt_template=player_prompt
            ),
            Player(
                id=2,
                name="Aggressive",
                profile="You are an aggressive agent. Try to maximize your profit without depleting the pot by carefully observing other players' decisions.", # should be replaced by RAG profile
                llm=llm1,
                prompt_template=player_prompt
            ),
            Player(
                id=3,
                name="Progressive",
                profile="You are a progressive agent. Try to maximize your profit without depleting the pot by carefully observing other players' decisions.", # should be replaced by RAG profile
                llm=llm1,
                prompt_template=player_prompt
            ), # add more players if you want
            Player(
                id=4,
                name="Smart",
                profile="You are a smart agent. Try to maximize your profit without depleting the pot by carefully observing other players' decisions.", # should be replaced by RAG profile
                llm=llm1,
                prompt_template=player_prompt
            )
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
    print(game.get_game_statistics())
    #game.plot_game_statistics() 