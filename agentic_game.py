from langgraph.graph import Graph, StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, Callable, List, Dict
import operator

from python_script.player import Player
from python_script.game_statistics import GameStatistics

class GameState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]
    players_pot: int
    game_pot: int
    iterations: int
    stage_number: int
    player_decisions: Annotated[Dict[int, List[int]], operator.or_]

class AgenticGame:
    # ... existing AgenticGame class code ... 