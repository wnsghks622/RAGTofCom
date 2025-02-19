from langgraph.graph import Graph, StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, Callable, List, Dict
import operator

from player import Player
from game_statistics import GameStatistics

class GameState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]
    players_pot: int
    game_pot: int
    iterations: int
    stage_number: int
    player_decisions: Annotated[Dict[int, List[int]], operator.or_]

class AgenticGame:
    def __init__(self,
                 game_state: TypedDict,
                 players: List[Player],
                 state_parameters: dict):

        self.game_state = game_state
        self.players = {player.id: player for player in players}
        self.state_parameters = state_parameters
        self.graph = None
        self.stats = GameStatistics()
        self.initial_pot = state_parameters['game_pot']
        self.max_iterations = state_parameters['iterations']

    
        # Validate players
        if not players:
            raise ValueError("At least one player must be provided")
        if len(set(p.id for p in players)) != len(players):
            raise ValueError("Player IDs must be unique")

        # Create node functions that only take state as parameter
        self.nodes = {
            "supervisor": self._create_supervisor_node(),
            **{f"player{p.id}": self._create_player_node(p.id) for p in players},
            "aggregator": self._create_aggregator_node(),
        }

    def _create_supervisor_node(self) -> Callable:
        def supervisor_node(state: GameState) -> GameState:
            self.stats.rounds_played += 1
            return {
                'messages': [f"---- Stage {state['stage_number']} ----"]
            }
        return supervisor_node

    def _create_player_node(self, player_id: int) -> Callable:
        player = self.players[player_id]

        def player_node(state: GameState) -> GameState:
            # Create a log of what other players chose last round, excluding current player
            previous_decisions1 = "\n".join(
                f"Player {pid}: {state['player_decisions'].get(pid, [])[-1] if state['player_decisions'].get(pid, []) else 'No previous decision'}"
                for pid in self.players if pid != player_id  # Only show other players' decisions
            )
            # Shows all previous decisions, excluding current player.
            previous_decisions2 = "\n".join(
            f"Player {pid}: {state['player_decisions'].get(pid, [])} "
            for pid in self.players if pid != player_id
            )

            print(previous_decisions2)

            prompt = player.prompt_template.format(
                num_players=len(self.players),
                game_pot=state['game_pot'],
                stages_left=state['iterations'] - state['stage_number'],
                profile=player.profile,
                previous_decisions=previous_decisions2 
            )

            decision = player.llm.invoke(prompt).content
            try:
                claim = int(decision)
                self.stats.player_claims[player_id].append(claim)
            except ValueError:
                print(f"Warning: {player.name} made invalid decision: {decision}")
                claim = 0
                self.stats.player_claims[player_id].append(claim)

            # Update to use dictionary for player decisions
            current_decisions = state.get('player_decisions', {})
            current_decisions.setdefault(player_id, []).append(claim)

            return {
                'messages': [str(claim)],
                'player_decisions': current_decisions
            }

        return player_node

    def _create_aggregator_node(self) -> Callable:
        def aggregator_node(state: GameState) -> GameState:
            try:
                players_aggregate = sum([int(decision) for decision in state['messages'][-len(self.players):]])
                self.stats.total_claimed += players_aggregate

                new_pot = state['game_pot'] - players_aggregate

                # doubling what's remaining in the pot after each stage
                double_pot = new_pot * 2
                self.stats.final_pot = double_pot

                return {
                    'players_pot': players_aggregate,
                    'game_pot': double_pot,
                    'stage_number': state['stage_number'] + 1
                }
            except ValueError:
                print("Warning: Invalid decision format in aggregator")
                return {
                    'players_pot': 0,
                    'game_pot': state['game_pot'] * 2,
                    'stage_number': state['stage_number'] + 1
                }
        return aggregator_node

    def _decision_node(self, state: GameState) -> bool:
        should_continue = state['game_pot'] > 0 and state['iterations'] > state['stage_number']

        if not should_continue:
            self.stats.game_completed = True
            if state['game_pot'] <= 0:
                self.stats.end_condition = "POT_DEPLETED"
            elif state['iterations'] <= state['stage_number']:
                self.stats.end_condition = "MAX_ITERATIONS"

        return should_continue

    def create_graph(self):
        graph = StateGraph(self.game_state)

        # Add nodes using the factory-created node functions
        for node_name, node_func in self.nodes.items():
            graph.add_node(node_name, node_func)

        # Add edges
        graph.add_edge(START, 'supervisor')

        # Add edges from supervisor to each player
        for player_id in self.players:
            graph.add_edge('supervisor', f'player{player_id}')

        # Add edges from each player to aggregator
        for player_id in self.players:
            graph.add_edge(f'player{player_id}', 'aggregator')

        graph.add_conditional_edges(
            'aggregator',
            self._decision_node,
            {
                True: 'supervisor',
                False: END
            }
        )

        self.graph = graph.compile()

    def run_game(self):
        if self.graph is None:
            raise ValueError("Graph not created. Call create_graph() first.")

        for chunk in self.graph.stream(self.state_parameters, stream_mode='values'):
            print(chunk)

    def get_game_statistics(self) -> Dict:
        """Return computed statistics about the game"""
        if not self.stats.game_completed:
            print("Warning: Getting statistics for incomplete game")
        return self.stats.compute_summary()

    def plot_game_statistics(self):
        """Create visualizations of game statistics"""
        if not self.stats.game_completed:
            print("Cannot plot statistics for incomplete game")
            return

        try:
            import matplotlib.pyplot as plt

            # Create figure and axis
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot 1: Player decisions over time
            for player_id, claims in self.stats.player_claims.items():
                player = self.players[player_id]
                ax1.plot(range(1, len(claims) + 1), claims,
                        label=f'{player.name}',
                        marker='o')

            ax1.set_title('Player Decisions by Round')
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Decision Amount')
            ax1.legend()
            ax1.grid(True)

            # Plot 2: Remaining pot over time
            remaining_pot = [self.initial_pot]
            for round_num in range(len(list(self.stats.player_claims.values())[0])):
                round_claims = sum(player_claims[round_num]
                                 for player_claims in self.stats.player_claims.values())
                remaining_pot.append(remaining_pot[-1] - round_claims)

            ax2.plot(range(len(remaining_pot)), remaining_pot,
                    label='Remaining Pot', marker='s', color='green')
            ax2.set_title(f'Pot Depletion\nEnd Condition: {self.stats.end_condition}')
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Pot Amount')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("matplotlib is required for plotting. Please install it first.")