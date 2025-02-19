from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Literal
import numpy as np

@dataclass
class GameStatistics:
    rounds_played: int = 0
    total_claimed: int = 0
    player_claims: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))
    final_pot: int = 0
    game_completed: bool = False
    end_condition: Literal["POT_DEPLETED", "MAX_ITERATIONS", "ONGOING"] = "ONGOING"

    def compute_summary(self) -> Dict:
        """Compute summary statistics for the game"""
        if not self.game_completed:
            return {"status": "Game not completed"}

        return {
            "rounds_played": self.rounds_played,
            "total_claimed": self.total_claimed,
            "end_condition": self.end_condition,
            "average_claimed_per_round": self.total_claimed / self.rounds_played if self.rounds_played > 0 else 0,
            "player_statistics": {
                player_id: {
                    "total_claimed": sum(claims),
                    "average_claim": np.mean(claims),
                    "max_claim": max(claims),
                    "min_claim": min(claims),
                    "std_claim": np.std(claims) if len(claims) > 1 else 0,
                    "decisions_by_round": claims
                }
                for player_id, claims in self.player_claims.items()
            },
            "final_pot": self.final_pot,
            "resource_depletion_rate": (self.total_claimed / self.rounds_played) if self.rounds_played > 0 else 0
        }