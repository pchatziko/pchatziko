import numpy as np
from typing import Optional
from pydantic import BaseModel, Field
import random

sim_features_dtype = np.dtype(
    [
        ("serve_pct", np.float32, 2),   # t1, t2
        ("is_t1_serving", np.bool_),    
        ("is_match_over", np.bool_),
        ("serve_points", np.int32, 2),      # t1, t2
        ("serve_points_won", np.int32, 2),  # t1, t2
        ("score_current_set", np.int32, 2), # t1, t2
        ("score_match", np.int32, 2),    # t1, t2
        
    ]
)

sim_records_dtype = np.dtype(
    [
        ("set_scores", np.int32, (5, 2)),  # set, score
        ("match_score_points", np.int32, 2), # t1, t2
        ("t1_served_first", np.bool_)
    ]
)

class InputFeatures(BaseModel):
    t1_serve_pct: Optional[float] = Field(alias='T1Servepct')       
    t2_serve_pct: Optional[float] = Field(alias='T2Servepct')
    n0: Optional[int] = 170

class MatchDetails:
    def __init__(
            self,
            set_to_win: int = 3,
            set_first_to_points: int = 25
    ):
        self.set_to_win = set_to_win
        self.set_first_to_points = set_first_to_points


def simulate_match(input_features, sim_features, sim_records):
    while not sim_features["is_match_over"]:
        server_idx=0 if sim_features["is_t1_serving"] else 1
        server_won= simulate_point(server_idx=server_idx, input_features=input_features)

def simulate_point(server_idx, input_features):
    # simulate the point
    serve_pct=input_features["t1_serve_pct"][server_idx]
    prior_wins= serve_pct*input_features["n0"]
    prior_losses= (1-serve_pct)*input_features["n0"]
    actual_wins= sim_features_dtype["serve_points_won"][server_idx]
    actual_losses= sim_features_dtype["serve_points"][server_idx]-actual_wins
    return random()<np.random.beta(prior_wins+actual_wins, prior_losses+actual_losses)

    
 

def run_simulation(_):
    t1_serve_pct = 0.7
    t2_serve_pct = 0.6
    is_match_over = False
    is_t1_serving = random() < 0.5
    serve_points = [0, 0]
    serve_points_won = [0, 0]
    score_current_set = [0, 0]
    score_match = [0, 0]
    input_features=InputFeatures(T1Servepct=t1_serve_pct, T2Servepct=t2_serve_pct, n0=170)
    sim_features = np.array(t1_serve_pct, t2_serve_pct, is_t1_serving, is_match_over, serve_points, serve_points_won, score_current_set, score_match, dtype=sim_features_dtype)
    sim_records = np.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [0, 0], True], dtype=sim_records_dtype)
    simulate_match(input_features, sim_features, sim_records)
    return sim_features["score_match"]
