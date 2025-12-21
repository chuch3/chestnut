import os
import pickle
import unittest
from collections import namedtuple
from multiprocessing import Pool, cpu_count

import chess
from mcts import MCTSNode
from tqdm import tqdm

_MAP_NAME: str = "movemap.pickle"
_MAP_PATH: str = os.path.realpath(os.path.join("..", ".data", _MAP_NAME))

# NOTE:
# - So we just set the traning data to be the same using the move tensor, but
# expert data and supervised will have different training loss.
# - We also added a more generalized move map which has to pass to other functions (gotta find this)

MCTSData = namedtuple("MCTSData", ("state", "policy", "value"))


def play_single_game(num_mcts_sim):
    game_history = []
    board = chess.Board()
    # Initializing current node as root node (start of chess board).
    current = MCTSNode(state=board, root_turn=board.turn)

    while not board.is_game_over():
        next_best_child = current.best_action(num_simulations=num_mcts_sim)
        policy = compute_policy(current)

        game_history.append((board.copy(), policy))

        board.push(next_best_child.parent_action)
        current = next_best_child

        """
        print(f"Policy of selecting next best action (children) : {policy}\n")
        print(f"Next best action : {next_best_child.parent_action}")
        print()
        print(f"Current Board :\n{board}\n")
        print(f"Number of moves {len(game_history)}")
        """
    # Game result of the terminated node (game over).
    value = current.game_result()

    # print(f"Value of the entire game : {value}")
    return [
        MCTSData(state=state, policy=policy, value=value)
        for state, policy in game_history
    ]


def compute_policy(node):
    total_visits = sum([child._number_of_visits for child in node.children])
    test = [child._number_of_visits / total_visits for child in node.children]
    print(test)
    return [
        (child.parent_action.uci(), child._number_of_visits / total_visits)
        for child in node.children
    ]


class SelfPlay:
    def __init__(self, num_mcts_sim=100, num_workers=None, num_games=150) -> None:
        self.num_mcts_sim = num_mcts_sim
        self.num_workers = num_workers or cpu_count()
        self.num_games = num_games
        print("{----------------------------------------}")
        print()
        print(f"Number of CPUs used : {self.num_workers}")
        print(f"Number of MCTS Simulations per Game: {self.num_mcts_sim}")
        print()
        print("{----------------------------------------}")
        print()

    def build_self_play(self):
        memory = []
        with Pool(self.num_workers) as pool:
            memory = list(
                tqdm(
                    pool.imap_unordered(
                        play_single_game, [self.num_mcts_sim] * self.num_games
                    ),
                    total=self.num_games,
                    desc="Number of Self-play MCTS Games",
                )
            )

        memory = [move for game in memory for move in game]
        return memory


class TestSelfPlay(unittest.TestCase):
    def test_build_self_play_dataset(self):
        num_mcts_sim = 150
        num_games = 200
        file_name = (
            f"SELF_PLAY_DATASET_{num_mcts_sim}SIMULATIONS_{num_games}GAMES.pickle"
        )

        self_play = SelfPlay(num_mcts_sim=num_mcts_sim, num_games=num_games)
        memory = self_play.build_self_play()

        with open(file_name, "wb") as file:
            pickle.dump(memory, file)

        print(f"=> Saved pickle object {file_name}")


def main():
    pass


if __name__ == "__main__":
    unittest.main()
