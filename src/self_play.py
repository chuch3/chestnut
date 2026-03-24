import pickle
import unittest
from collections import namedtuple
from multiprocessing import Pool, cpu_count

import chess
from tqdm import tqdm

from mcts import MCTSNode

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

    def run(self):
        memory = []

        with Pool(self.num_workers) as pool:
            results = tqdm(
                pool.imap_unordered(
                    play_single_game, [self.num_mcts_sim] * self.num_games
                ),
                total=self.num_games,
                desc="Number of Self-play MCTS Games",
            )

            for i, game in enumerate(results, start=1):
                memory.append(game)
                if i % 100 == 0 or i == self.num_games:
                    with open(f"save_play_{700 + i}_games.pickle", "wb") as f:
                        pickle.dump(memory, f)
                        print(f"Checkpoint saved after {i} games")

        memory = [move for game in memory for move in game]
        return memory


class TestSelfPlay(unittest.TestCase):
    @unittest.skip
    def test_load_memory(self):
        with open("save_play_1100_games.pickle", "rb") as file:
            memory1 = pickle.load(file)
            print("=> Loaded pickle object")

        memory1 = [move for game in memory1 for move in game]

        with open("SELF_PLAY_DATASET_150SIMULATIONS_700GAMES.pickle", "rb") as file:
            memory2 = pickle.load(file)
            print("=> Loaded pickle object")

        memory = memory2 + memory1

        file_name = f"SELF_PLAY_DATASET_{150}SIMULATIONS_{1100}GAMES.pickle"

        print(len(memory))
        with open(file_name, "wb") as file:
            pickle.dump(memory, file)
            print(f"=> Saved pickle object {file_name}")

    @unittest.skip
    def test_build_self_play_dataset(self):
        num_mcts_sim = 150
        num_games = 800
        file_name = (
            f"SELF_PLAY_DATASET_{num_mcts_sim}SIMULATIONS_{num_games}GAMES.pickle"
        )

        self_play = SelfPlay(num_mcts_sim=num_mcts_sim, num_games=num_games)
        memory = self_play.run()

        with open(file_name, "wb") as file:
            pickle.dump(memory, file)
            print(f"=> Saved pickle object {file_name}")


def main():
    pass


if __name__ == "__main__":
    unittest.main()
