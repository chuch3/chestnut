"""

Monte Carlo Tree Search (MCTS) with Upper Confidence Bound (UCB) Algorithm for Self-play

Pseudocode Reference : [link](https://ai-boson.github.io/mcts/)

Documentation reference : [link](https://www.reddit.com/r/reinforcementlearning/comments/cc5mv4/how_to_incorporate_neural_networks_into_a_mcts/)

"""

import unittest
from collections import defaultdict

import chess
import numpy as np


class MCTSNode:
    def __init__(
        self,
        state: chess.Board,
        parent=None,
        parent_action=None,
        root_turn=None,
    ):
        self.state = state
        self.parent = parent
        self.root_turn = root_turn
        self.parent_action = parent_action
        self.children: list(MCTSNode) = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0  # Number of wins
        self._results[-1] = 0  # Number of losses
        self._untried_actions = list(self.state.legal_moves)

    def get_n_visits(self):
        return self._number_of_visits

    def get_rewards(self):
        wins = self._results[1]
        losses = self._results[-1]
        return wins - losses

    def expand(self):
        """
        Expansion step:
        Randomly selects a non-tested legal move.
        """

        action = self._untried_actions.pop()

        next_state = self.state.copy()
        # print(f"Inside EXPAND, self\n{self.state}")
        next_state.push(action)
        # print(f"Inside EXPAND, next_state\n{next_state}")

        child_node = MCTSNode(
            next_state, parent=self, parent_action=action, root_turn=self.root_turn
        )
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state.copy()  # Avoid mutating the MCTS tree while experimenting in another sandbox copy tree
        while not current_rollout_state.is_game_over():
            possible_moves = list(current_rollout_state.legal_moves)

            action = self.rollout_policy(possible_moves)
            current_rollout_state.push(action)
        return self.game_result(current_rollout_state)

    def rollout_policy(self, possible_moves):
        """
        Light playout policy:
        Retuns randomly selected move out of the set of possible moves.
        """

        return possible_moves[np.random.randint(len(possible_moves))]

    def backpropagate(self, result):
        """
        Backpropagation step:

        """
        self._number_of_visits += 1.0
        self._results[result] += 1.0
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, exploration_rate=0.2):
        ucb_weights = [
            (child.get_rewards() / child.get_n_visits())
            + exploration_rate
            * np.sqrt((2 * np.log(self.get_n_visits()) / child.get_n_visits()))
            for child in self.children
        ]
        # print(f"For best child : {ucb_weights=}")
        return self.children[np.argmax(ucb_weights)]

    # NOTE :
    # If its fully expanded, returns the best children
    # then, that best children will be the terminal node?

    def _tree_policy(self):
        """
        Selection step:
        Traversing the tree to the leaf node based on the exploration / exploitation UCB algorithm.
        """

        current = self
        while not current.is_terminal_node():
            # print(f"Current Tree Policy {current}")
            # print(f"Expanded Tree Policy : {current.is_fully_expanded()}")
            if not current.is_fully_expanded():
                return current.expand()
            else:
                current = current.best_child()
        return current

    def best_action(self, num_simulations=100):
        """
        Main steps of entire MCTS tree loop to find the next best action.

        Steps:
        1. Selection step (`MCTSNode._tree_policy()`)
        2. Expansion step (`MCTSNode.expand()`)
        3. Rollout or Simulation step (`MCTSNode.rollout()`0
        4. Backpropagation (`MCTSNode.backpropagate()`)

        """

        for _ in range(num_simulations):
            node = self._tree_policy()
            # print(f"Node Tree Policy : {node}")
            reward = node.rollout()
            node.backpropagate(reward)

        return self.best_child(
            exploration_rate=0.0
        )  # Directly selects the best child (exploitation)

    def game_result(self, state=None):
        # If the state is given, the values is returned based on short-circuit evaluation.
        state = state or self.state
        outcome = state.outcome()

        if outcome is None:
            # print("Outcome is still ongoing!!!")
            return None

        if outcome.termination == chess.Termination.CHECKMATE:
            result = 1 if outcome.winner == self.root_turn else -1
            # print(f"{self.root_turn} player wins!")
            return result

        draw_termination = {
            chess.Termination.STALEMATE,
            chess.Termination.INSUFFICIENT_MATERIAL,
            chess.Termination.THREEFOLD_REPETITION,
            chess.Termination.FIFTY_MOVES,
            chess.Termination.SEVENTYFIVE_MOVES,
        }

        if outcome.termination in draw_termination:
            # print(f"Draw as {outcome.termination.name.lower().replace('_', ' ')}!")
            return 0
        return 0


class TestMCTSNode(unittest.TestCase):
    def test_best_action(self):
        initial_state = chess.Board()
        root = MCTSNode(state=initial_state, root_turn=initial_state.turn)
        selected_node = root.best_action()

        print(f"Best selected node : \n{selected_node.state}\n")
        # print(f"Final reward of selected node : {selected_node.game_result()}")


if __name__ == "__main__":
    unittest.main()
