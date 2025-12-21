import sys
import unittest

import chess
import chess.svg
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import (
    QApplication,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from predict import predict_best_move, start_chess_model

# Source - https://stackoverflow.com/a/61439875
# Posted by SmartKittie, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-20, License - CC BY-SA 4.0


class MainWindow(QWidget):
    def __init__(self, engine):
        super().__init__()

        self.is_white = self.ask_player_color()
        self.engine = engine
        self.engine.is_white = self.is_white

        self.setWindowTitle("Chestnut 🌰")
        self.resize(500, 500)

        layout = QVBoxLayout(self)

        self.widgetSvg = QSvgWidget()
        layout.addWidget(self.widgetSvg, stretch=1)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Enter move in UCI format (e.g. e2e4, a2a3q) : ")
        layout.addWidget(self.input)

        self.update_board()

        self.input.returnPressed.connect(self.on_user_move)

    def update_board(self):
        svg = chess.svg.board(
            self.engine.board, flipped=not self.engine.is_white
        ).encode("UTF-8")
        self.widgetSvg.load(svg)

    def check_game_over(self):
        outcome = self.engine.board.outcome()
        if outcome is None:
            return

        if outcome.termination == chess.Termination.CHECKMATE:
            winner = "White" if outcome.winner else "Black"
            QMessageBox.information(self, "Checkmate", f"Checkmate! {winner} wins.")
            QApplication.quit()
            return

        if outcome.termination in {
            chess.Termination.STALEMATE,
            chess.Termination.INSUFFICIENT_MATERIAL,
            chess.Termination.THREEFOLD_REPETITION,
            chess.Termination.FIFTY_MOVES,
            chess.Termination.SEVENTYFIVE_MOVES,
        }:
            QMessageBox.information(
                self,
                "Draw",
                f"Draw by {outcome.termination.name.replace('_', ' ').lower()}.",
            )
            QApplication.quit()

    def ask_player_color(self):
        reply = QMessageBox.question(
            self,
            "Choose side",
            "Do you want to play as White?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        return reply == QMessageBox.Yes

    def on_user_move(self):
        move = self.input.text().strip()
        self.input.clear()

        try:
            self.engine.board.push_san(move)
        except Exception:
            QMessageBox.warning(self, "Invalid move!", f'"{move}" is an illegal move!')
            return

        self.check_game_over()
        if self.engine.board.is_game_over():
            return
        self.engine.move()
        self.update_board()


class ChestnutWidget:
    def __init__(self):
        self.board = chess.Board()
        self.is_white = True
        self.app = QApplication(sys.argv)
        self.window = MainWindow(self)
        self.model, self.idx_move_map = start_chess_model()

    def move(self):
        best_move = predict_best_move(self.board, self.model, self.idx_move_map)
        self.board.push_uci(best_move)

    def run(self):
        self.window.show()
        if not self.is_white:
            self.move()
            self.window.update_board()
        sys.exit(self.app.exec())


class TestQTWidget(unittest.TestCase):
    def test_chestnut_widget(self):
        chestnut = ChestnutWidget()
        chestnut.run()


if __name__ == "__main__":
    unittest.main()
