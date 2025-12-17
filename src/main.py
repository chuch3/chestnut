import sys
import traceback

import chess.svg

from utils import arg_logs, board_to_matrix


def main():
    board = chess.Board()
    board_to_matrix(board)


if __name__ == "__main__":
    try:
        arg_logs()
        main()
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted\n")
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)
