import chess.svg

import sys
import traceback

from utils import board_to_matrix, arg_logs


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
