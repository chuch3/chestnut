import sys
import traceback

from widget import ChestnutWidget


def main():
    try:
        chestnut = ChestnutWidget()
        chestnut.run()
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted\n")
    except Exception:
        traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    main()
