import src
from src import config

if __name__ == "__main__":
    try:
        src.main()

    except Exception as e:
        print(f"Application crashed {e}")
