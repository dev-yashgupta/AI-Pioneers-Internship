from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import download_and_extract

if __name__ == "__main__":
    print("Dataset ready:", download_and_extract(ROOT / "data" / "raw"))
