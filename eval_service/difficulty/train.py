"""
eval_service/difficulty/train.py — Train the difficulty model.

Usage:
    python eval_service/difficulty/train.py
    python eval_service/difficulty/train.py --students 800 --seed 42

Saves model artifact to eval_service/difficulty/artifacts/difficulty_model.joblib.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval_service.difficulty.trainer import train


def main():
    ap = argparse.ArgumentParser(description="Train difficulty model")
    ap.add_argument("--students",  type=int,   default=400)
    ap.add_argument("--seed",      type=int,   default=7)
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    print("=" * 55)
    print("Difficulty Model — Training")
    print("=" * 55)

    result = train(n_students=args.students, seed=args.seed, test_frac=args.test_frac)

    print("\n" + "=" * 55)
    print("RESULT:", "PASS — model ready" if result["passed_gate"] else "FAIL — gates not met")
    print("=" * 55)
    sys.exit(0 if result["passed_gate"] else 1)


if __name__ == "__main__":
    main()
