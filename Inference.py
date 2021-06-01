import sys
from utils import get_logger
import argparse

logger = get_logger()


def inference():
    parser = argparse.ArgumentParser(description="Image Captioning Evaluation Script")
    parser.add_argument("--test_path", type=str, help="model path")
    args = parser.parse_args()


if __name__ == '__main__':
    inference()
