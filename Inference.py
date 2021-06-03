import sys
import os
from utils import get_logger
from config import TestOption
import argparse
import torch
from model.translator import Translator

logger = get_logger()


def inference():
    parser = argparse.ArgumentParser(description="Image Captioning Evaluation")
    parser.add_argument('--vocab_path', default='vocab/vocab.pickle', type=str)
    parser.add_argument("--test_path", type=str, help="model path")
    parser.add_argument("--eval_path", default='eval', type=str, help="model path")
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()
    print(args.test_path)

    checkpoint = torch.load(os.path.join(args.test_path, 'model.ckpt'))

    Translator = Translator(args, )


if __name__ == '__main__':
    inference()
