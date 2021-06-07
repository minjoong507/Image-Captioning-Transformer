import sys
import os
from utils import get_logger, mkdirp, start_time, load_pickle
import argparse
from pycocotools.coco import COCO
from model.data_loader import get_dataloader, prepare_batch_input
from model.model import BertCaptioning

import torch
from tqdm import tqdm
from model.translator import Translator
import time
from vocab.make_vocab import Vocab


logger = get_logger()


def inference():
    parser = argparse.ArgumentParser(description="Image Captioning Evaluation")
    parser.add_argument('--vocab_path', default='data/vocab.pickle', type=str)
    parser.add_argument('--img_path', default='data/test2017/', type=str)
    parser.add_argument('--test_visual_feature_path', default='data/visual_feature_test.pickle', type=str)
    parser.add_argument("--test_path", type=str, help="model path")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--is_train', type=str, default=False)
    parser.add_argument('--max_sub_len', type=int, default=30)
    parser.add_argument('--eval_coco_idx_path', default='data/test_coco_idx.npy', type=str)
    parser.add_argument("--eval_path", default='eval/', type=str, help="evaluation result path")
    parser.add_argument("--shuffle", default='False', type=str)
    parser.add_argument('--device', type=int, default=0)
    # parser.add_argument('--max_sub_len', type=int, default=30)

    args = parser.parse_args()
    print(args.test_path)

    checkpoint = torch.load(os.path.join(args.test_path, 'model.ckpt'))
    eval_loafder = get_dataloader(checkpoint['config'])
    # a = BertCaptioning(checkpoint["config"], checkpoint)
    # a.load_state_dict(torch.load(checkpoint["model"]))
    translator = Translator(args, checkpoint)
    print(translator.model)
    # translate(args, translator, eval_loader)


def translate(config, translator, dataloader):
    mkdirp(config.eval_path)
    result_path = os.path.join(config.eval_path, start_time())
    mkdirp(result_path)
    print(translator.model)
    # with torch.no_grad():
    #     for batch_idx, batch in tqdm(enumerate(eval_loader), desc=" Testing =>", total=len(eval_loader)):
    #         batch = prepare_batch_input(batch, config.device)
    #
    #         dec_output = translator.run_translate(batch)
    #
    #         for image_id, cur_caption in zip(batch['img_id'], dec_output):
    #             cur_data = {
    #                 'image_id': image_id,
    #                 'caption': cur_caption
    #
    #             }


if __name__ == '__main__':
    inference()

