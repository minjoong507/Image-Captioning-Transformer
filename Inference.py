import sys
import os
from utils import get_logger, mkdirp, start_time, load_pickle, save_jsonl
import argparse
from pycocotools.coco import COCO
from model.data_loader import get_dataloader, prepare_batch_input, get_eval_dataloader
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
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--is_train', type=str, default=False)
    parser.add_argument('--eval_coco_idx_path', default='data/test_coco_idx.npy', type=str)
    parser.add_argument("--eval_path", default='eval/', type=str, help="evaluation result path")
    parser.add_argument("--shuffle", default='False', type=str)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--max_sub_len', type=int, default=30)

    args = parser.parse_args()

    checkpoint = torch.load(os.path.join(args.test_path, 'model.ckpt'))
    eval_dataloader = get_eval_dataloader(args)
    translator = Translator(args, checkpoint)

    eval_result = translate(args, translator, eval_dataloader)

    mkdirp(args.eval_path)
    result_path = os.path.join(args.eval_path, start_time())
    mkdirp(result_path)

    filename = os.path.join(result_path, 'pred.jsonl')
    save_jsonl(eval_result, filename)
    logger.info("Save predict json file at {}".format(result_path))


def translate(config, translator, dataloader):
    logger.info("Now testing..")

    batch_result = []

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), desc=" Testing =>", total=len(dataloader)):
            batch = prepare_batch_input(batch, config.device)
            dec_output = translator.run_translate(batch)
            for image_id, cur_caption in zip(batch['img_id'], dec_output):
                cur_data = {
                    'image_id': image_id,
                    'caption': dataloader.dataset.convert_ids_to_sentence(cur_caption.cpu().tolist())
                }
                batch_result.append(cur_data)

    return batch_result


if __name__ == '__main__':
    inference()

