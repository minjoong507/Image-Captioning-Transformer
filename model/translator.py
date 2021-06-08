import torch.nn as nn
import torch
import logging
import os
from model.model import BertCaptioning
from model.data_loader import get_eval_dataloader, prepare_batch_input
from utils import load_pickle, get_logger, mkdirp, start_time
from tqdm import tqdm
from vocab.make_vocab import Vocab

logger = get_logger()


class Translator(object):
    def __init__(self, config, checkpoint, model=None):
        self.config = config
        self.model_cfg = checkpoint["config"]
        self.device = torch.device("cuda:{}".format(self.config.device) if self.config.device >= 0 else "cpu")
        self.vocab = load_pickle(self.model_cfg.vocab_path)
        self.model = model

        if model is None:
            self.model = BertCaptioning(checkpoint["config"], len(self.vocab)).to(self.device)
            self.model.load_state_dict(checkpoint["model"])
            logger.info("Loading pre-trained model.")

        self.model.eval()

    def translate_batch(self, model, inputs):

        enc_output = model.encode(inputs)

        # initialize input
        text_input_ids = torch.zeros((self.config.batch_size, self.model_cfg.max_sub_len + 3)).to(self.device).long()
        text_input_mask = torch.zeros((self.config.batch_size, self.model_cfg.max_sub_len + 3)).to(self.device).long()
        next_symbols = torch.LongTensor([self.vocab.word2idx[self.vocab.BOS_TOKEN]] * self.config.batch_size).to(self.device)

        for step in range(self.config.max_sub_len):
            # For first step, first token of input will be EOS token.
            text_input_ids[:, step] = next_symbols
            text_input_mask[:, step] = 1

            output = model.decode_for_eval(inputs, text_input_ids, text_input_mask, enc_output)

            next_words = output[:, step].max(1)[1]
            next_symbols = next_words

        return text_input_ids

    def run_translate(self, inputs):

        return self.translate_batch(self.model, inputs)
