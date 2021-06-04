import torch.nn as nn
import torch
import logging
import os
from model.model import BertCaptioning
from model.data_loader import get_eval_dataloader, prepare_batch_input
from utils import load_pickle, get_logger, mkdirp, start_time
from tqdm import tqdm

logger = get_logger()


class Translator(object):
    def __init__(self, config, checkpoint, model=None):
        self.config = config
        self.device = config.device
        self.vocab = load_pickle(self.config.vocab_path)
        self.eval_loader = get_eval_dataloader(self.config)
        self.model = model

        if model is None:
            self.model = BertCaptioning(checkpoint['config'], len(self.vocab)).to(self.device)
            self.model.load_state_dict(checkpoint['model'])
            logger.info("Loading pre-trained model.")

    def translate_batch(self, model, inputs):
        enc_out = model.encode(inputs)

        # initialize input
        text_input_ids = torch.Tensor.new_zeros((self.config.batch_size, self.config.max_sub_len))
        text_input_mask = torch.Tensor.new_zeros((self.config.batch_size, self.config.max_sub_len)).float()

        next_symbols = torch.LongTensor([self.eval_loader.BOS_TOKEN] * self.config.batch_size)

        for step in range(self.config.max_sub_len):
            # For first step, first token of input will be EOS token.
            text_input_ids[:, step] = next_symbols
            text_input_mask[:, step] = 1

            output, _ = model.decode(inputs, enc_out)

            next_words = output[:, step].max(1)[1]
            next_symbols = next_words

        return text_input_ids

    def run_translate(self, inputs):

        return self.translate_batch(self.model, inputs)
