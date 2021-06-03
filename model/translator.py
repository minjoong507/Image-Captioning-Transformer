import torch.nn as nn
import torch
import logging
from model.model import BertCaptioning
from utils import load_pickle, get_logger, mkdirp, start_time

logger = get_logger()


class Translator(object):
    def __init__(self, config, checkpoint, model=None):
        self.config = config
        self.device = config.device
        self.vocab = load_pickle(self.config.vocab_path)
        self.model = model

        if model is None:
            self.model = BertCaptioning(checkpoint['config'], len(self.vocab)).to(self.device)
            self.model.load_state_dict(checkpoint['model'])
            logger.info("Loading pre-trained model.")

    def translate(self):
        mkdirp(self.config.eval_path)
        result_path = self.config.eval_path + '/' + start_time()
        mkdirp(result_path)

        print(self.model)
