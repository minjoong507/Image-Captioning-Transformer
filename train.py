import torch
from model.data_loader import get_dataloader, prepare_batch_input
from model.model import BertCaptioning
from utils import load_pickle, start_time, mkdirp, write_log
from vocab.make_vocab import Make_vocab, Vocab
from config import BasicOption
import argparse
import torch.nn as nn
import logging
import torch.optim as optim
import pickle
import os
from tqdm import tqdm
logger = logging.getLogger(__name__)


class Bert_Captioning:
    def __init__(self):
        self.config = BasicOption().parse()
        self.vocab = load_pickle(self.config.vocab_path)
        self.config.device = torch.device('cuda:'.format(self.config.device) if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.config.device)
        self.config.vocab_size = len(self.vocab)
        self.DataLoader = get_dataloader(self.config)
        self.Model = BertCaptioning(self.config, len(self.vocab))

    def translate(self, output, batch):
        outputs, inputs, targets = output.cpu(), batch['captions_input_ids'].cpu(), batch['captions_label'].cpu()
        translate = ""

        for predict, input, target in zip(outputs, inputs, targets):
            _, predict = predict.max(dim=1)

            predict = [self.vocab.idx2word[idx] for idx in predict.tolist()]
            input = [self.vocab.idx2word[idx] for idx in input.tolist()]
            target = [self.vocab.idx2word[idx] for idx in target.tolist() if idx != -1]

            translate += ("predict : {} \n input : {}\n answer : {}\n".format(' '.join(predict),
                                                                              ' '.join(input),
                                                                              ' '.join(target)))
        return translate

    def train(self):
        mkdirp(self.config.result_path)
        result_path = self.config.result_path + '/' + start_time()
        mkdirp(result_path)
        filename = result_path + '/' + 'train-log.txt'

        self.Model.to(self.config.device)
        optimizer = optim.Adam(self.Model.parameters(), lr=self.config.lr)

        print("Now training")
        self.Model.train()
        for epoch in range(self.config.epochs):
            self.train_epoch(epoch, self.DataLoader, optimizer, self.Model, filename)

        checkpoint = {
            "model": self.Model.state_dict(),
            "model_cfg": self.Model.config,
            "config": self.config,
            "epoch": self.config.epochs
        }

        model_name = result_path + '/' + 'model.ckpt'
        torch.save(checkpoint, model_name)

    def train_epoch(self, epoch, dataloader, optimizer, model, filename):
        epoch_loss = 0.0
        last_batch = None
        last_output = None

        for batch_idx, batch in tqdm(enumerate(dataloader), desc=" Training =>", total=len(dataloader)):
            optimizer.zero_grad()

            batch = prepare_batch_input(batch, self.config.device)
            output, loss = model(batch)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            last_batch = batch
            last_output = output

        epoch_result = "Epoch : [{}/{}]\t Loss : {:.4f}".format(epoch, self.config.epochs, epoch_loss / len(self.DataLoader))
        translate = self.translate(last_output, last_batch)

        result = epoch_result + '\n' + translate + '\n'
        print(result)

        write_log(filename, result)


if __name__ == '__main__':
    b = Bert_Captioning()
    b.train()
