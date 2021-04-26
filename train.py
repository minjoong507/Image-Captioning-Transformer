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
        self.MultiGPU = False
        if torch.cuda.device_count() > 1:
            print("Using Multi GPU")
            logger.info("Using Multi GPU")
            self.MultiGPU = True
        self.config.vocab_size = len(self.vocab)
        self.DataLoader = get_dataloader(self.config)
        self.Model = BertCaptioning(self.config, len(self.vocab))

    def translate(self, output, batch):
        outputs, inputs, targets = output.cpu(), batch['captions_input_ids'].cpu(), batch['captions_label'].cpu()
        translate = ""

        for batch_idx, (predict, input, target) in enumerate(zip(outputs, inputs, targets)):
            _, predict = predict.max(dim=1)

            """
                Print the result before [EOS] token.
            """

            predict = [self.vocab.idx2word[idx] for idx in predict.tolist()]
            input = [self.vocab.idx2word[idx] for idx in input.tolist()]
            target = [self.vocab.idx2word[idx] for idx in target.tolist() if idx != -1]

            predict, input = self.remove_eos(predict), self.remove_eos(input)

            translate += ("[Result : {}] \n predict : {} \n input : {}\n target : {}\n".format(batch_idx, predict, input, ' '.join(target)))

        return translate, predict, target

    def remove_eos(self, input):
        result = ''
        for word in input:
            if word == '[EOS]':
                result += word + ' '
                break
            result += word + ' '

        return result

    def train(self):
        mkdirp(self.config.result_path)
        result_path = self.config.result_path + '/' + start_time()
        mkdirp(result_path)
        filename = result_path + '/' + 'train-log.txt'

        # self.Model.to(self.config.device)
        if self.MultiGPU:
            self.Model = nn.DataParallel(self.Model)
        self.Model.cuda()
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

        model_name = result_path + '/' + 'model-{}.ckpt'.format(self.config.epochs)
        torch.save(checkpoint, model_name)

    def cal_performance(self, predict, target):

        total_words = 0
        total_correct_words = 0

        for (p, t) in zip(predict, target):
            correct = 0
            vaild_len = min(len(p), len(t))
            for i in range(vaild_len):
                if p[i] == t[i]:
                    correct += 1

            total_words += vaild_len
            total_correct_words += correct

        return total_words, total_correct_words

    def train_epoch(self, epoch, dataloader, optimizer, model, filename):
        epoch_loss = 0.0
        last_batch = None
        last_output = None

        total_words = 0
        total_correct_words = 0

        for batch_idx, batch in tqdm(enumerate(dataloader), desc=" Training =>", total=len(dataloader)):
            optimizer.zero_grad()

            batch = prepare_batch_input(batch, self.config.device)
            output, loss = model(batch)

            if self.MultiGPU:
                loss.sum().backward()
                epoch_loss += loss.sum().item()
            else:
                loss.backward()
                epoch_loss += loss.item()

            optimizer.step()

            _, p, t = self.translate(output, batch)
            p = p.split(' ')
            words, correct_words = self.cal_performance(p, t)
            total_words += words
            total_correct_words += correct_words

            last_batch = batch
            last_output = output

        epoch_result = "Train Epoch : [{}/{}]\t Loss : {:.4f}".format(epoch, self.config.epochs, epoch_loss / len(self.DataLoader))
        correct_result = "Train Acc : {:.4f}".format((total_correct_words / total_words) * 100)
        translate, _, _ = self.translate(last_output, last_batch)

        result = epoch_result + '\n' + translate + '\n' + correct_result + '\n'
        print(result)

        write_log(filename, result)


if __name__ == '__main__':
    b = Bert_Captioning()
    b.train()
