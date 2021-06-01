import torch
from model.data_loader import get_dataloader, prepare_batch_input
from model.model import BertCaptioning
from utils import load_pickle, start_time, mkdirp, write_log, get_logger
from vocab.make_vocab import Make_vocab, Vocab
from config import BasicOption
import argparse
import torch.nn as nn
import logging
import torch.optim as optim
import pickle
import os
from tqdm import tqdm

logger = get_logger()

class Bert_Captioning:
    def __init__(self):
        self.config = BasicOption().parse()
        self.vocab = load_pickle(self.config.vocab_path)
        self.config.n_gpu = torch.cuda.device_count()
        self.config.vocab_size = len(self.vocab)
        self.DataLoader = get_dataloader(self.config)
        self.Model = BertCaptioning(self.config, len(self.vocab))

    def translate(self, output, batch):
        outputs, inputs, targets, img_ids = output.cpu(), batch['captions_input_ids'].cpu(), batch['captions_label'].cpu(), batch['img_id']
        translate = ""

        batch_predict = []
        batch_label = []

        for batch_idx, (predict, input, target, img_id) in enumerate(zip(outputs, inputs, targets, img_ids)):
            _, predict = predict.max(dim=1)

            """
                Print the result before [EOS] token.
            """

            predict = [self.vocab.idx2word[idx] for idx in predict.tolist()]
            # input = [self.vocab.idx2word[idx] for idx in input.tolist()]
            target = [self.vocab.idx2word[idx] for idx in target.tolist() if idx != -1]

            # predict, input = self.clean_text(predict), self.clean_text(input)
            predict = self.clean_text(predict)
            batch_predict.append(predict)
            batch_label.append(target)
            translate = "[Image id : {}] \n predict : {} \n target : {}\n".format(img_id, ' '.join(predict), ' '.join(target))

        return translate, batch_predict, batch_label

    def clean_text(self, input):
        result = []
        for word in input:
            if word == '[EOS]':
                result.append(word)
                break
            result.append(word)

        return result

    def cal_performance(self, output, target):
        total_words = 0
        total_correct_words = 0

        for (p, t) in zip(output, target):
            correct = 0
            vaild_len = min(len(p), len(t))
            for i in range(vaild_len):
                if p[i] == t[i]:
                    correct += 1

            total_words += vaild_len
            total_correct_words += correct

        return total_words, total_correct_words

    def train_epoch(self, epoch, dataloader, optimizer, model, filename):
        translate = ""
        epoch_loss = 0.0
        total_words = 0
        total_correct_words = 0

        model.train()
        for batch_idx, batch in tqdm(enumerate(dataloader), desc=" Training =>", total=len(dataloader)):
            optimizer.zero_grad()

            batch = prepare_batch_input(batch, self.config.device)
            output, loss = model(batch)

            if self.config.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            translate, batch_output, batch_label = self.translate(output, batch)
            words, correct_words = self.cal_performance(batch_output, batch_label)
            total_words += words
            total_correct_words += correct_words

        epoch_result = "[Train] Epoch : [{}/{}]\t Loss : {:.4f}\t Acc : {:.4f}".format(epoch, self.config.epochs, epoch_loss / len(self.DataLoader), (total_correct_words / total_words) * 100)

        result = epoch_result + '\n' + translate + '\n' + '-' * 100
        logger.info(result)

        write_log(filename, result)

    def train(self):
        mkdirp(self.config.result_path)
        result_path = self.config.result_path + '/' + start_time()
        mkdirp(result_path)
        filename = result_path + '/' + 'train-log.txt'

        if self.config.n_gpu > 1:
            self.Model = nn.DataParallel(self.Model)
        self.Model.cuda()
        logger.info("Using {} GPU ".format(torch.cuda.device_count()))

        optimizer = optim.Adam(self.Model.parameters(), lr=self.config.lr)

        logger.info("Now Training..")
        self.Model.train()
        for epoch in range(self.config.epochs):
            self.train_epoch(epoch, self.DataLoader, optimizer, self.Model, filename)

        checkpoint = {
            "model": self.Model.state_dict(),
            "config": self.config,
            "epoch": self.config.epochs
        }

        model_name = result_path + '/' + 'model-{}.ckpt'.format(self.config.epochs)
        torch.save(checkpoint, model_name)


if __name__ == '__main__':
    b = Bert_Captioning()
    b.train()
