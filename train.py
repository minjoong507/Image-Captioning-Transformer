import torch
from model.data_loader import get_dataloader, prepare_batch_input
from model.model import BertCaptioning
from utils import load_pickle
from model.optimization import BertAdam
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


# device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
# torch.cuda.set_device(device)
# cuda = torch.device(device)
# print(cuda)


class Image_Captioning:
    def __init__(self):
        self.args = BasicOption().parse()
        self.vocab = load_pickle(self.args.vocab_path)
        self.args.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.args.device)
        print(self.args.device)
        self.args.vocab_size = len(self.vocab)
        self.DataLoader = get_dataloader(self.args)
        self.Model = BertCaptioning(self.args, len(self.vocab))

    def translate(self, output, batch):
        outputs, inputs, targets = output.cpu(), batch['captions_input_ids'].cpu(), batch['captions_label'].cpu()

        for predict, input, target in zip(outputs, inputs, targets):
            _, predict = predict.max(dim=1)

            predict = [self.vocab.idx2word[idx] for idx in predict.tolist()]
            input = [self.vocab.idx2word[idx] for idx in input.tolist()]
            target = [self.vocab.idx2word[idx] for idx in target.tolist() if idx != -1]

            print("predict : {} \n input : {}\n answer : {}\n".format(' '.join(predict),
                                                                      ' '.join(input),
                                                                      ' '.join(target)))

    def train(self):
        self.Model.to(self.args.device)
        # parameters = self.Model.parameters()
        # num_train_optimization_steps = len(self.DataLoader) * self.args.epochs
        # optimizer = BertAdam(parameters,
        #                      lr=self.args.lr,
        #                      t_total=num_train_optimization_steps,
        #                      warmup=self.args.lr_warmup_proportion,
        #                      schedule="warmup_linear")

        optimizer = optim.Adam(self.Model.parameters(), lr=self.args.lr)

        print("Now training")
        self.Model.train()
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch, self.DataLoader, optimizer, self.Model)
            # epoch_loss = 0.0
            # last_batch = None
            # last_output = None
            #
            # for batch_idx, batch in tqdm(enumerate(self.DataLoader)):
            #     optimizer.zero_grad()
            #
            #     batch = prepare_batch_input(batch, self.args.device)
            #     output, loss = self.Model(batch)
            #
            #     loss.backward()
            #     optimizer.step()
            #     epoch_loss += loss.item()
            #
            #     last_batch = batch
            #     last_output = output
            #
            # print("Epoch : [{}/{}]\t Loss : {:.4f}".format(epoch, self.args.epochs, epoch_loss / len(self.DataLoader)))
            # self.translate(last_output, last_batch)

    def train_epoch(self, epoch, dataloader, optimizer, model):
        epoch_loss = 0.0
        last_batch = None
        last_output = None

        for batch_idx, batch in tqdm(enumerate(dataloader), desc=" Training =>", total=len(dataloader)):
            optimizer.zero_grad()

            batch = prepare_batch_input(batch, self.args.device)
            output, loss = model(batch)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            last_batch = batch
            last_output = output

        print("Epoch : [{}/{}]\t Loss : {:.4f}".format(epoch, self.args.epochs, epoch_loss / len(self.DataLoader)))
        self.translate(last_output, last_batch)


if __name__ == '__main__':
    b = Image_Captioning()
    b.train()
