from pycocotools.coco import COCO
import nltk
from collections import Counter
import pickle
from config import BasicOption
import argparse
from utils import save_pickle


class Vocab:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        self.word2idx[word] = self.idx
        self.idx2word[self.idx] = word
        self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['[UNK]']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class Make_vocab:
    def __init__(self, config):
        self.coco = COCO('../' + config.annotations_dir)
        self.ids = self.coco.anns.keys()
        self.Counter = Counter()
        self.min_fq = config.min_fq
        self.length = 0
        self.vocab = Vocab()

    def get_vocab(self):
        for i, id in enumerate(self.ids):
            caption = str(self.coco.anns[id]['caption'])
            tokens = nltk.word_tokenize(caption.lower())
            self.length = max(len(tokens), self.length)
            self.Counter.update(tokens)

            if (i + 1) % 1000 == 0:
                print("[{}/{}] Tokenized the captions.".format(i + 1, len(self.ids)))

        PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
        CLS_TOKEN = "[CLS]"  # leading token of the joint sequence
        SEP_TOKEN = "[SEP]"  # a separator for video and text
        IMG_TOKEN = "[IMG]"  # used as placeholder in the clip+text joint sequence
        BOS_TOKEN = "[BOS]"  # beginning of the sentence
        EOS_TOKEN = "[EOS]"  # ending of the sentence
        UNK_TOKEN = "[UNK]"
        PAD = 0
        CLS = 1
        SEP = 2
        IMG = 3
        BOS = 4
        EOS = 5
        UNK = 6

        self.vocab.add_word('[PAD]')
        self.vocab.add_word('[CLS]')
        self.vocab.add_word('[SEP]')
        self.vocab.add_word('[IMG]')
        self.vocab.add_word('[BOS]')
        self.vocab.add_word('[EOS]')
        self.vocab.add_word('[UNK]')

        words = [word for word, cnt in self.Counter.items() if cnt >= self.min_fq]

        for i, word in enumerate(words):
            self.vocab.add_word(word)

        return self.vocab


if __name__ == '__main__':
    config = BasicOption().parse()

    a = Make_vocab(config)
    vocab = a.get_vocab()
    save_pickle(vocab, 'vocab.pickle')


