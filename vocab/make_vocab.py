from pycocotools.coco import COCO
import nltk
from collections import Counter
import pickle
import argparse
from utils import save_pickle, get_logger
from tqdm import tqdm

logger = get_logger()


class Vocab:
    def __init__(self):
        self.PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
        self.CLS_TOKEN = "[CLS]"  # leading token of the joint sequence
        self.SEP_TOKEN = "[SEP]"  # a separator for video and text
        self.IMG_TOKEN = "[IMG]"  # used as placeholder in the clip+text joint sequence
        self.BOS_TOKEN = "[BOS]"  # beginning of the sentence
        self.EOS_TOKEN = "[EOS]"  # ending of the sentence
        self.UNK_TOKEN = "[UNK]"
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
        self.coco = COCO(config.annotations_path)
        self.ids = self.coco.anns.keys()
        self.Counter = Counter()
        self.min_fq = config.min_fq
        self.length = 0
        self.vocab = Vocab()

    def initialize_vocab(self):
        PAD_TOKEN = "[PAD]"  # padding of the whole sequence
        CLS_TOKEN = "[CLS]"  # leading token of the joint sequence
        SEP_TOKEN = "[SEP]"  # a separator for video and text
        IMG_TOKEN = "[IMG]"  # used as placeholder in the clip+text joint sequence
        BOS_TOKEN = "[BOS]"  # beginning of the sentence
        EOS_TOKEN = "[EOS]"  # ending of the sentence
        UNK_TOKEN = "[UNK]"

        self.vocab.add_word(PAD_TOKEN)
        self.vocab.add_word(CLS_TOKEN)
        self.vocab.add_word(SEP_TOKEN)
        self.vocab.add_word(IMG_TOKEN)
        self.vocab.add_word(BOS_TOKEN)
        self.vocab.add_word(EOS_TOKEN)
        self.vocab.add_word(UNK_TOKEN)


    def get_vocab(self):
        for i, id in tqdm(enumerate(self.ids), desc=' Build Vocab =>', total=len(self.ids)):
            caption = str(self.coco.anns[id]['caption'])
            tokens = nltk.word_tokenize(caption.lower())
            self.length = max(len(tokens), self.length)
            self.Counter.update(tokens)

        # Add speical tokens
        self.initialize_vocab()

        # Add words
        words = [word for word, cnt in self.Counter.items() if cnt >= self.min_fq]

        for i, word in enumerate(words):
            self.vocab.add_word(word)

        return self.vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build Vocab")
    parser.add_argument('--vocab_path', default='data/vocab.pickle', type=str)
    parser.add_argument('--annotations_path', default='data/annotations/captions_train2017.json', type=str)
    parser.add_argument('--min_fq', type=int, default=5)
    args = parser.parse_args()

    a = Make_vocab(args)
    vocab = a.get_vocab()
    save_pickle(vocab, args.vocab_path)
    logger.info('Save vocab file at {}.'.format(args.vocab_path))


