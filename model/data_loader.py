import torch
import nltk
import numpy as np
from pycocotools.coco import COCO
from torchvision.transforms import transforms
from torch.utils import data
from vocab.make_vocab import Make_vocab, Vocab
import pickle
import os
from utils import load_pickle


class ImageCaption_TrainDataLoader(data.Dataset):
    PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
    CLS_TOKEN = "[CLS]"  # leading token of the joint sequence
    SEP_TOKEN = "[SEP]"  # a separator for video and text
    IMG_TOKEN = "[IMG]"  # used as placeholder in the clip+text joint sequence
    BOS_TOKEN = "[BOS]"  # beginning of the sentence
    EOS_TOKEN = "[EOS]"  # ending of the sentence
    UNK_TOKEN = "[UNK]"  # UnKnown words

    PAD = 0
    CLS = 1
    SEP = 2
    IMG = 3
    BOS = 4
    EOS = 5
    UNK = 6
    IGNORE = -1  # used to calculate loss

    def __init__(self, config):
        self.config = config
        self.is_train = self.config.is_train
        # with open(self.config.vocab_path, 'rb') as f:
        #     vocab = pickle.load(f)
        self.vocab = load_pickle(self.config.vocab_path)

        # with open(self.config.visual_feature_path, 'rb') as f:
        #     img_feautre = pickle.load(f)
        self.img_feautre = load_pickle(self.config.visual_feature_path)

        self.coco = COCO(self.config.annotations_path)
        self.coco_ids = list(self.coco.anns.keys())
        self.ids = list(np.load(self.config.coco_idx_path))

    def __getitem__(self, idx):
        data = self.coco.anns[self.ids[idx]]
        captions = data['caption']
        img_id = str(data['image_id'])
        img_id = '0' * (12 - len(img_id)) + img_id

        img_feat = self.img_feautre[img_id]
        img_feature, img_tokens, img_mask = self.convert_img_feature(img_feat)
        sen_tokens, sen_mask, captions_tokens, captions_mask = self.convert_sentence_feature(str(captions).lower(), self.config.max_sub_len)

        img_sen_tokens = img_tokens + sen_tokens

        img_sen_input_ids = [self.vocab(tokens) for tokens in img_sen_tokens]
        img_sen_mask = img_mask + sen_mask

        captions_input_ids = [self.vocab(tokens) for tokens in captions_tokens]

        captions_label = [self.IGNORE if m == 0 else token
                         for token, m, in zip(captions_input_ids, captions_mask)][1:] + [self.IGNORE]

        data = dict(
            img_id=img_id,
            img_feature=np.array(img_feat),
            img_mask=np.array(img_mask).astype(np.float32),
            img_sen_input_ids=np.array(img_sen_input_ids).astype(np.int64),
            img_sen_mask=np.array(img_sen_mask).astype(np.float32),
            captions_input_ids=np.array(captions_input_ids).astype(np.int64),
            captions_mask=np.array(captions_mask).astype(np.float32).astype(np.int64),
            captions_label=np.array(captions_label).astype(np.int64),
        )

        return data

    def __len__(self):
        return len(self.ids)

    def convert_img_feature(self, img_feature):
        Img_tokens = [self.CLS_TOKEN] + [self.IMG_TOKEN] + [self.SEP_TOKEN]
        feat = np.zeros((3, img_feature.shape[0]))
        feat[1] = img_feature.cpu()
        mask = [1] * 3

        return feat, Img_tokens, mask

    def convert_sentence_feature(self, sentence, max_sen_len):
        sentence_tokens = nltk.tokenize.word_tokenize(sentence)[:max_sen_len - 2]
        sentence_tokens = [self.BOS_TOKEN] + sentence_tokens + [self.EOS_TOKEN]

        vaild_len = len(sentence_tokens)
        mask = [1] * vaild_len + [0] * (max_sen_len - vaild_len)
        sentence_tokens += [self.PAD_TOKEN] * (max_sen_len - vaild_len)

        caption_mask = mask + [0] * 3
        caption_tokens = sentence_tokens + [self.PAD_TOKEN] * 3

        return sentence_tokens, mask, caption_tokens, caption_mask


class ImageCaption_EvalDataLoader(data.Dataset):
    PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
    CLS_TOKEN = "[CLS]"  # leading token of the joint sequence
    SEP_TOKEN = "[SEP]"  # a separator for video and text
    IMG_TOKEN = "[IMG]"  # used as placeholder in the clip+text joint sequence
    BOS_TOKEN = "[BOS]"  # beginning of the sentence
    EOS_TOKEN = "[EOS]"  # ending of the sentence
    UNK_TOKEN = "[UNK]"  # UnKnown words

    PAD = 0
    CLS = 1
    SEP = 2
    IMG = 3
    BOS = 4
    EOS = 5
    UNK = 6
    IGNORE = -1  # used to calculate loss

    def __init__(self, config):
        self.config = config
        self.vocab = load_pickle(self.config.vocab_path)
        self.img_feautre = load_pickle(self.config.test_visual_feature_path)
        self.ids = list(self.img_feautre.keys())

    def __getitem__(self, idx):
        img_feat = self.img_feautre[self.ids[idx]]
        img_feature, img_tokens, img_mask = self.convert_img_feature(img_feat)
        sen_tokens, sen_mask = [self.PAD_TOKEN] * (self.config.max_sub_len + 3), [0] * (self.config.max_sub_len + 3)

        img_sen_tokens = img_tokens + sen_tokens

        img_sen_input_ids = [self.vocab(tokens) for tokens in img_sen_tokens]
        img_sen_mask = img_mask + sen_mask

        data = dict(
            img_id=self.ids[idx],
            img_feature=np.array(img_feat),
            img_mask=np.array(img_mask).astype(np.float32),
            img_sen_input_ids=np.array(img_sen_input_ids).astype(np.int64),
            img_sen_mask=np.array(img_sen_mask).astype(np.float32),
        )

        return data

    def __len__(self):
        return len(self.ids)

    def convert_ids_to_sentence(self, ids):
        return [self.vocab.idx2word[wid] for wid in ids if wid not in [self.PAD, self.IGNORE]]

    def convert_img_feature(self, img_feature):
        Img_tokens = [self.CLS_TOKEN] + [self.IMG_TOKEN] + [self.SEP_TOKEN]
        feat = np.zeros((3, img_feature.shape[0]))
        feat[1] = img_feature.cpu()
        mask = [1] * 3

        return feat, Img_tokens, mask

    def convert_sentence_feature(self, sentence, max_sen_len):
        sentence_tokens = nltk.tokenize.word_tokenize(sentence)[:max_sen_len - 2]
        sentence_tokens = [self.BOS_TOKEN] + sentence_tokens + [self.EOS_TOKEN]

        vaild_len = len(sentence_tokens)
        mask = [1] * vaild_len + [0] * (max_sen_len - vaild_len)
        sentence_tokens += [self.PAD_TOKEN] * (max_sen_len - vaild_len)

        caption_mask = mask + [0] * 3
        caption_tokens = sentence_tokens + [self.PAD_TOKEN] * 3

        return sentence_tokens, mask, caption_tokens, caption_mask


def collate_fn(data):
    for i in data:
        for (key, value) in i.items():
            print("key : {}, value : {}, shape : {}".format(key, value, value.shape))
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)
    lengths = [len(caption) for caption in captions]
    target = torch.zeros((len(captions), max(lengths))).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        target[i, :end] = cap[:end]

    return images, target, lengths


def prepare_batch_input(batch, device):
    batch_input = dict()
    for k, v in batch.items():
        if k == 'img_id':
            batch_input[k] = v
        else:
            batch_input[k] = v.to(device)

    return batch_input


def get_dataloader(config):
    CocoData = ImageCaption_TrainDataLoader(config)
    TrainDataloader = torch.utils.data.DataLoader(dataset=CocoData,
                                             batch_size=config.batch_size,
                                             shuffle=config.shuffle,
                                             num_workers=config.num_workers,
                                             drop_last=False)

    return TrainDataloader


def get_eval_dataloader(config):
    CocoData = ImageCaption_EvalDataLoader(config)
    EvalDataloader = torch.utils.data.DataLoader(dataset=CocoData,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             drop_last=False)

    return EvalDataloader
