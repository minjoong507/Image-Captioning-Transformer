import torch
import nltk
import pickle
import numpy as np
import os
from tqdm import tqdm
from pycocotools.coco import COCO
from torchvision.transforms import transforms
from torch.utils import data
from PIL import Image
from config import BasicOption
from vocab.make_vocab import Make_vocab, Vocab


class DataLoader(data.Dataset):
    def __init__(self, config):
        self.config = config
        with open('../' + self.config.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        self.vocab = vocab
        self.coco = COCO('../' + self.config.annotations_dir)
        self.coco_ids = list(self.coco.anns.keys())

        if not os.path.isfile('../vocab/coco_idx.npy'):
            self.preprocess_idx()
        self.ids = list(np.load('../vocab/coco_idx.npy')) # self.preprocess_idx()

        self.transform = transforms.Compose([
            transforms.RandomCrop(self.config.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
            ])

    def __getitem__(self, idx):
        data = self.coco.anns[self.ids[idx]]
        img_id = str(data['image_id'])

        # load image data
        img_id = '0' * (12 - len(img_id)) + img_id
        img_path = '../' + self.config.img_dir + str(img_id) + '.jpg'
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return img_id, image

    def __len__(self):
        return len(self.ids)

    def preprocess_idx(self):
        T = transforms.ToTensor()
        preprocess = []
        for i in tqdm(range(len(self.coco_ids))):
            data = self.coco.anns[self.coco_ids[i]]
            img_id = str(data['image_id'])

            # load image data
            img_id = '0' * (12 - len(img_id)) + img_id
            img_path = '../' + self.config.img_dir + str(img_id) + '.jpg'
            image = Image.open(img_path).convert('RGB')
            image = T(image)

            if image.shape[1] < 224 or image.shape[2] < 224:
                continue
            else:
                preprocess.append(self.coco_ids[i])

        print('Saved coco idx file!')

        np.save('../vocab/coco_idx.npy', np.array(preprocess))

        return preprocess


def get_dataloader(config):
    CocoData = DataLoader(config)
    Dataloader = torch.utils.data.DataLoader(dataset=CocoData,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             drop_last=False)

    return Dataloader