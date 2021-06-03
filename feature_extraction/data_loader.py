import torch
import nltk
import pickle
import numpy as np
import os
from tqdm import tqdm
from pycocotools.coco import COCO
from torchvision.transforms import transforms
from torch.utils import data
from utils import get_logger
from PIL import Image
# from vocab.make_vocab import Make_vocab, Vocab

logger = get_logger()


class DataLoader(data.Dataset):
    def __init__(self, config):
        self.config = config
        # with open(self.config.vocab_path, 'rb') as f:
        #     vocab = pickle.load(f)
        # self.vocab = vocab
        self.coco = COCO(self.config.annotations_path)
        self.coco_ids = list(self.coco.anns.keys())

        if not os.path.isfile(self.config.coco_idx_path):
            self.preprocess_idx()
        self.ids = list(np.load(self.config.coco_idx_path))

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
        img_path = os.path.join(self.config.img_path, str(img_id) + '.jpg')
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return img_id, image

    def __len__(self):
        return len(self.ids)

    def preprocess_idx(self):
        T = transforms.ToTensor()
        preprocess = []
        for i in tqdm(range(len(self.coco_ids)), desc=' Make coco keys', total=len(self.coco_ids)):
            data = self.coco.anns[self.coco_ids[i]]
            img_id = str(data['image_id'])

            # load image data
            img_id = '0' * (12 - len(img_id)) + img_id
            img_path = os.path.join(self.config.img_path, str(img_id) + '.jpg')
            image = Image.open(img_path).convert('RGB')
            image = T(image)

            if image.shape[1] < 224 or image.shape[2] < 224:
                continue
            else:
                preprocess.append(self.coco_ids[i])

        logger.info('Saved coco idx file!')

        np.save(self.config.coco_idx_path, np.array(preprocess))

        return preprocess


def get_dataloader(config):
    CocoData = DataLoader(config)
    Dataloader = torch.utils.data.DataLoader(dataset=CocoData,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             drop_last=False)

    return Dataloader