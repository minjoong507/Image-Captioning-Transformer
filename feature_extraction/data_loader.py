import torch
import nltk
import numpy as np
import os
from tqdm import tqdm
from pycocotools.coco import COCO
from torchvision.transforms import transforms
from torch.utils import data
from utils import get_logger, load_json
from PIL import Image
# from vocab.make_vocab import Make_vocab, Vocab

logger = get_logger()


class TrainDataLoader(data.Dataset):
    def __init__(self, config):
        self.config = config
        self.coco = COCO(self.config.annotations_path)
        self.coco_ids = list(self.coco.anns.keys())

        if not os.path.isfile(self.config.coco_idx_path):
            logger.info("Now get coco image id for train")
            self.preprocess_idx()
        self.image_info = list(np.load(self.config.coco_idx_path))
        self.transform = transforms.Compose([
            transforms.RandomCrop(self.config.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
            ])

    def __getitem__(self, idx):
        data = self.coco.anns[self.image_info[idx]]
        img_id = str(data['image_id'])

        # load image data
        img_id = '0' * (12 - len(img_id)) + img_id
        img_path = os.path.join(self.config.img_path, str(img_id) + '.jpg')
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return img_id, image
        # image_id, image = self.image_info[idx]['image_id'], self.image_info[idx]['image_feat']
        # image = self.transform(image)
        #
        # return image_id, image

    def __len__(self):
        return len(self.image_info)

    def preprocess_idx(self):
        T = transforms.ToTensor()
        preprocess = []
        for i in tqdm(range(len(self.coco_ids)), desc=' Build coco dict for training', total=len(self.coco_ids)):
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

        logger.info('Saved train coco idx file!')

        np.save(self.config.coco_idx_path, np.array(preprocess))

        return preprocess


class EvalDataLoader(data.Dataset):
    def __init__(self, config):
        self.config = config
        self.test_image_dict = load_json(self.config.eval_annotations_path)

        if not os.path.isfile(self.config.eval_coco_idx_path):
            logger.info("Now get coco image id for eval")
            self.preprocess_idx()
        self.image_info = list(np.load(self.config.eval_coco_idx_path))

        self.transform = transforms.Compose([
            transforms.RandomCrop(self.config.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])

    def __getitem__(self, idx):
        image_id = str(self.image_info[idx])
        image_id = '0' * (12 - len(image_id)) + image_id
        img_path = os.path.join(self.config.eval_img_path, image_id + '.jpg')
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image_id, image

    def __len__(self):
        return len(self.image_info)

    def preprocess_idx(self):
        T = transforms.ToTensor()
        preprocess = []
        for i in tqdm(range(len(self.test_image_dict['images'])), desc=' Build coco dict for evaluation', total=len(self.test_image_dict['images'])):
            img_info = self.test_image_dict['images'][i]
            image_id, image_filename = img_info['id'], img_info['file_name']

            # load image data
            image_path = os.path.join(self.config.eval_img_path, image_filename)
            image = Image.open(image_path).convert('RGB')
            image = T(image)

            if image.shape[1] < 224 or image.shape[2] < 224:
                continue
            else:
                preprocess.append(image_id)

        logger.info('Saved test coco idx file!')

        np.save(self.config.eval_coco_idx_path, np.array(preprocess))

        return preprocess


def get_dataloader(config):
    CocoData = TrainDataLoader(config)
    Dataloader = torch.utils.data.DataLoader(dataset=CocoData,
                                             batch_size=config.batch_size,
                                             shuffle=True,
                                             num_workers=config.num_workers,
                                             drop_last=False)

    return Dataloader


def get_eval_dataloader(config):
    CocoData = EvalDataLoader(config)
    Dataloader = torch.utils.data.DataLoader(dataset=CocoData,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             drop_last=False)

    return Dataloader