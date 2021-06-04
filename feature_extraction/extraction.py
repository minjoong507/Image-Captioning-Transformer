from feature_extraction.resnet import ResNet
from feature_extraction.data_loader import get_dataloader, get_eval_dataloader
from utils import save_pickle, get_logger
import torch
from tqdm import tqdm
import argparse
from vocab.make_vocab import Make_vocab, Vocab

logger = get_logger()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = torch.device(device)


class Feature_extraction:
    def __init__(self, config):
        self.config = config
        self.TrainDataloader = get_dataloader(self.config)
        self.EvalDataloader = get_eval_dataloader(self.config)
        self.visual_feature = {}
        self.Resnet = self.get_model()


    def get_model(self):
        model = ResNet()
        if torch.cuda.is_available():
            model = model.cuda()

        logger.info('Use {}'.format(cuda))

        return model

    def run_extract(self, is_train=True):
        if is_train:
            self.extract(self.TrainDataloader)
        else:
            self.extract(self.EvalDataloader)

    def extract(self, dataloader):
        for batch_idx, (batch) in tqdm(enumerate(dataloader), desc=' Visual feature extraction', total=len(dataloader)):
            image_id, image = list(batch[0]), batch[1].cuda()
            with torch.no_grad():
                output = self.Resnet(image)

            for i in range(len(image_id)):
                self.visual_feature[image_id[i]] = output[i].cpu()

        logger.info("Saving the output features at {}".format(self.config.visual_feature_path))
        save_pickle(self.visual_feature, self.config.visual_feature_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visual feature extraction")
    parser.add_argument('--vocab_path', default='data/vocab.pickle', type=str)
    parser.add_argument('--annotations_path', default='data/annotations/captions_train2017.json', type=str)
    parser.add_argument('--img_path', default='data/train2017/', type=str, help='image path')
    parser.add_argument('--eval_img_path', default='data/test2017/', type=str, help='image path')
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--visual_feature_path', default='data/visual_feature.pickle', type=str)
    parser.add_argument('--test_visual_feature_path', default='data/visual_feature_test.pickle', type=str)
    parser.add_argument('--coco_idx_path', default='data/coco_idx.npy', type=str)
    parser.add_argument('--eval_coco_idx_path', default='data/eval_coco_idx.npy', type=str)
    parser.add_argument('--eval_annotations_path', default='data/annotations/image_info_test2017.json', type=str)

    args = parser.parse_args()

    feature = Feature_extraction(args)
    # For train dataset image features
    feature.run_extract()

    # For eval dataset test features
    feature.run_extract(is_train=False)