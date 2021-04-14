from feature_extraction.resnet import Model
from feature_extraction.data_loader import get_dataloader
from config import BasicOption
from utils import save_pickle
import torch
from tqdm import tqdm
from vocab.make_vocab import Make_vocab, Vocab
import logging
logger = logging.getLogger(__name__)


device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
cuda = torch.device(device)
print('Use {}'.format(cuda))


class Feature_extraction:
    def __init__(self):
        self.config = BasicOption().parse()
        self.DataLoader = get_dataloader(self.config)
        self.Resnet = Model(self.config).cuda()
        self.output_feature = {}

    def extract(self):
        logger.info("Now extracting image features")
        for batch_idx, (batch) in tqdm(enumerate(self.DataLoader)):
            image_id, image = list(batch[0]), batch[1].cuda()
            with torch.no_grad():
                output = self.Resnet(image)

            for i in range(len(image_id)):
                self.output_feature[image_id[i]] = output[i].cpu()

        logger.info("Saving the output features")
        save_pickle(self.output_feature, '../' + self.config.output_feature_path)


if __name__ == '__main__':
    feature = Feature_extraction()
    feature.extract()