import argparse


class BasicOption:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Image Captioning')
        self.is_initialized = False

    def initialize(self):
        self.is_initialized = True

        self.parser.add_argument('--annotations_dir', default='data/annotations/captions_val2017.json', type=str)
        self.parser.add_argument('--img_dir', default='data/val2017/', type=str)
        self.parser.add_argument('--crop_size', default=224, type=int)
        self.parser.add_argument('--epochs', default=100, type=int)
        self.parser.add_argument('--lr', default=1e-4, type=float)
        self.parser.add_argument('--batch_size', default=4, help='')
        self.parser.add_argument('--num_workers', default=4, type=int)
        self.parser.add_argument('--img_feature_size', default=2048, type=int)
        self.parser.add_argument('--hidden_size', default=768, type=int)
        self.parser.add_argument('--word_vec_size', default=256, type=int)
        self.parser.add_argument('--num_layers', default=2, type=int)
        self.parser.add_argument("-intermediate_size", type=int, default=768)
        self.parser.add_argument("--num_attention_heads", type=int, default=12)
        self.parser.add_argument('--max_position_len', default=500, type=int)
        self.parser.add_argument('--model_path', default='model/', type=str)
        self.parser.add_argument('--vocab_path', default='vocab/vocab.pickle', type=str)
        self.parser.add_argument('--vocab_size', default=None, type=int)
        self.parser.add_argument('--output_feature_path', default='data/output_feature.pickle', type=str)
        self.parser.add_argument('--save_step', default=1000, type=int)
        self.parser.add_argument('--eval_batch_size', type=int, default=16)
        self.parser.add_argument('--shuffle', type=str, default="True")
        self.parser.add_argument('--dropout', type=float, default=0.2)
        self.parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.1)
        self.parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
        self.parser.add_argument("--lr_warmup_proportion", type=float, default=0.1)
        self.parser.add_argument('--max_sub_len', type=int, default=30)
        self.parser.add_argument('--min_fq', type=int, default=5)
        self.parser.add_argument('--device', type=int, default=4)

    def parse(self):
        if not self.is_initialized:
            self.initialize()

        opt = self.parser.parse_args()

        return opt