import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from model.layers import BertLayerNorm, PositionEncoding, BertLayer, BertLMPredictionHead


class BertEmbedding(nn.Module):
    def __init__(self, config, vocab_size):
        super(BertEmbedding, self).__init__()
        self.config = config
        self.img_embedding = Image_Encoder(config)
        self.word_embedding = nn.Embedding(vocab_size, config.word_vec_size)
        self.word_fc = nn.Sequential(
            BertLayerNorm(config.word_vec_size, eps=config.layer_norm_eps),
            nn.Dropout(config.dropout),
            nn.Linear(config.word_vec_size, config.hidden_size),
            nn.ReLU(),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.position_embedding = PositionEncoding(config.hidden_size, config.max_position_len)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, image, word):
        image_embedding = self.img_embedding(image)
        words_embedding = self.word_fc(self.word_embedding(word))

        words_embedding[:, 1, :] += image_embedding

        embedding = self.position_embedding(words_embedding)
        embedding = self.LayerNorm(embedding)
        embedding = self.dropout(embedding)

        return embedding


class Image_Encoder(nn.Module):
    def __init__(self, config):
        super(Image_Encoder, self).__init__()
        self.config = config
        self.fc = nn.Linear(config.img_feature_size, config.hidden_size)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = self.fc(img)
        x = self.relu(x)

        return x


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []

        for layer_idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers


class BertCaptioning(nn.Module):
    def __init__(self, config, vocab_size):
        super(BertCaptioning, self).__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.config = config
        self.BertEmbedding = BertEmbedding(self.config, vocab_size)
        self.BertLayer = BertEncoder(self.config)
        self.Classifier = BertLMPredictionHead(self.config)

    def forward(self, inputs):
        embedding = self.BertEmbedding(inputs['img_feature'], inputs['img_sen_tokens'])
        output = self.BertLayer(embedding, inputs['img_sen_mask'])[-1]
        output = self.Classifier(output)

        loss = self.loss(output.view(-1, self.config.vocab_size), inputs['img_sen_label'].view(-1))

        return loss