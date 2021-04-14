import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from model.layers import BertLayerNorm, PositionEncoding, BertLayer, BertLMPredictionHead, BertSelfAttention, BertOutput


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

    def get_caption_word_embedding(self, caption_input_ids):
        """ text_input_ids: (N, Lt) """
        words_embeddings = self.word_fc(self.word_embedding(caption_input_ids))  # (N, Lt, D)
        words_embeddings = self.position_embedding(words_embeddings)
        return words_embeddings

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


class BertDecoderLayer(nn.Module):
    def __init__(self, config):
        super(BertDecoderLayer, self).__init__()
        self.config = config
        self.self_attention = BertSelfAttention(config)
        self.norm1 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dec_enc_attention = BertSelfAttention(config)
        self.norm2 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output = BertOutput(config)  # linear + residual + layernorm

    def forward(self, dec_hidden_states, dec_mask, enc_outputs, enc_mask, diagonal_mask=True):
        """
        Args:
            dec_hidden_states: (N, Lt, D)
            dec_mask: (N, Lt)
            enc_outputs: (N, Lv, D)
            enc_mask: (N, Lv)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property.
        Returns:

        """
        self_attention_mask = dec_mask.unsqueeze(1)
        if diagonal_mask:  # mask subsequent words
            max_len = dec_mask.size(1)  # Lt
            self_attention_mask = self_attention_mask * \
                torch.tril(self_attention_mask.new_ones(max_len, max_len), diagonal=0)

        # 1, dec self attn + add_norm
        attention_output = self.self_attention(
            dec_hidden_states, dec_hidden_states, dec_hidden_states, self_attention_mask)  # (N, Lt, D)
        attention_output = self.norm1(attention_output + dec_hidden_states)  # (N, Lt, D)

        # 2, dec enc attn + add_norm
        # Is the attention mask correct?
        # Yes! Use the mask associated with key/value, not query. (query, key, value)
        # Additionally, there is no need to do subsequent masking, since each word has the right to see
        # all the video info.
        dec_enc_attention_output = self.dec_enc_attention(
            attention_output, enc_outputs, enc_outputs, enc_mask.unsqueeze(1))  # (N, Lt, D)
        dec_enc_attention_output = self.norm2(attention_output + dec_enc_attention_output)  # (N, Lt, D)

        # 3, linear + add_norm
        dec_enc_attention_output = self.output(dec_enc_attention_output, dec_enc_attention_output)  # (N, Lt, D)
        return dec_enc_attention_output  # (N, Lt, D)


class BertDecoder(nn.Module):
    def __init__(self, config):
        super(BertDecoder, self).__init__()
        self.layer = nn.ModuleList([BertDecoderLayer(config)
                                    for _ in range(config.num_layers)])

    def forward(self, dec_hidden_states, dec_mask, enc_outputs, enc_mask,
                diagonal_mask=True, output_all_encoded_layers=False):
        """
        Args:
            dec_hidden_states: (N, Lt, D)
            dec_mask: (N, Lt)
            enc_outputs: (N, Lv, D)
            enc_mask: (N, Lv)
            diagonal_mask: bool, if True mask subsequent words to preserve auto-regressive property
            output_all_encoded_layers:

        Returns:

        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            dec_hidden_states = layer_module(
                dec_hidden_states, dec_mask, enc_outputs, enc_mask, diagonal_mask=diagonal_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(dec_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(dec_hidden_states)
        return all_encoder_layers


class BertCaptioning(nn.Module):
    def __init__(self, config, vocab_size):
        super(BertCaptioning, self).__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.config = config
        self.BertEmbedding = BertEmbedding(self.config, vocab_size)
        self.BertEncoder = BertEncoder(self.config)
        self.BertDeocder = BertDecoder(self.config)
        self.Classifier = BertLMPredictionHead(self.config)

    def forward(self, inputs):
        embedding = self.BertEmbedding(inputs['img_feature'], inputs['img_sen_input_ids'])
        enc_output = self.BertEncoder(embedding, inputs['img_sen_mask'])[-1]

        caption_embedding = self.BertEmbedding.get_caption_word_embedding(inputs['captions_input_ids'])
        dec_output = self.BertDeocder(caption_embedding, inputs['captions_mask'], enc_output, inputs['img_sen_mask'])[-1]

        output = self.Classifier(dec_output)

        loss = self.loss(output.view(-1, self.config.vocab_size), inputs['captions_label'].view(-1))

        return output, loss