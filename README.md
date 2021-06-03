# Image Captioning Transformer

## Intro

- This project is an implementation of Image Captiong model based on Transformer

- Used Pytorch for the code. ResNet152 is used for extracting the features. You can check pre-trained models [here](https://github.com/pytorch/vision/tree/master/torchvision/models).

- Using [COCO dataset](https://cocodataset.org/#home) 2017 Val images [5K/1GB], annotations [241MB] for train.

- Please check `config.py`. Also, you can train on multi GPUs.

## Getting Started
**Prerequisites**

0. Clone this repo
```
git clond https://github.com/minjoong507/Image-Captioning-Transformer.git
cd Image-Captioning-Transformer
```

1. Download COCO dataset
```
mkdir data

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip

```

```
data
├── annotations
├── ls
├── train2017
├── val2017
└── test2017
```

2. Install packages:

- Python 3.8.5
- Pytorch 1.7.0+cu110
- nltk
- tqdm

3. Add project root to PYTHONPATH

```
source setup.sh
```

**Training**

4. Build Vocabulary


```
python vocab/make_vocab.py
```

5. Extract Image features


```
python feature_extraction/extraction.py
```

6. Training Model

```
python train.py
```

Training using the above config will stop at epoch 100. I use single or six RTX 3090 GPU. `result` dir containing the result of code. `2021-*`(=Start time) containing the saved model and train-log.txt.

Example
```
result
└──2021-04-16-12-00
    ├── model.ckpt
    └── train-log.txt
```

**Testing**

```
python Inference --test_path MODEL_DIR_NAME
```
`MODEL_DIR_NAME` is the name of the dir containing the saved model, e.g., `2021-*.`

## Evaluation

- Train loss & acc (100 epoch)
    - Single GPU : 
        - Accuracy : 98.6766 %
        - Result
        ```
        predict : a [UNK] with people is near a pier on clear water . [EOS]
        target : a [UNK] with people is near a pier on clear water . [EOS]
        ```
         <img src = "https://github.com/minjoong507/Image-Captioning-Transformer/blob/master/image/000000239274.jpg" width="300px;" align="center">    
         
    - Six GPUs : 
        - Accuracy : 99.4794 %
        - Result
        ```
        predict : a picture of a giraffe standing in a zoo exhibit . [EOS]
        target : a picture of a giraffe standing in a zoo exhibit . [EOS]

        predict : people and buses on a city street under cloudy skies . [EOS]
        target : people and buses on a city street under cloudy skies . [EOS]

        predict : a man at an office desk drinking a glass of wine . [EOS]
        target : a man at an office desk drinking a glass of wine . [EOS]

        predict : two zebras are standing next to a log . [EOS]
        target : two zebras are standing next to a log . [EOS]
        ```
- Inference
    - unconstructed

## TODO List
- [ ] Description of the model and other details
- [ ] Code Refactoring
- [ ] Add Inference.py

## File
```
Image-Captioning-Transformer
├── model
│   ├── data_loader.py
│   ├── layers.py
│   ├── model.py
│   └── optimization.py
├── data
│   ├── output_feature.pickle # after python extraction.py
│   ├── annotations
│   ├── ls
│   └── val2017
├── feature_extraction
│   ├── data_loader.py
│   ├── extraction.py
│   └── resnet.py
├── vocab
│   ├── vocab.pickle # after python make_vocab.py
│   ├── coco_idx.npy # after python extraction.py
│   └── make_vocab.py
├── LICENSE
├── .gitignore
└── README.md
```

## License
- [MIT License](https://opensource.org/licenses/MIT)

## Reference
- [1] [TVCaption](https://github.com/jayleicn/TVCaption)
- [2] [huggingface/transformer](https://github.com/huggingface/transformers)
