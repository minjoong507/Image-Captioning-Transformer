# Image Captioning Transformer

## Intro

- This project is an implementation of Image Captiong model based on Transformer

- Used Pytorch for the code. ResNet152 is used for extracting the features. You can check pre-trained models [here](https://github.com/pytorch/vision/tree/master/torchvision/models).

- Using [COCO dataset](https://cocodataset.org/#home) 2017 Val images [5K/1GB], annotations [241MB].

- Please check `config.py`

## Environment

- Python 3.8.5
- Pytorch 1.7.1
- cuda 11.0

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
data
├── annotations
├── ls
└── val2017
```

2. Install packages:

- Python 3.8.5
- Pytorch 1.7.0+cu110
- nltk
- tqdm

**Training**

3. Build Vocabulary


```
python vocab/make_vocab.py
```

4. Extract Image features


```
python feature_extraction/extraction.py
```

5. Training Model

```
python train.py
```

Training using the above config will stop at epoch 100. I use single RTX 3090 GPU. `result` dir containing the result of code. `2021-*`(=Start time) containing the saved model and train-log.txt.

```
result
└──2021-04-16-12-00
    ├── model.ckpt
    └── train-log.txt
```

## Evaluation
Unconstructed

## TODO List
- [ ] Description of the model and other details
- [ ] Code Refactoring
- [ ] Upload requirements.txt
- [ ] Add Inference.py

## License
[MIT License](https://opensource.org/licenses/MIT)

## Reference
[1] [TVCaption](https://github.com/jayleicn/TVCaption)
