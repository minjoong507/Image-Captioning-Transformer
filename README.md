# Image Captioning

## Intro

- This project is an implementation of Image Captiong model based on Transformer Encoder and Deocder.

- Used Pytorch for the code. ResNet101 is used for extracting the features. You can check pre-trained models [here](https://github.com/pytorch/vision/tree/master/torchvision/models).

- Using [COCO dataset](https://cocodataset.org/#home) 2017 Val images [5K/1GB], annotations [241MB].

- Please check the make_vocab.py and data_loader.py. 
  - **Vocab.pickle** is a pickle file which contains all the words in the annotations. 
  - **coco_ids.npy** stores the image ID to be used. Also, you have to set the path or other settings. Execute *prerocess_idx* function.
  - **output_feature.pickle** stores image features.

## Environment

- Python 3.8.5
- Pytorch 1.7.1
- cuda 11.0

## How to use
- For train

```
cd Image-Caption-Bert/vocab
python make_vocab.py

cd feature_extraction
python extraction.py

python train.py

```

- For test




## TODO List
- [ ] Description of the model and other details
- [ ] Code Refactoring
- [ ] Upload requirements.txt
- [ ] Add Inference.py

## License
[MIT License](https://opensource.org/licenses/MIT)

## Reference
[1] [TVCaption](https://github.com/jayleicn/TVCaption)
