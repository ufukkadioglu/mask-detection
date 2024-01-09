Ufuk Kadioglu
91220000413

# Mask Detection

## Running:
Run main.py. It will train and save the model if it is not already trained and will load and use the saved model if it exists. 
Then the model will be evaluated, and some sample images will be shown.

## Datasets:

https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset

## Code:

https://www.kaggle.com/code/nageshsingh/mask-and-social-distancing-detection-using-vgg19

## Haarcascades:

https://www.kaggle.com/datasets/lalitharajesh/haarcascades


## Results:

Training and evaluating takes around 13 minutes.
Loading and evaluating takes around 2 minutes for pretrained models.

Evaluation results:
- ResNet101V2: [0.04492330178618431, 0.9959677457809448]
- DenseNet169: [0.06367169320583344, 0.9949596524238586]
- InceptionV3: [0.044638995081186295, 0.9909273982048035]
- NASNetMobile: [0.041281137615442276, 0.9889112710952759]
- EfficientNetB2: [0.7767146229743958, 0.5131048560142517]
