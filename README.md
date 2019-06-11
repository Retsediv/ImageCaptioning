# Image Captioning


The goal of the project is to convert a given input image into a natural language description. 

## Content

- [Usage](#usage)
- [Architecture](#architecture)
- [Training](#training)
- [Testing](#testing)
- [Experiments](#experiments)
- [Metrics](#metrics)
- [Results](#results)
- [Dataset](#dataset)


## Usage

#### Clone the repository

#### Install requirements & download dataset and vocabulary

```bash
$ pip install -r requirements.txt
```
Download [Flickr8k](http://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b) and
vocabulary file [here](https://www.dropbox.com/s/26adb7y9m98uisa/vocap.zip?dl=0)

#### Train the model

```bash
$ python train.py    
```

#### Test the model 

```bash
$ python sample.py --image='png/example.png'
```

<br>

## Pretrained model
If you do not want to train the model from scratch, you can use a pretrained model. You can download the pretrained model [here]() 


## Architecture
We use Encoder - Decoder architecture
Encoder -  [resnet-152](https://arxiv.org/abs/1512.03385) pretrained on ImageNet classification dataset
Decoder - LSTM (long short-term memory)

![alt text](https://images.app.goo.gl/q4zBwk9799hKpXaf6)
## Training

## Testing

## Metrics

BLEU

## Results
# Experiments
We tried Inception.v3 and Resnet152 as a decoder for our task and ..
## Dataset
We use [Flickr8k](http://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b)
