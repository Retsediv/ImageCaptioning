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
$ python test.py --image='{name}.{ext}'
```

<br>

## Pretrained model
If you do not want to train the model from scratch, you can use a pretrained model. You can download the pretrained model [here]() 


## Architecture
We use Encoder - Decoder architecture
![alt text](https://raw.githubusercontent.com/yunjey/pytorch-tutorial/master/tutorials/03-advanced/image_captioning/png/model.png)

Image Encoder -  [resnet-152](https://arxiv.org/abs/1512.03385) We load in a pretrained on ImageNet classification dataset model and remove the final layer
Text Decoder - LSTM (long short-term memory)

```python
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):

        super(EncoderCNN, self).__init__()
        model = models.resnet152(pretrained=True)
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)
        self.linear = nn.Linear(model.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):

        with torch.no_grad():
            features = self.model(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):

        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):

        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

```
## Training

## Testing

## Metrics

BLEU

## Results

| ``` <start> two small girls laugh as a wave hits them at the beach . <end>``` | ```<start> the tan and white dog is running in the field . <end>``` | ```<start> two woman stand together and one holds a purple cloth . <end>``` | ```<start> two women play <unk> while a third man in the background . <end>``` |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                   ![wave](images/wave.jpg)                   |                   ![a dog](images/dog.jpg)                   |                  ![girls](images/girls.jpg)                  |                   ![guys](images/hip.jpg)                    |


​    


## Dataset
We use [Flickr8k](http://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b), which contains 8000 images with 5 description each 

<br>
# Thank u for your attention!
![Thank u](https://media.giphy.com/media/jNdw5Qmy5MOpq/giphy.gif)
