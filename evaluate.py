import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence

from data.data import get_loader
from build_vocab import Vocabulary
from model.model import EncoderCNN, DecoderRNN
from nltk.translate.bleu_score import corpus_bleu
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, args.test_img,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Evaluate the model
    total_step = len(data_loader)
    for i, (images, captions, lengths) in enumerate(data_loader):
        # Set mini-batch dataset
        images = images.to(device)

        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # Generate an caption from the image
        feature = encoder(images)

        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids.cpu().numpy() 

        targets_captions = []
        captions = captions.cpu().numpy()
        for samp in captions:
            caption = []

            for word_id in samp:
                word = vocab.idx2word[word_id]
                caption.append(word)
                if word == '<end>':
                    break
            targets_captions.append(caption)

        # Convert word_ids to words
        predicted_captions = []
        for samp in sampled_ids:
            caption = []

            for word_id in samp:
                word = vocab.idx2word[word_id]
                caption.append(word)
                if word == '<end>':
                    break
            predicted_captions.append(caption)


        print("targets_captions: ", targets_captions[:20])
        print("predicted_captions: ", predicted_captions[:20])

        references = [[targets_captions[0]]]
        candidates = [predicted_captions[0]]
        score = corpus_bleu(references, candidates)

        print("references: ", references)
        print("candidates: ", candidates)
        print(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='Flickr8k/dataset/', help='directory for resized images')
    parser.add_argument('--encoder_path', type=str, default='./model/third_try/encoder-epoch-200-loss-0.0010738003766164184.ckpt',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./model/third_try/decoder-epoch-200-loss-0.0010738003766164184.ckpt',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--test_img', type=str, default='./Flickr8k/text/Flickr_8k.testImages.txt',
                        help='path to test(or dev) images listing')

    parser.add_argument('--crop_size', type=int, default=298, help='resizing size')
    parser.add_argument('--caption_path', type=str, default='./Flickr8k/text/Flickr8k.token.txt',
                        help='path for annotation file')

    parser.add_argument('--batch_size', type=int, default=128)

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()
    main(args)
