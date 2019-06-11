import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data.data import get_loader
from build_vocab import Vocabulary
from model.model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

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
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, args.train_img,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):

            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

        # Save the model checkpoints
        #if  (epoch + 1) % args.save_step == 0:
        print('saving weights')
        torch.save(decoder.state_dict(), os.path.join(
            args.model_path, 'decoder-epoch-{}-loss-{}.ckpt'.format(epoch + 1, loss)))
        torch.save(encoder.state_dict(), os.path.join(
            args.model_path, 'encoder-epoch-{}-loss-{}.ckpt'.format(epoch + 1, loss)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=298, help='resizing size')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='Flickr8k/dataset/', help='directory for resized images')
    parser.add_argument('--train_img', type=str, default='./Flickr8k/text/Flickr_8k.trainImages.txt',
                        help='path to train images listing')
    parser.add_argument('--caption_path', type=str, default='./Flickr8k/text/Flickr8k.token.txt',
                        help='path for annotation file')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=400, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
