import torch.utils.data as data
import torch

import net

from config import *
from tqdm import tqdm
from dataloader import load_data


class Solver(object):
    def __init__(self, args):
        self.args = args
    
    def test_data(self):
        content_iter, style_iter = load_data(self.args)

        for i in tqdm(range(self.args.max_iter)):
            content_image = next(content_iter)
            style_images = next(style_iter)

            print("content_image: ", content_image.shape)
            print("style_image: ", style_images.shape)
    
    def train(self):
        # Get device
        device = torch.device('cuda')

        # Get data iteration
        content_iter, style_iter = load_data(self.args)

        # Get model
        vgg = net.vgg
        decoder = net.decoder
        network = net.Net(vgg, decoder)
        network.train()
        network.to(device)
        
        for i in tqdm(range(self.args.max_iter)):
            content_image = next(content_iter).to(device)
            style_images = next(style_iter).to(device)

            # print("style_feats: ", content_image.shape)
            # print("content_feats: ", style_images.shape)

            t = network(content_image, style_images)

            print("t: ", t.shape)