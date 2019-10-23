import torch.utils.data as data
import torch.nn as nn
import torch

import net

from config import *
from tqdm import tqdm
from dataloader import load_data
from loss import calc_total_loss, adjust_learning_rate


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

        vgg.load_state_dict(torch.load(self.args.vgg))
        vgg = nn.Sequential(*list(vgg.children())[:31])

        network = net.Net(vgg, decoder)
        network.train()
        network.to(device)

        # define optimizer
        optimizer = torch.optim.Adam(network.decoder.parameters(), lr=self.args.lr)

        for i in tqdm(range(self.args.max_iter)):
            adjust_learning_rate(self.args, optimizer, iteration_count=i)

            content_image = next(content_iter).to(device)
            style_images = next(style_iter).to(device)
            
            # Run the network and compute the loss
            t_adain, g_t_feats, style_feats = network(content_image, style_images)
            total_loss = calc_total_loss(self.args, t_adain, g_t_feats, style_feats)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print("total_loss: ", total_loss)