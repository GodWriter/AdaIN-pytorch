import torch.utils.data as data
import torch.nn as nn
import torch

import net

from config import *
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from dataloader import load_data, test_transform
from tensorboardX import SummaryWriter
from loss import calc_total_loss, adjust_learning_rate
from function import adaptive_instance_normalization as adain

from torchvision.utils import save_image


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

        # Prepare directory
        save_dir = Path(self.args.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        log_dir = Path(self.args.log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)

        # Tensorboard visualization
        writer = SummaryWriter(log_dir=str(log_dir))

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
            loss_c, loss_s, total_loss = calc_total_loss(self.args, t_adain, g_t_feats, style_feats)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # visual loss
            writer.add_scalar('loss_content', loss_c.item(), i + 1)
            writer.add_scalar('loss_style', loss_s.item(), i + 1)

            if (i+1) % self.args.save_model_interval == 0 or (i+1) == self.args.max_iter:
                state_dict = net.decoder.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].to(torch.device('cpu'))
                torch.save(state_dict, save_dir/'decoder_iter_{:d}.pth.tar'.format(i + 1))
        writer.close()
    
    def test(self):
        def style_transfer(vgg, decoder, content, style, alpha=1.0, interpolation_weights=None):
            assert (0.0 <= alpha <= 1.0)
            content_f = vgg(content)
            style_f = vgg(style)

            if interpolation_weights:
                pass
            else:
                feat = adain(content_f, style_f)
            feat = feat * alpha + content_f * (1 - alpha)

            return decoder(feat)

        do_interpolation = False
        
        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        output_dir = Path(self.args.output)
        output_dir.mkdir(exist_ok=True, parents=True)

        # ensure content images
        assert (self.args.content or self.args.content)
        if self.args.content:
            content_paths = [Path(self.args.content)]
        else:
            # if input many images, get path of each image
            content_dir = Path(self.args.content_dir)
            content_paths = [f for f in content_dir.glob('*')]
        
        # ensure style images
        assert (self.args.style or self.args.style_dir)
        if self.args.style:
            style_paths = self.args.style.split(',')
            if len(style_paths) == 1:
                style_paths = [Path(self.args.style)]
            else:
                # do_interpolation = True
                # assert (self.args.style_interpolation_weights != '')
                # weights = [int(i) for i in self.args.style_interpolation_weights.split(',')]
                # interpolation_weights = [w / sum(weights) for w in weights]
                pass
        else:
            style_dir = Path(self.args.style_dir)
            style_paths = [f for f in style_dir.glob('*')]
        
        # define network
        vgg = net.vgg
        decoder = net.decoder
        vgg.eval()
        decoder.eval()

        # restore weights
        vgg.load_state_dict(torch.load(self.args.vgg))
        decoder.load_state_dict(torch.load(self.args.decoder))
        vgg = nn.Sequential(*list(vgg.children())[:31])
        vgg.to(device)
        decoder.to(device)

        # operation of data process
        content_tf = test_transform(self.args.content_size, self.args.crop)
        style_tf = test_transform(self.args.style_size, self.args.crop)

        # begin testing
        for content_path in content_paths:
            if do_interpolation:
                # one content image, N style image
                pass
            else:
                # process one content and one style
                for style_path in style_paths:
                    content = content_tf(Image.open(str(content_path)))
                    style = style_tf(Image.open(str(style_path)))

                    if self.args.preserve_color:
                        pass

                    style = style.to(device).unsqueeze(0)
                    content = content.to(device).unsqueeze(0)

                    with torch.no_grad():
                        output = style_transfer(vgg, decoder, content, style, self.args.alpha)
                    
                    output = output.cpu()
                    output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(content_path.stem, style_path.stem, self.args.save_ext)

                    save_image(output, str(output_name))