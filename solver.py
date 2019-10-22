import torch.utils.data as data

from config import *
from tqdm import tqdm
from dataloader import train_transform, FlatFolderDataset


class Solver(object):
    def __init__(self, args):
        self.args = args
    
    def test_data(self):
        contnet_tf = train_transform()
        style_tf = train_transform()

        # define dataset
        content_dataset = FlatFolderDataset(self.args.content_dir, contnet_tf)
        style_dataset = FlatFolderDataset(self.args.style_dir, style_tf)

        # define iteration
        content_iter = iter(data.DataLoader(content_dataset,
                                            batch_size=self.args.batch_size))
        style_iter = iter(data.DataLoader(style_dataset,
                                        batch_size=self.args.batch_size))
        
        for i in tqdm(range(self.args.max_iter)):
            content_image = next(content_iter)
            style_images = next(style_iter)

            print("content_image: ", content_image.shape)
            print("style_image: ", style_images.shape)