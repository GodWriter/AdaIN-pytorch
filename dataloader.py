import torch.utils.data as data

from pathlib import Path
from torchvision import transforms
from PIL import Image, ImageFile


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
    
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        
        return img
    
    def __len__(self):
        return len(self.paths)
    
    def name(self):
        return 'FlatFolderDataset'


def load_data(args):
    contnet_tf = train_transform()
    style_tf = train_transform()

    # define dataset
    content_dataset = FlatFolderDataset(args.content_dir, contnet_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

    # define iteration
    content_iter = iter(data.DataLoader(content_dataset,
                                        batch_size=args.batch_size))
    style_iter = iter(data.DataLoader(style_dataset,
                                    batch_size=args.batch_size))
    
    return content_iter, style_iter