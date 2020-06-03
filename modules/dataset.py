import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ImageNetSubset(Dataset):
    '''Imports subset of the ImageNet dataset from the Kaggle competion'''
    
    def __init__(self, csv_file, root_dir, transform=None):
        '''
        Args:
        csv_file (string)              -- Path to the csv file with metadata like labels and fileId.
        root_dir (string)              -- Directory with all the images.
        transform (callable, optional) -- Optional transform to be applied on a sample.
        '''

        self.images_meta = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
               
    def __len__(self):
        return len(self.images_meta)
    
    def __getitem__(self, idx):     
        image_path = self.root_dir
        image_name = self.images_meta["ImageId"][idx]
        label = self.images_meta["TrueLabel"][idx]
        
        ## Load image
        image = Image.open(image_path + image_name + ".png")
        
        if self.transform is not None:
            image = self.transform(image)
            
        ## Format label. Labels in dataset are 1 indexed but 0 indexed in model. Make all 0 indexed.
        label = torch.tensor(label-1, dtype=torch.long)
        
        ## Move data to cuda if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        image = image.to(device)
        label = label.to(device)
        
        
        return image, label