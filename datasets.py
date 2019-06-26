from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

class DSpritesDataset(Dataset):
    """D Sprites dataset."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.imgs = np.load(path_to_data)['imgs']
        np.random.shuffle(self.imgs)
        self.imgs = self.imgs[:subsample]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx].astype(np.float32)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0
        
def get_dsprites_dataloader(batch_size=128,
                            path_to_data='data/DSprites/dsprites_ndarray_64x64.npz',
                            subsample=1):
    """DSprites dataloader."""
    dsprites_data = DSpritesDataset(path_to_data,
                                    transform=transforms.ToTensor(), 
                                    subsample=subsample)
    dsprites_loader = DataLoader(dsprites_data, batch_size=batch_size, shuffle=True)
    return dsprites_loader