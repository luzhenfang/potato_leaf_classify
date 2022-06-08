import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from classify_label import label_to_index


class LeafDataSet(Dataset):
    def __init__(self, root_path):
        self.images = glob.glob(root_path)
        self.label_to_index = label_to_index
        self.trans = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        image = Image.open(self.images[item])
        image = self.trans(image)
        label = self.images[item].split("\\")[-2]
        return image, self.label_to_index[label]

    def __len__(self):
        return len(self.images)
