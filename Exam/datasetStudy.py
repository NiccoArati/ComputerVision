from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import pathlib
import argparse
import re
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, default_parcel, train=True, transform=None):
        self.img_labels = pd.read_excel(annotations_file)
        #self.img_labels.to_csv("classi.csv", index=False)
        img_paths = pathlib.Path(img_dir).glob('**/**/*.jpg')
        self.img_sorted = sorted([x for x in img_paths])
        self.default_parcel = default_parcel
        self.train = train
        self.transform = transform

        self.class_transforms = {
            0: transforms.Compose([  
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]),
            1: transforms.Compose([  # Class 1 Augmentations
                transforms.Resize((224, 224)),  # Resize to ViT input size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]),
            2: transforms.Compose([  # Class 2 Augmentations
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        }

    def __len__(self):
        return len(self.img_sorted)

    def __getitem__(self, idx):
        img_path = self.img_sorted[idx]
        image = Image.open(img_path).convert("RGB")
        filename = pathlib.Path(img_path).name

        match = re.match(r"(\d+)_(\d+)\.jpg", filename)
        if match:
            parcel, pic = match.groups()
            filtered_df = self.img_labels[(self.img_labels['parcel'] == int(parcel)) & (self.img_labels['pic'] == int(pic))]
            if not filtered_df.empty:
                label = filtered_df['class'].values[0] - 1 # 0, 1, 2 instead of 1, 2, 3
            else:
                print("Not found")
                label = None
        else:
            match2 = re.match(r"(\d+)\.jpg", filename) # P39
            if match2:
                parcel = self.default_parcel
                pic = match2.group(1)
                filtered_df = self.img_labels[(self.img_labels['parcel'] == int(parcel)) & (self.img_labels['pic'] == int(pic))]
                if not filtered_df.empty:
                    label = filtered_df['class'].values[0] - 1 # 0, 1, 2 instead of 1, 2, 3
                else:
                    print("Not found")
                    label = None
            else:
                print("No match")
                label = None
        
        if self.train:
            transform = self.class_transforms.get(label)
            image = transform(image)
        elif self.transform:
            image = self.transform(image)
        
        return image, label
    
    def getNumImages(self):
        num_img = len(self.img_labels.index)
        num_img2 = len(self.img_sorted)
        print(f"Number of images in the dataset: {num_img}")
        print(f"Number of images in the dataset: {num_img2}")
        return num_img2
    
    def getNumClasses(self):
        df_class = self.img_labels.pivot_table(index = ['class'], aggfunc='size')
        print("Get count of instances for each class:\n", df_class)
        return df_class


def main(labels_file, img_directory, default_parcel):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomImageDataset(annotations_file=labels_file, img_dir=img_directory, default_parcel=default_parcel, transform=transform)
    image, _ = dataset.__getitem__(400)
    print(f"Images resolution: {image.shape}")

    # Number of images in the dataset
    num_img = dataset.getNumImages()

    # Check instances for each class
    df1 = dataset.getNumClasses()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Locations')
    parser.add_argument('--labels_file', metavar='path', required=True)
    parser.add_argument('--img_directory', metavar='path', required=True)
    parser.add_argument('--default_parcel', metavar='int', required=True)
    args = parser.parse_args()
    main(labels_file=args.labels_file, img_directory=args.img_directory, default_parcel=args.default_parcel)