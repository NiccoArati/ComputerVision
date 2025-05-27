from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import pathlib
import argparse
import re
from PIL import Image
import random
import numpy as np
import torch
from collections import Counter, defaultdict


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, default_parcel, transform=None, file_filter=None):
        self.img_labels = pd.read_excel(annotations_file)
        img_paths = pathlib.Path(img_dir).glob('**/**/*.jpg')
        all_images = sorted([x for x in img_paths])

        if file_filter is not None:
            all_images = [x for x in all_images if x.name in file_filter]

        self.img_sorted = []
        self.labels = []
        self.parcels = []
        self.default_parcel = default_parcel
        self.transform = transform

        for img_path in all_images:
            filename = img_path.name
            label = None
            parcel = None
            #print(img_path)
            match = re.match(r"(\d+)_(\d+)\.jpg", filename)
            if match:
                parcel, pic = match.groups()
                df = self.img_labels[(self.img_labels['parcel'] == int(parcel)) & (self.img_labels['pic'] == int(pic))]
            else:
                match2 = re.match(r"(\d+)\.jpg", filename)
                if match2:
                    parcel = self.default_parcel
                    pic = match2.group(1)
                    df = self.img_labels[(self.img_labels['parcel'] == int(parcel)) & (self.img_labels['pic'] == int(pic))]
                else:
                    continue  # invalid filename, skip

            if not df.empty:
                label = df['class'].values[0] - 1
                self.img_sorted.append(img_path)
                self.labels.append(label)
                self.parcels.append(int(parcel))
            
        print(f"Loaded {len(self.img_sorted)} valid images after filtering")

    def __len__(self):
        return len(self.img_sorted)

    def __getitem__(self, idx):
        img_path = self.img_sorted[idx]
        label = self.labels[idx]
        parcel = self.parcels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label, parcel
    
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


# Returns dictionary with parcel as key and all the file names and labels in that parcel as values
# and another one with parcel as key and its label as value
def group_images_by_parcel(excel_file, default_parcel):
    df = pd.read_excel(excel_file)
    parcels = defaultdict(list)

    for _, row in df.iterrows():
        parcel = row["parcel"]
        pic = row["pic"]
        label = row["class"]
        filename = f"{parcel}_{pic}.jpg"
        if int(parcel) == int(default_parcel):
            filename = f"{pic}.jpg"
        parcels[parcel].append((filename, label))
    
    parcel_class = {}
    for parcel_id, items in parcels.items():
        mean_label = round(np.mean([lbl for _, lbl in items]))
        parcel_class[parcel_id] = mean_label

    return parcels, parcel_class

# Returns parcel ids split into 5 different train and test
def create_splits(parcel_class, n_splits=5):
    class_to_parcels = defaultdict(list)
    for pid, cls in parcel_class.items():
        class_to_parcels[cls].append(pid)
    
    splits = []
    for _ in range(n_splits):
        test_parcels = []
        for cls, parcels in class_to_parcels.items():
            test_parcels += random.sample(parcels, 2)
        train_parcels = [pid for pid in parcel_class if pid not in test_parcels]
        splits.append((train_parcels, test_parcels))
    
    return splits

def print_image_class_distribution(all_parcels, parcels):
    labels = []

    for parcel_id in parcels:
        if parcel_id in all_parcels:
            labels.extend([label for _, label in all_parcels[parcel_id]])
        else:
            print(f"Warning: Parcel {parcel_id} not found in parcels dictionary.")

    class_distribution = Counter(labels)
    print("Image class distribution in parcels:")
    for cls in sorted(class_distribution):
        print(f"Class {cls}: {class_distribution[cls]} images")

def print_parcel_image_distribution(parcels, parcel_class):
    class_presence = defaultdict(list)

    for parcel_id, labels in parcels.items():
        if parcel_id in parcel_class:
            parcel_cls = parcel_class[parcel_id]
            class_presence[parcel_cls].extend([label for _, label in labels])
        else:
            print(f"Warning: Parcel {parcel_id} not in parcel_class mapping.")

    for pclass in sorted(class_presence.keys()):
        label_counts = Counter(class_presence[pclass])
        total = sum(label_counts.values())

        print(f"Parcel Class {pclass} - Image Class Distribution (Total Images: {total}):")
        for cls in sorted(label_counts):
            percent = 100 * label_counts[cls] / total
            print(f"  Class {cls}: {percent:.2f}%")
        print()

def print_test_folds_class_presence(parcels, splits):
    for fold_idx, (train_parcels, test_parcels) in enumerate(splits):
        print(f"\n=== Fold {fold_idx + 1} - Test Set ===")
        for parcel_id in sorted(test_parcels):
            if parcel_id not in parcels:
                print(f"Parcel {parcel_id} not found in dataset.")
                continue

            labels = [label for _, label in parcels[parcel_id]]
            label_counts = Counter(labels)
            total = sum(label_counts.values())

            print(f"Parcel {parcel_id} - Total Images: {total}")
            for cls in sorted(label_counts):
                print(f"  Class {cls}: {label_counts[cls]} images")


def main(labels_file, img_directory, default_parcel, k):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomImageDataset(annotations_file=labels_file, img_dir=img_directory, default_parcel=default_parcel, transform=transform)
    image, _, _ = dataset.__getitem__(400)
    print(f"Images resolution: {image.shape}")

    # Number of images in the dataset
    num_img = dataset.getNumImages()

    # Check instances for each class
    df1 = dataset.getNumClasses()

    # Image distribution in parcels
    random.seed(111)
    np.random.seed(111)
    torch.manual_seed(111)
    torch.cuda.manual_seed_all(111)

    parcels, parcel_class = group_images_by_parcel(labels_file, default_parcel)
    print_parcel_image_distribution(parcels, parcel_class)
    splits = create_splits(parcel_class, n_splits=k)
    print_test_folds_class_presence(parcels, splits)
    for fold, (train_parcels, test_parcels) in enumerate(splits):
        print(f"\n--- Fold {fold + 1} train parcel image distribution ---")
        print_image_class_distribution(parcels, train_parcels)
        print(f"\n--- Fold {fold + 1} test parcel image distribution ---")
        print_image_class_distribution(parcels, test_parcels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Locations')
    parser.add_argument('--labels_file', metavar='path', required=True)
    parser.add_argument('--img_directory', metavar='path', required=True)
    parser.add_argument('--default_parcel', metavar='int', required=True)
    parser.add_argument('--splits', type=int, required=True)
    args = parser.parse_args()
    main(labels_file=args.labels_file, img_directory=args.img_directory, default_parcel=args.default_parcel, k=args.splits)