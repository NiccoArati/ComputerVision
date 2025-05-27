import torch
import wandb
import argparse
from datasetStudy import *
from torch.utils.data import DataLoader
from collections import Counter
from transformers import ViTModel
from torchvision import transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
import numpy as np


def main(labels_file, img_directory, default_parcel):
    wandb.login()
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ViT input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    torch.manual_seed(111)
    np.random.seed(111)

    dataset = CustomImageDataset(annotations_file=labels_file, img_dir=img_directory, default_parcel=default_parcel, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(set(labels.tolist()))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
    model.eval()

    run = wandb.init(
    # Set the project where this run will be logged
    project="Computer Vision ViT hf KNN",
    # Pass a run name
    name=f"ViT_b_16 zero-shot knn-classifierNC ImageNet1k",
    # Track hyperparameters and run metadata
    config={
    "normalize": True,
    "metric": 'cosine',
    "architecture": "ViT_b_16",
    "dataset": "cnr"})

    features = []
    labels = []

    with torch.no_grad():
        for data in dataloader:
            images, labels_dl = data
            images = images.to(device)
            output = model(images) # Extract 768-dim feature vector from ViT-B/16
            cls_embedding = output.last_hidden_state[:, 0, :] 
            features.append(cls_embedding.cpu().numpy())
            labels.extend(labels_dl.numpy())
    
    features = np.vstack(features)  # Stack all feature vectors
    if run.config["normalize"]:
        features = normalize(features, axis=1)
    labels = np.array(labels)

    class_counts = Counter(labels)
    min_class_count = min(class_counts.values())  # Smallest class size

    # Create balanced test set
    test_indices = []
    train_indices = []

    # Randomly sample min_class_count instances per class for test set
    for cls in class_counts.keys():
        cls_indices = np.where(labels == cls)[0]  # Get indices of this class
        np.random.shuffle(cls_indices)  # Shuffle indices
        test_indices.extend(cls_indices[:min_class_count])  # Take equal samples
        train_indices.extend(cls_indices[min_class_count:])  # Rest for training

    # Convert to numpy arrays
    test_indices = np.array(test_indices)
    train_indices = np.array(train_indices)

    # Split into training and balanced test set
    X_train, X_test = features[train_indices], features[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]

    k_values = list(range(1, 71, 2))
    for k in k_values:
        # Train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k, metric=run.config["metric"])
        knn.fit(X_train, y_train)

        # Evaluate model
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        test_metrics = {"Test Accuracy": accuracy}

        unique_classes = np.unique(y_test)
        class_accuracies = {}

        for cls in unique_classes:
            cls_indices = np.where(y_test == cls)[0]
            cls_accuracy = accuracy_score(y_test[cls_indices], y_pred[cls_indices])
            class_accuracies[cls] = cls_accuracy
            print(f"Class {cls} Accuracy: {cls_accuracy * 100:.2f}%")

        wandb.log({**test_metrics, **{f"Class {cls} Accuracy": acc for cls, acc in class_accuracies.items()}})
        print(f"KNN Accuracy with k={k}: {accuracy * 100:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Locations')
    parser.add_argument('--labels_file', metavar='path', required=True)
    parser.add_argument('--img_directory', metavar='path', required=True)
    parser.add_argument('--default_parcel', metavar='int', required=True)
    args = parser.parse_args()
    main(labels_file=args.labels_file, img_directory=args.img_directory, default_parcel=args.default_parcel)