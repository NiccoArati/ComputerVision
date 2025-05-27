from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import numpy as np
from datasetStudy import *
from torch.utils.data import DataLoader, Subset
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model

def extract_features(model, dataloader, device='cuda'):
    model.eval()
    features = []
    labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images, output_hidden_states=True)

            # Take the last hidden state of the CLS token
            last_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)
            cls_embeddings = last_hidden_state[:, 0, :]     # (batch_size, hidden_dim)

            features.append(cls_embeddings.cpu())
            labels_list.extend(labels.cpu().numpy())

    features = torch.cat(features, dim=0).numpy()
    return features, np.array(labels_list)

def run_tsne(features, labels, save_path=None):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    for class_idx in np.unique(labels):
        idxs = labels == class_idx
        plt.scatter(tsne_result[idxs, 0], tsne_result[idxs, 1], label=f'Class {class_idx}', alpha=0.7)

    plt.legend()
    plt.title("t-SNE of ViT Features")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"t-SNE plot saved to {save_path}")
    plt.show()

def main(labels_file, img_directory, default_parcel, model_path, save_path, lora):
    train_transforms = transforms.Compose([  
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ViT input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    torch.manual_seed(111)
    np.random.seed(111)

    dataset = CustomImageDataset(annotations_file=labels_file, img_dir=img_directory, default_parcel=default_parcel, transform=train_transforms)
    # Split into train and test with balanced test set
    indices = np.arange(len(dataset))
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(set(labels.tolist()))
    test_samples_per_class = int(len(dataset) * 0.2 / 3)
    test_indices = []
    train_indices = indices.tolist()

    for class_label in range(num_classes):
        class_indices = np.where(labels == class_label)[0]
        selected_test_indices = np.random.choice(class_indices, test_samples_per_class, replace=False)
        
        test_indices.extend(selected_test_indices)
        train_indices = [idx for idx in train_indices if idx not in selected_test_indices]  # Remove from train
    
    train_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, test_indices)
    test_set.dataset.transform = test_transforms

    trainloader = DataLoader(train_set, batch_size=32, shuffle=True)
    testloader = DataLoader(test_set, batch_size=32, shuffle=False)

    model_name = "google/vit-base-patch16-224-in21k"
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label={str(i): c for i, c in enumerate(set(labels.tolist()))},
        label2id={c: str(i) for i, c in enumerate(set(labels.tolist()))},
    )
    if lora:
        target_modules=["vit.encoder.layer.0.attention.attention.query", "vit.encoder.layer.0.attention.attention.value",
                        "vit.encoder.layer.1.attention.attention.query", "vit.encoder.layer.1.attention.attention.value",
                        "vit.encoder.layer.2.attention.attention.query", "vit.encoder.layer.2.attention.attention.value",
                        "vit.encoder.layer.3.attention.attention.query", "vit.encoder.layer.3.attention.attention.value",
                        "vit.encoder.layer.4.attention.attention.query", "vit.encoder.layer.4.attention.attention.value", 
                        "vit.encoder.layer.5.attention.attention.query", "vit.encoder.layer.5.attention.attention.value",
                        "vit.encoder.layer.6.attention.attention.query", "vit.encoder.layer.6.attention.attention.value",
                        "vit.encoder.layer.7.attention.attention.query", "vit.encoder.layer.7.attention.attention.value",
                        "vit.encoder.layer.8.attention.attention.query", "vit.encoder.layer.8.attention.attention.value",
                        "vit.encoder.layer.9.attention.attention.query", "vit.encoder.layer.9.attention.attention.value",
                        "vit.encoder.layer.10.attention.attention.query", "vit.encoder.layer.10.attention.attention.value",
                        "vit.encoder.layer.11.attention.attention.query", "vit.encoder.layer.11.attention.attention.value",]
        lora_config = LoraConfig(r=8, lora_alpha=32, target_modules = target_modules, lora_dropout=0.05, bias="none")
        model = get_peft_model(model, lora_config)
    
    model.load_state_dict(torch.load(model_path))
    model.to('cuda')
    features, labels = extract_features(model, testloader, device='cuda')
    run_tsne(features, labels, save_path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Locations')
    parser.add_argument('--labels_file', metavar='path', required=True)
    parser.add_argument('--img_directory', metavar='path', required=True)
    parser.add_argument('--default_parcel', metavar='int', required=True)
    parser.add_argument('--model_path', metavar='path', required=True)
    parser.add_argument('--save_path', metavar='path', required=True)
    parser.add_argument("--loRA", action="store_true", help="performs LoRA")
    parser.add_argument("--noLoRA", action="store_false", dest="loRA", help="does not perform LoRA")
    args = parser.parse_args()
    main(labels_file=args.labels_file, img_directory=args.img_directory, default_parcel=args.default_parcel,
         model_path=args.model_path, save_path=args.save_path, lora=args.loRA)