from datasetStudy import *
from torch.utils.data import DataLoader, Subset
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
from transformers import ViTForImageClassification
import wandb
import cv2
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pytorch_grad_cam import GradCAM, EigenGradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from peft import LoraConfig, get_peft_model

#TODO matrice di confusione DONE
#TODO MLP invece di linear layer con classification head DONE
#TODO tsne DONE
#TODO provare con modelli più grossi (L14, H14)
#TODO prova focal loss, paper e concatenare patch a input

# To train a MLP on top of classification head
class CustomMLPHead(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, activation=torch.nn.ReLU):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(torch.nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(torch.nn.Linear(hidden_dim, num_classes))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

def train(model, optimizer, trainloader, criterion, epoch, mixUp, alpha=0.0, device='cuda'):
    model.train()
    train_losses = []
    mixup = None
    all_labels = []
    if mixUp:
        mixup = v2.CutMix(alpha=alpha, num_classes=3)
    for data in trainloader:
        images, labels, _ = data
        if mixup is not None:
            images, labels = mixup(images, labels)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output.logits, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        print(f"train loss: {loss.item()}")
        all_labels.extend(labels.tolist())

    # At end of epoch: check class distribution
    class_counts = Counter(all_labels)
    print(f"Epoch {epoch} - Images per class seen:")
    for cls in sorted(class_counts):
        print(f"  Class {cls}: {class_counts[cls]} images")
        
    print(f"Epoch {epoch} mean loss: {np.mean(train_losses)}")
    return np.mean(train_losses)

def plot_confusion_matrix(cm, class_names=[0, 1, 2], save_path=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.close()

def test(model, testloader, criterion, device='cuda'):
    model.eval()
    test_loss = 0
    correct = 0
    class_correct = Counter()
    class_total = Counter()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data in testloader:
            images, labels, _ = data
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            logits = output.logits
            test_loss += criterion(logits, labels).item()
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()

            for label, pred in zip(labels, predictions):
                class_total[label.item()] += 1
                if label.item() == pred.item():
                    class_correct[label.item()] += 1
                
                all_predictions.append(pred.item())
                all_labels.append(label.item())


    test_acc = correct / len(testloader.dataset)
    class_accuracy = {cls: (class_correct[cls] / class_total[cls]) if class_total[cls] > 0 else 0.0 for cls in class_total}
    mean_test_loss = test_loss / len(testloader.dataset)
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2])

    print(f"Test accuracy: {test_acc:.3f}")
    for cls, acc in class_accuracy.items():
        print(f"Class {cls} Accuracy: {acc:.2f}")
    plot_confusion_matrix(cm, class_names=["Class 0", "Class 1", "Class 2"], save_path="/home/narati/cnr/confusionM/cm_model_all.jpg")

    return mean_test_loss, test_acc, class_accuracy


def chFineTuning(model, trainloader, testloader, mixUp, lora=False, device='cuda'):
    run = wandb.init(
    # Set the project where this run will be logged
    project="Computer Vision ViT hf Final Tests CM",
    # Pass a run name
    name=f"ViT_b_16 finetuning LoRA 0.05 all (DELETE)",
    # Track hyperparameters and run metadata
    config={
    "learning_rate": 1e-2,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "milestones": [35, 60, 85],
    "gamma": 0.1,
    "alphaMixUp": 0.0,
    "label_smoothing": 0.0,
    "architecture": "ViT_b_16",
    "dataset": "cnr",
    "epochs": 100})

    model.to(device)
    if not lora:
        # Freeze all the weights of the network, except for classifier head ones
        for param in model.parameters():
            param.requires_grad = False
        model.classifier.weight.requires_grad = True
        model.classifier.bias.requires_grad = True
    optimizer = torch.optim.SGD(model.parameters(), lr=run.config["learning_rate"], momentum=run.config["momentum"], weight_decay=run.config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=run.config["milestones"], gamma=run.config["gamma"])
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=run.config["label_smoothing"])

    train_losses = []
    for epoch in range(1, run.config["epochs"] + 1):
        print(f"Training, epoch {epoch}")
        train_loss = train(model, optimizer, trainloader, criterion, epoch, mixUp, device)
        train_losses.append(train_loss)
        metrics = {"Train Loss": train_loss}
        print(f"Epoch {epoch} mean Loss: {train_loss}")
        scheduler.step()

        val_loss, val_acc, class_acc = test(model, testloader, criterion, device=device)
        print(val_acc)
        val_metrics = {"Val_accuracy": val_acc}
        wandb.log({**metrics, **val_metrics})
    
    #torch.save(model.state_dict(), "model.pth")
    test_loss, test_acc, class_acc = test(model, testloader, criterion, device=device)
    test_metrics = {"Test Accuracy": test_acc}
    wandb.log({**test_metrics}, **{f"Class {cls} Accuracy": acc for cls, acc in class_acc.items()})
    wandb.finish()
    return test_loss, test_acc

def layerFineTuning(model, trainloader, testloader, mixUp, toFinetune=1, device='cuda'):
    run = wandb.init(
    # Set the project where this run will be logged
    project="Computer Vision ViT hf Final Tests CM",
    # Pass a run name
    name=f"Vit_b_16 fineTuning 4-2 100 MixUp",
    # Track hyperparameters and run metadata
    config={
    "learning_rate": 5e-3,
    "momentum": 0.9,
    "learning_rate2": 1e-3,
    "weight_decay": 1e-4,
    "milestones": [35, 60, 85],
    "gamma": 0.1,
    "alpha_mixUp": 0.0,
    "label_smoothing": 0.0,
    "architecture": "ViT_b_16",
    "dataset": "cnr",
    "epochs": 100})
    model.to(device)
    # frozen all the weights of the network, except for fc ones and last layer
    for param in model.parameters():
        param.requires_grad = False
    model.classifier.weight.requires_grad = True
    model.classifier.bias.requires_grad = True
    for param in model.vit.encoder.layer[-toFinetune:].parameters():
        param.requires_grad = True
    ch_params = {'params': model.classifier.parameters(), 'lr': run.config["learning_rate"]}
    layer_params = {'params': model.vit.encoder.layer[-toFinetune:].parameters(), 'lr': run.config["learning_rate2"]}
    optimizer = torch.optim.SGD([layer_params, ch_params], momentum=run.config["momentum"], weight_decay=run.config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=run.config["milestones"], gamma=run.config["gamma"])
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=run.config["label_smoothing"])

    train_losses = []
    for epoch in range(1, run.config["epochs"] + 1):
        print(f"Training, epoch {epoch}")
        train_loss = train(model, optimizer, trainloader, criterion, epoch, mixUp, run.config["alpha_mixUp"], device)
        train_losses.append(train_loss)
        metrics = {"Train Loss": train_loss}
        print(f"Epoch {epoch} mean Loss: {train_loss}")
        scheduler.step()

        val_loss, val_acc, class_acc = test(model, testloader, criterion, device=device)
        print(val_acc)
        val_metrics = {"Val_accuracy": val_acc}
        wandb.log({**metrics, **val_metrics})
    
    #torch.save(model.state_dict(), "model.pth")
    test_loss, test_acc, class_acc = test(model, testloader, criterion, device)
    test_metrics = {"Test Accuracy": test_acc}
    wandb.log({**test_metrics, **{f"Class {cls} Accuracy": acc for cls, acc in class_acc.items()}})
    wandb.finish()
    return test_loss, test_acc


def gradCAM(test_set, model_trained, originalImg_path, gradCAM_path, avgCAM_path, model, device='cuda'):
    model.load_state_dict(torch.load(model_trained))
    for param in model.parameters():
        param.requires_grad = True
    
    """ Model wrapper to return a tensor"""
    class HuggingfaceToTensorModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super(HuggingfaceToTensorModelWrapper, self).__init__()
            self.model = model

        def forward(self, x):
            return self.model(x).logits
    
    model = HuggingfaceToTensorModelWrapper(model)
    model.to(device)
    model.eval()

    dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    target_layers = [model.model.vit.encoder.layer[-2].output]

    def reshape_transform(tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        # Bring the channels to the first dimension, like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    # Apply Grad-CAM
    cam_object = EigenGradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    def compute_cam(image, cam_object):
        grayscale_cam = cam_object(image, targets=None)[0]  # targets=None -> highest scoring category
        grayscale_cam_resized = np.array(grayscale_cam)
        resized_cam = cv2.resize(grayscale_cam_resized, (224, 224))
        return resized_cam
    
    def reverse_transform(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        
        # Reverse normalization
        image_tensor = image_tensor * std + mean
        return image_tensor
    
    def save_visualization(visualization, predicted_class, ground_truth, path):
        plt.imshow(visualization)
        plt.axis("off")
        plt.title(f"Predicted: {predicted_class}, Ground Truth: {ground_truth}")
        plt.savefig(path + ".jpg", bbox_inches=None, pad_inches=0)
        plt.close()
    
    def save_original_image(image, originalImg_path):
        original_image = reverse_transform(image)  # Reversing the normalization for the first image in the batch
        original_image = original_image.squeeze(0)
        original_image = original_image.cpu().detach().numpy().transpose(1, 2, 0)  # Convert to numpy and reorder dimensions (C,H,W to H,W,C)

        # Clip the values to [0, 1] to handle potential overflow/underflow issues after reverse normalization
        original_image = np.clip(original_image, 0, 1)

        # Save original image reshaped to compare
        plt.imshow(original_image)
        plt.axis("off")
        plt.savefig(originalImg_path + ".jpg", bbox_inches=None, pad_inches=0)
        plt.close()
        return original_image
    
    count0=0
    accum_cam0 = None
    count0_1=0
    accum_cam0_1 = None
    count0_2=0
    accum_cam0_2 = None
    count1=0
    accum_cam1 = None
    count1_0=0
    accum_cam1_0 = None
    count1_2=0
    accum_cam1_2 = None
    count2=0
    accum_cam2 = None
    count2_0=0
    accum_cam2_0 = None
    count2_1=0
    accum_cam2_1 = None

    for image, label in dataloader:
        image, label = image.to(device), label.to(device)
        image.requires_grad_()

        # Get the predicted class
        output = model(image)
        predicted_class = output.argmax(dim=1).item()

        if label == 0:
            if predicted_class == 0:
                count0 += 1
                result_cam = compute_cam(image, cam_object)
                if accum_cam0 is None:
                    accum_cam0 = result_cam
                    original_image = save_original_image(image, originalImg_path + "-00")
                    visualization = show_cam_on_image(original_image, result_cam, use_rgb=True)
                    save_visualization(visualization, predicted_class + 1, label.item() + 1, gradCAM_path + "-00")
                else:
                    accum_cam0 += result_cam
            elif predicted_class == 1:
                count0_1 += 1
                result_cam = compute_cam(image, cam_object)
                if accum_cam0_1 is None:
                    accum_cam0_1 = result_cam
                    original_image = save_original_image(image, originalImg_path + "-01")
                    visualization = show_cam_on_image(original_image, result_cam, use_rgb=True)
                    save_visualization(visualization, predicted_class + 1, label.item() + 1, gradCAM_path + "-01")
                else:
                    accum_cam0_1 += result_cam
            else: # predicted_class == 2
                count0_2 += 1
                result_cam = compute_cam(image, cam_object)
                if accum_cam0_2 is None:
                    accum_cam0_2 = result_cam
                    original_image = save_original_image(image, originalImg_path + "-02")
                    visualization = show_cam_on_image(original_image, result_cam, use_rgb=True)
                    save_visualization(visualization, predicted_class + 1, label.item() + 1, gradCAM_path + "-02")
                else:
                    accum_cam0_2 += result_cam
        elif label == 1:
            if predicted_class == 0:
                count1_0 += 1
                result_cam = compute_cam(image, cam_object)
                if accum_cam1_0 is None:
                    accum_cam1_0 = result_cam
                    original_image = save_original_image(image, originalImg_path + "-10")
                    visualization = show_cam_on_image(original_image, result_cam, use_rgb=True)
                    save_visualization(visualization, predicted_class + 1, label.item() + 1, gradCAM_path + "-10")
                else:
                    accum_cam1_0 += result_cam
            elif predicted_class == 1:
                count1 += 1
                result_cam = compute_cam(image, cam_object)
                if accum_cam1 is None:
                    accum_cam1 = result_cam
                    original_image = save_original_image(image, originalImg_path + "-11")
                    visualization = show_cam_on_image(original_image, result_cam, use_rgb=True)
                    save_visualization(visualization, predicted_class + 1, label.item() + 1, gradCAM_path + "-11")
                else:
                    accum_cam1 += result_cam
            else: # predicted_class == 2
                count1_2 += 1
                result_cam = compute_cam(image, cam_object)
                if accum_cam1_2 is None:
                    accum_cam1_2 = result_cam
                    original_image = save_original_image(image, originalImg_path + "-12")
                    visualization = show_cam_on_image(original_image, result_cam, use_rgb=True)
                    save_visualization(visualization, predicted_class + 1, label.item() + 1, gradCAM_path + "-12")
                else:
                    accum_cam1_2 += result_cam
        else: # label == 2
            if predicted_class == 0:
                count2_0 += 1
                result_cam = compute_cam(image, cam_object)
                if accum_cam2_0 is None:
                    accum_cam2_0 = result_cam
                    original_image = save_original_image(image, originalImg_path + "-20")
                    visualization = show_cam_on_image(original_image, result_cam, use_rgb=True)
                    save_visualization(visualization, predicted_class + 1, label.item() + 1, gradCAM_path + "-20")
                else:
                    accum_cam2_0 += result_cam
            elif predicted_class == 1:
                count2_1 += 1
                result_cam = compute_cam(image, cam_object)
                if accum_cam2_1 is None:
                    accum_cam2_1 = result_cam
                    original_image = save_original_image(image, originalImg_path + "-21")
                    visualization = show_cam_on_image(original_image, result_cam, use_rgb=True)
                    save_visualization(visualization, predicted_class + 1, label.item() + 1, gradCAM_path + "-21")
                else:
                    accum_cam2_1 += result_cam
            else: # predicted_class == 2
                count2 += 1
                result_cam = compute_cam(image, cam_object)
                if accum_cam2 is None:
                    accum_cam2 = result_cam
                    original_image = save_original_image(image, originalImg_path + "-22")
                    visualization = show_cam_on_image(original_image, result_cam, use_rgb=True)
                    save_visualization(visualization, predicted_class + 1, label.item() + 1, gradCAM_path + "-22")
                else:
                    accum_cam2 += result_cam

    # Visualize average cams on a blank image
    blank_img = np.ones((224, 224, 3), dtype=np.float32)
    if count0 != 0:
        visualization0 = show_cam_on_image(blank_img, accum_cam0 / count0, use_rgb=True)
        save_visualization(visualization0, predicted_class=1, ground_truth=1, path=avgCAM_path + "-00")
    else:
        print("No predictions of Class 1 were right")
    if count0_1 != 0:
        visualization0_1 = show_cam_on_image(blank_img, accum_cam0_1 / count0_1, use_rgb=True)
        save_visualization(visualization0_1, predicted_class=2, ground_truth=1, path=avgCAM_path + "-01")
    else:
        print("Class 1 was never misclassified with Class 1")
    if count0_2 != 0:
        visualization0_2 = show_cam_on_image(blank_img, accum_cam0_2 / count0_2, use_rgb=True)
        save_visualization(visualization0_2, predicted_class=3, ground_truth=1, path=avgCAM_path + "-02")
    else:
        print("Class 1 was never misclassified with Class 3")
    if count1_0 != 0:
        visualization1_0 = show_cam_on_image(blank_img, accum_cam1_0 / count1_0, use_rgb=True)
        save_visualization(visualization1_0, predicted_class=1, ground_truth=2, path=avgCAM_path + "-10")
    else:
        print("Class 2 was never misclassified with Class 1")
    if count1 != 0:
        visualization1 = show_cam_on_image(blank_img, accum_cam1 / count1, use_rgb=True)
        save_visualization(visualization1, predicted_class=2, ground_truth=2, path=avgCAM_path + "-11")
    else:
        print("No predictions of Class 2 were right")
    if count1_2 != 0:
        visualization1_2 = show_cam_on_image(blank_img, accum_cam1_2 / count1_2, use_rgb=True)
        save_visualization(visualization1_2, predicted_class=3, ground_truth=2, path=avgCAM_path + "-12")
    else:
        print("Class 2 was never misclassified with Class 3")
    if count2_0 != 0:
        visualization2_0 = show_cam_on_image(blank_img, accum_cam2_0 / count2_0, use_rgb=True)
        save_visualization(visualization2_0, predicted_class=1, ground_truth=3, path=avgCAM_path + "-20")
    else:
        print("Class 3 was never misclassified with Class 1")
    if count2_1 != 0:
        visualization2_1 = show_cam_on_image(blank_img, accum_cam2_1 / count2_1, use_rgb=True)
        save_visualization(visualization2_1, predicted_class=2, ground_truth=3, path=avgCAM_path + "-21")
    else:
        print("Class 3 was never misclassified with Class 2")
    if count2 != 0:
        visualization2 = show_cam_on_image(blank_img, accum_cam2 / count2, use_rgb=True)
        save_visualization(visualization2, predicted_class=3, ground_truth=3, path=avgCAM_path + "-22")
    else:
        print("No predictions of Class 3 were right")


def main(labels_file, img_directory, default_parcel, fineTuning, num_layer, gradcam, model_trained, img_path, gradCAM_path, avgCAM_path, mixUp, lora):
    wandb.login()
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

    # ch fine tuning
    if fineTuning == 0:
        model_name = "google/vit-base-patch16-224-in21k"
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            id2label={str(i): c for i, c in enumerate(set(labels.tolist()))},
            label2id={c: str(i) for i, c in enumerate(set(labels.tolist()))},
        )
        if num_layer > 0:
            model.classifier = CustomMLPHead(
                input_dim=model.config.hidden_size,
                hidden_dim=256,
                num_layers=num_layer,
                num_classes=num_classes
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
        if gradcam:
            gradCAM(test_set, model_trained, img_path, gradCAM_path, avgCAM_path, model, device='cuda')
        else:
            test_loss, test_acc = chFineTuning(model, trainloader, testloader, mixUp, lora, device='cuda')

    else:
        # fine tuning last layer
        model_name = "google/vit-base-patch16-224-in21k"
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            id2label={str(i): c for i, c in enumerate(set(labels.tolist()))},
            label2id={c: str(i) for i, c in enumerate(set(labels.tolist()))},
        )
        if gradcam:
            gradCAM(test_set, model_trained, img_path, gradCAM_path, avgCAM_path, model, device='cuda')
        else:
            test_loss, test_acc = layerFineTuning(model, trainloader, testloader, mixUp, toFinetune=fineTuning, device='cuda')
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset locations, partial experiment setup and results locations")
    parser.add_argument("--labels_file", type=str, required=True, help="path of the xlsx file related to classes")
    parser.add_argument("--img_directory", type=str, required=True, help="path of the dataset")
    parser.add_argument("--default_parcel", type=int, required=True, help="required for images with wrong names")
    parser.add_argument("--fineTuning", type=int, required=True, help="setup of finetuning")
    parser.add_argument("--linearLayers", type=int, required=True, help="defines MLP as classification head")
    parser.add_argument("--gradCAM", action="store_true", help="performs gradCAM")
    parser.add_argument("--noGradCAM", action="store_false", dest="gradCAM", help="does not perform gradCAM")
    parser.add_argument("--model", type=str, required=False, help="if gradCAM is True, patho of the trained model")
    parser.add_argument("--originalImg_path", type=str, required=False, help="if gradCAM is True, path where to save original image")
    parser.add_argument("--gradCAM_path", type=str, required=False, help="if gradCAM is True, path where to save image with heatmap")
    parser.add_argument("--avgCAM_path", type=str, required=False, help="if gradCAM is True, path where to save average cams")
    parser.add_argument("--mixUp", action="store_true", help="performs MixUp data augmentation")
    parser.add_argument("--noMixUp", action="store_false", dest="mixUp", help="does not perform MixUp")
    parser.add_argument("--loRA", action="store_true", help="performs LoRA")
    parser.add_argument("--noLoRA", action="store_false", dest="loRA", help="does not perform LoRA")
    args = parser.parse_args()
    main(labels_file=args.labels_file, img_directory=args.img_directory, default_parcel=args.default_parcel, fineTuning=args.fineTuning,
         num_layer=args.linearLayers, gradcam=args.gradCAM, model_trained=args.model, img_path=args.originalImg_path, gradCAM_path=args.gradCAM_path,
         avgCAM_path=args.avgCAM_path, mixUp=args.mixUp, lora=args.loRA)