from datasetStudy import *
from torch.utils.data import DataLoader, Subset
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import wandb
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from peft import LoraConfig, get_peft_model, TaskType

def train(model, optimizer, trainloader, criterion, epoch, mixUp, device='cuda'):
    model.train()
    train_losses = []
    mixup = None
    if mixUp:
        mixup = v2.CutMix(alpha=1.0, num_classes=3)
    for data in trainloader:
        images, labels = data
        if mixup is not None:
            images, labels = mixup(images, labels)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        print(f"train loss: {loss.item()}")
    print(f"Epoch {epoch} mean loss: {np.mean(train_losses)}")
    return np.mean(train_losses)

def test(model, testloader, criterion, test=False, device='cuda'):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            test_loss += criterion(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_acc = correct / len(testloader.dataset)
    if test:
        print(f"Test accuracy: {test_acc:.3f}")

    else:
        print(f"Validation accuracy: {test_acc:.3f}")
    return test_loss, test_acc


def chFineTuning(model, trainloader, valloader, testloader, mixUp, device='cuda'):
    run = wandb.init(
    # Set the project where this run will be logged
    project="Computer Vision ViT balanced",
    # Pass a run name
    name=f"ViT_h_14 classification head",
    # Track hyperparameters and run metadata
    config={
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "architecture": "ViT_h_14",
    "dataset": "cnr",
    "epochs": 25})

    model.to(device)
    # frozen all the weights of the network, except for fc ones
    for param in model.parameters():
        param.requires_grad = False
    model.heads.head.weight.requires_grad = True
    model.heads.head.bias.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=run.config["learning_rate"], weight_decay=run.config["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    best_accuracy = 0
    state_dict = None
    for epoch in range(1, run.config["epochs"] + 1):
        print(f"Training, epoch {epoch}")
        train_loss = train(model, optimizer, trainloader, criterion, epoch, mixUp, device)
        train_losses.append(train_loss)
        metrics = {"Train Loss": train_loss}
        print(f"Epoch {epoch} mean Loss: {train_loss}")

        # Save best model during validation
        val_loss, val_acc = test(model, valloader, criterion, test=False, device=device)
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
        if val_acc > best_accuracy:
            state_dict = model.state_dict()
            best_accuracy = val_acc
            torch.save(model.state_dict(), "model.pth")
        val_metrics = {"Validation Loss": val_loss, "Validation Accuracy": val_acc}
        wandb.log({**metrics, **val_metrics})
    
    model.load_state_dict(state_dict)
    test_loss, test_acc = test(model, testloader, criterion, test=True, device=device)
    test_metrics = {"Test Accuracy": test_acc}
    wandb.log({**test_metrics})
    wandb.finish()
    print(f"Best model acc: {test_acc}")
    return test_loss, test_acc

def layerFineTuning(model, trainloader, valloader, testloader, mixUp, toFinetune=1, device='cuda'):
    run = wandb.init(
    # Set the project where this run will be logged
    project="Computer Vision ViT balanced",
    # Pass a run name
    name=f"ViT_b fineTuning 5 MixUp",
    # Track hyperparameters and run metadata
    config={
    "learning_rate": 1e-4,
    "learning_rate2": 1e-5,
    "weight_decay": 1e-4,
    "architecture": "ViT_b_16",
    "dataset": "cnr",
    "epochs": 25})
    model.to(device)
    # frozen all the weights of the network, except for fc ones and last layer
    for param in model.parameters():
        param.requires_grad = False
    model.heads.head.weight.requires_grad = True
    model.heads.head.bias.requires_grad = True
    for param in model.encoder.layers[-toFinetune:].parameters():
        param.requires_grad = True
    ch_params = {'params': model.heads.head.parameters(), 'lr': run.config["learning_rate"]}
    layer_params = {'params': model.encoder.layers[-toFinetune:].parameters(), 'lr': run.config["learning_rate2"]}
    optimizer = torch.optim.AdamW([layer_params, ch_params], weight_decay=run.config["weight_decay"])
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    best_accuracy = 0
    state_dict = None
    for epoch in range(1, run.config["epochs"] + 1):
        print(f"Training, epoch {epoch}")
        train_loss = train(model, optimizer, trainloader, criterion, epoch, mixUp, device)
        train_losses.append(train_loss)
        metrics = {"Train Loss": train_loss}
        print(f"Epoch {epoch} mean Loss: {train_loss}")

        # Save best model during validation
        val_loss, val_acc = test(model, valloader, criterion, test=False, device=device)
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
        if val_acc >= best_accuracy:
            state_dict = model.state_dict()
            best_accuracy = val_acc
            torch.save(model.state_dict(), "model.pth")
        val_metrics = {"Validation Loss": val_loss, "Validation Accuracy": val_acc}
        wandb.log({**metrics, **val_metrics})
    
    model.load_state_dict(state_dict)
    test_loss, test_acc = test(model, testloader, criterion, device)
    test_metrics = {"Test Accuracy": test_acc}
    wandb.log({**test_metrics})
    wandb.finish()
    return test_loss, test_acc


def gradCAM(test_set, originalImg_path, gradCAM_path, device='cuda'):
    model = models.vit_b_16(pretrained=False)
    num_features = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(num_features, 3)
    model.load_state_dict(torch.load("model.pth"))
    for param in model.parameters():
        param.requires_grad = True
    model.to('cuda')
    model.eval()
    
    def reverse_transform(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        
        # Reverse normalization
        image_tensor = image_tensor * std + mean
        return image_tensor

    dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    target_layers = [model.encoder.layers[-1].ln_1]

    def reshape_transform(tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        # Bring the channels to the first dimension, like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    # Apply Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    for image, label in dataloader:
        image, label = image.to(device), label.to(device)
        image.requires_grad_()

        # Get the predicted class
        output = model(image)
        predicted_class = output.argmax(dim=1).item()

        # Compute Grad-CAM for the predicted class
        grayscale_cam = cam(image, targets=None)[0]  # targets=None -> highest scoring category

        original_image = reverse_transform(image)  # Reversing the normalization for the first image in the batch
        original_image = original_image.squeeze(0)
        original_image = original_image.cpu().detach().numpy().transpose(1, 2, 0)  # Convert to numpy and reorder dimensions (C,H,W to H,W,C)

        # Clip the values to [0, 1] to handle potential overflow/underflow issues after reverse normalization
        original_image = np.clip(original_image, 0, 1)

        # Save original image reshaped to compare
        plt.imshow(original_image)
        plt.axis("off")
        plt.savefig(originalImg_path)
        plt.close()

        # Resize Grad-CAM to match the input size (224x224)
        grayscale_cam_resized = np.array(grayscale_cam)
        resized_cam = cv2.resize(grayscale_cam_resized, (224, 224))

        # Visualize Grad-CAM on the original image
        visualization = show_cam_on_image(original_image, resized_cam, use_rgb=True)

        # Show Grad-CAM visualization
        plt.imshow(visualization)
        plt.axis("off")
        plt.savefig(gradCAM_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        break        


def main(labels_file, img_directory, default_parcel, fineTuning, gradcam, img_path, gradCAM_path, mixUp, lora):
    wandb.login()
    val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ViT input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = CustomImageDataset(annotations_file=labels_file, img_dir=img_directory, default_parcel=default_parcel, train=True, transform=val_transforms)
    X = torch.arange(len(dataset))  # Indices of dataset
    y = torch.tensor([dataset[i][1] for i in range(len(dataset))])  # Extract labels
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=111)
    for train_idx, temp_idx in splitter.split(X, y):
        train_set = Subset(dataset, train_idx)
        temp_set = Subset(dataset, temp_idx)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=111)
    for val_idx, test_idx in splitter.split(X[temp_idx], y[temp_idx]):
        val_set = Subset(dataset, temp_idx[val_idx])
        test_set = Subset(dataset, temp_idx[test_idx])

    torch.manual_seed(111)
    val_set.dataset.train = False
    test_set.dataset.train = False
    trainloader = DataLoader(train_set, batch_size=32, shuffle=True)
    valloader = DataLoader(val_set, batch_size=32, shuffle=False)
    testloader = DataLoader(test_set, batch_size=32, shuffle=False)

    
    # ch fine tuning
    if fineTuning == 0:
        model = models.vit_h_14(weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
        num_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(num_features, 3) # change classification head to match number of classes needed
        if lora: # TODO fix
            target_modules = ["encoder.layers.encoder_layer_0.self_attention.out_proj",
                            "encoder.layers.encoder_layer_1.self_attention.out_proj",
                            "encoder.layers.encoder_layer_2.self_attention.out_proj",
                            "encoder.layers.encoder_layer_3.self_attention.out_proj",
                            "encoder.layers.encoder_layer_4.self_attention.out_proj",
                            "encoder.layers.encoder_layer_5.self_attention.out_proj",
                            "encoder.layers.encoder_layer_6.self_attention.out_proj",
                            "encoder.layers.encoder_layer_7.self_attention.out_proj",
                            "encoder.layers.encoder_layer_8.self_attention.out_proj",
                            "encoder.layers.encoder_layer_9.self_attention.out_proj",
                            "encoder.layers.encoder_layer_10.self_attention.out_proj",
                            "encoder.layers.encoder_layer_11.self_attention.out_proj"]
            lora_config = LoraConfig(r=8, lora_alpha=16, target_modules = target_modules,
                                      lora_dropout=0.1, bias="none", task_type=TaskType.FEATURE_EXTRACTION)
            model = get_peft_model(model, lora_config)
        test_loss, test_acc = chFineTuning(model, trainloader, valloader, testloader, mixUp, device='cuda')
        if gradcam:
            gradCAM(test_set, img_path, gradCAM_path, device='cuda')

    else:
        # fine tuning last layer
        model = models.vit_b_16(pretrained=True)
        num_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(num_features, 3) # change classification head to match number of classes needed
        test_loss, test_acc = layerFineTuning(model, trainloader, valloader, testloader, mixUp, toFinetune=fineTuning, device='cuda')
        if gradcam:
            gradCAM(test_set, img_path, gradCAM_path, device='cuda')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset locations, partial experiment setup and results locations")
    parser.add_argument("--labels_file", type=str, required=True, help="path of the xlsx file related to classes")
    parser.add_argument("--img_directory", type=str, required=True, help="path of the dataset")
    parser.add_argument("--default_parcel", type=int, required=True, help="required for images with wrong names")
    parser.add_argument("--fineTuning", type=int, required=True, help="setup of finetuning")
    parser.add_argument("--gradCAM", action="store_true", help="performs gradCAM")
    parser.add_argument("--noGradCAM", action="store_false", dest="gradCAM", help="does not perform gradCAM")
    parser.add_argument("--originalImg_path", type=str, required=False, help="if gradCAM is True, path where to save original image")
    parser.add_argument("--gradCAM_path", type=str, required=False, help="if gradCAM is True, path where to save image with heatmap")
    parser.add_argument("--mixUp", action="store_true", help="performs MixUp data augmentation")
    parser.add_argument("--noMixUp", action="store_false", dest="mixUp", help="does not perform MixUp")
    parser.add_argument("--loRA", action="store_true", help="performs LoRA")
    parser.add_argument("--noLoRA", action="store_false", dest="loRA", help="does not perform LoRA")
    args = parser.parse_args()
    main(labels_file=args.labels_file, img_directory=args.img_directory, default_parcel=args.default_parcel, fineTuning=args.fineTuning,
         gradcam=args.gradCAM, img_path=args.originalImg_path, gradCAM_path=args.gradCAM_path, mixUp=args.mixUp, lora=args.loRA)