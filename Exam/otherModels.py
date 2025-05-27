from datasetStudy import *
from ViTFineTuning import train, test
import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from transformers import ViTForImageClassification, AutoModelForImageClassification


def chFineTuningResNet(model, trainloader, testloader, mixUp, lora=False, device='cuda'):
    run = wandb.init(
    # Set the project where this run will be logged
    project="Computer Vision ViT hf Final Tests foo",
    # Pass a run name
    name=f"ResNet18 classification 100",
    # Track hyperparameters and run metadata
    config={
    "learning_rate": 1e-2,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "milestones": [35, 60, 85],
    "gamma": 0.1,
    "alphaMixUp": 0.0,
    "label_smoothing": 0.0,
    "architecture": "ResNet18",
    "dataset": "cnr",
    "epochs": 100})

    model.to(device)
    if not lora:
        # Freeze all the weights of the network, except for classifier head ones
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1].weight.requires_grad = True
        model.classifier[1].bias.requires_grad = True
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
    
    torch.save(model.state_dict(), "model_lora_ls.pth")
    test_loss, test_acc, class_acc = test(model, testloader, criterion, device=device)
    test_metrics = {"Test Accuracy": test_acc}
    wandb.log({**test_metrics}, **{f"Class {cls} Accuracy": acc for cls, acc in class_acc.items()})
    wandb.finish()
    return test_loss, test_acc

def layerFineTuningResNet(model, trainloader, testloader, mixUp, toFinetune=1, device='cuda'):
    run = wandb.init(
    # Set the project where this run will be logged
    project="Computer Vision ViT hf Final Tests foo",
    # Pass a run name
    name=f"ViT_l_16 fineTuning 6-2 100",
    # Track hyperparameters and run metadata
    config={
    "learning_rate": 1e-2,
    "momentum": 0.9,
    "learning_rate2": 1e-3,
    "weight_decay": 1e-4,
    "milestones": [25, 50, 85],
    "gamma": 0.1,
    "alpha_mixUp": 0.0,
    "label_smoothing": 0.0,
    "architecture": "ViT_l_16",
    "dataset": "cnr",
    "epochs": 100})
    model.to(device)
    # frozen all the weights of the network, except for fc ones and last layer
    for param in model.parameters():
        param.requires_grad = False
    '''#ResNet
    model.classifier[1].weight.requires_grad = True
    model.classifier[1].bias.requires_grad = True
    for param in model.resnet.encoder.stages[-toFinetune:].parameters():
        param.requires_grad = True
    ch_params = {'params': model.classifier.parameters(), 'lr': run.config["learning_rate"]}
    layer_params = {'params': model.resnet.encoder.stages[-toFinetune:].parameters(), 'lr': run.config["learning_rate2"]}
    '''
    #ViT
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
    
    torch.save(model.state_dict(), "models_gradcam/modells.pth")
    test_loss, test_acc, class_acc = test(model, testloader, criterion, device)
    test_metrics = {"Test Accuracy": test_acc}
    wandb.log({**test_metrics, **{f"Class {cls} Accuracy": acc for cls, acc in class_acc.items()}})
    wandb.finish()
    return test_loss, test_acc

def main(labels_file, img_directory, default_parcel, fineTuning, vit):
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
    model = None
    
    # ch fine tuning
    if fineTuning == 0:
        if vit:
            #model_name = "google/vit-base-patch32-224-in21k"
            model_name = "google/vit-large-patch16-224-in21k"
            #model_name = "google/vit-huge-patch14-224-in21k"
            model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                id2label={str(i): c for i, c in enumerate(set(labels.tolist()))},
                label2id={c: str(i) for i, c in enumerate(set(labels.tolist()))},
            )
        else:
            model_name = "microsoft/resnet-18"
            model = AutoModelForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
            model.classifier = torch.nn.Sequential(
                torch.nn.Flatten(),  # Converts [B, 512, 1, 1] -> [B, 512]
                torch.nn.Linear(512, 3)
            ) # TODO when running this code, change in ViTFineTuning.py model.classifier[1].weight.requires_grad = True
        test_loss, test_acc = chFineTuningResNet(model, trainloader, testloader, False, device='cuda')
    
    else:
        # fine tuning last layer
        if vit:
            #model_name = "google/vit-base-patch32-224-in21k"
            model_name = "google/vit-large-patch16-224-in21k"
            #model_name = "google/vit-huge-patch14-224-in21k"
            model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                id2label={str(i): c for i, c in enumerate(set(labels.tolist()))},
                label2id={c: str(i) for i, c in enumerate(set(labels.tolist()))},
            )
        else:
            model_name = "microsoft/resnet-18"
            model = AutoModelForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
            model.classifier = torch.nn.Sequential(
                torch.nn.Flatten(),  # Converts [B, 512, 1, 1] -> [B, 512]
                torch.nn.Linear(512, 3)
            ) # TODO when running this code, change in ViTFineTuning.py model.classifier[1].weight.requires_grad = True
        test_loss, test_acc = layerFineTuningResNet(model, trainloader, testloader, False, toFinetune=fineTuning, device='cuda')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Locations')
    parser.add_argument('--labels_file', metavar='path', required=True)
    parser.add_argument('--img_directory', metavar='path', required=True)
    parser.add_argument('--default_parcel', metavar='int', required=True)
    parser.add_argument("--fineTuning", type=int, required=True, help="setup of finetuning")
    parser.add_argument("--vit", action="store_true", help="fineTuning on a vit_b_32 model")
    parser.add_argument("--noViT", action="store_false", dest="loRA", help="fineTuning on a resNet18 model")
    args = parser.parse_args()
    main(labels_file=args.labels_file, img_directory=args.img_directory, default_parcel=args.default_parcel, fineTuning=args.fineTuning, vit=args.vit)