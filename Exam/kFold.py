from ViTFineTuning import *
from datasetStudy import *
from sklearn.model_selection import KFold


def main(labels_file, img_directory, default_parcel, fineTuning):
    wandb.login()
    train_transforms = transforms.Compose([  
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    torch.manual_seed(111)
    np.random.seed(111)

    dataset = CustomImageDataset(annotations_file=labels_file, img_dir=img_directory, default_parcel=default_parcel, transform=train_transforms)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(set(labels.tolist()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = wandb.init(
    # Set the project where this run will be logged
    project="Computer Vision ViT hf Kfold",
    # Pass a run name
    name=f"ViT_b_16 10-Fold 4.1",
    # Track hyperparameters and run metadata
    config={
    "learning_rate": 1e-2,
    "momentum": 0.9,
    "learning_rate2": 1e-3,
    "weight_decay": 1e-4,
    "milestones": [25, 50, 85],
    "gamma": 0.5,
    "alpha_mixUp": 0.5,
    "label_smoothing": 0.0,
    "k_folds": 10,
    "architecture": "ViT_b_16",
    "dataset": "cnr",
    "epochs": 100})

    kf = KFold(n_splits=run.config["k_folds"], shuffle=True, random_state=111)
    indices = list(range(len(dataset)))

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        k_folds = run.config["k_folds"]
        print(f"Fold {fold+1}/{k_folds}")
        
        # Create data loaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
        valloader = DataLoader(val_subset, batch_size=32, shuffle=False)
        
        # Load ViT model
        model_name = "google/vit-base-patch16-224-in21k"
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            id2label={str(i): c for i, c in enumerate(set(labels.tolist()))},
            label2id={c: str(i) for i, c in enumerate(set(labels.tolist()))},
        )
        model.to(device)
        
        # Define loss function and optimizer
        ch_params = {'params': model.classifier.parameters(), 'lr': run.config["learning_rate"]}
        layer_params = {'params': model.vit.encoder.layer[-fineTuning:].parameters(), 'lr': run.config["learning_rate2"]}
        optimizer = torch.optim.SGD([layer_params, ch_params], momentum=run.config["momentum"], weight_decay=run.config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=run.config["milestones"], gamma=run.config["gamma"])
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=run.config["label_smoothing"])    
        
        # Training loop
        dataset.transform = train_transforms
        train_losses = []
        for epoch in range(1, run.config["epochs"] + 1):
            print(f"Training, epoch {epoch}")
            train_loss = train(model, optimizer, trainloader, criterion, epoch, mixUp=False, alpha=run.config["alpha_mixUp"], device=device)
            train_losses.append(train_loss)
            metrics = {f"Train Loss {fold}": train_loss}
            print(f"Epoch {epoch} mean Loss: {train_loss}")
            scheduler.step()

            val_loss, val_acc, class_acc = test(model, valloader, criterion, device=device)
            print(val_acc)
            val_metrics = {f"Val_accuracy {fold}": val_acc}
            wandb.log({**metrics, **val_metrics})
        
        # Validation loop
        dataset.transform = test_transforms
        torch.save(model.state_dict(), "model.pth")
        test_loss, test_acc, class_acc = test(model, valloader, criterion, device)
        test_metrics = {f"Test Accuracy {fold}": test_acc}
        wandb.log({**test_metrics, **{f"Class {cls} Accuracy {fold}": acc for cls, acc in class_acc.items()}})
        print(f"Fold {fold+1} Accuracy: {test_acc:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Locations')
    parser.add_argument('--labels_file', metavar='path', required=True)
    parser.add_argument('--img_directory', metavar='path', required=True)
    parser.add_argument('--default_parcel', metavar='int', required=True)
    parser.add_argument("--fineTuning", type=int, required=True, help="setup of finetuning")
    args = parser.parse_args()
    main(labels_file=args.labels_file, img_directory=args.img_directory, default_parcel=args.default_parcel, fineTuning=args.fineTuning)