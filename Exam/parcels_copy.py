from ViTFineTuning import *
from datasetStudy import *
from sklearn.model_selection import KFold
from collections import defaultdict
import torch.nn.functional as F

def get_parcel_to_filenames(excel_file):
    df = pd.read_excel(excel_file)
    parcel_to_files = defaultdict(list)

    for _, row in df.iterrows():
        parcel = row["parcel"]
        pic = row["pic"]
        filename = f"{parcel}_{pic}.jpg"
        parcel_to_files[parcel].append(filename)

    return parcel_to_files

def get_kfold_image_splits(parcel_to_files, n_splits=5, seed=41):
    parcels = list(parcel_to_files.keys())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    splits = []
    for train_idx, test_idx in kf.split(parcels):
        train_files = []
        test_files = []
        train_parcels = [parcels[i] for i in train_idx]
        test_parcels = [parcels[i] for i in test_idx]

        for parcel in train_parcels:
            train_files.extend(parcel_to_files[parcel])
        for parcel in test_parcels:
            test_files.extend(parcel_to_files[parcel])

        splits.append((train_files, test_files))
    
    return splits

def testFold(model, testloader, criterion, scores=False, device='cuda'):
    model.eval()
    test_loss = 0
    correct = 0
    class_correct = Counter()
    class_total = Counter()
    all_predictions = []
    all_labels = []
    all_parcels = []

    with torch.no_grad():
        for data in testloader:
            images, labels, parcels = data
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            logits = output.logits
            test_loss += criterion(logits, labels).item()
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()

            for label, pred, parcel in zip(labels, predictions, parcels):
                class_total[label.item()] += 1
                if label.item() == pred.item():
                    class_correct[label.item()] += 1
                
                all_predictions.append(pred.item())
                all_labels.append(label.item())
                all_parcels.append(parcel)
    test_acc = correct / len(testloader.dataset)
    class_accuracy = {cls: (class_correct[cls] / class_total[cls]) if class_total[cls] > 0 else 0.0 for cls in class_total}
    mean_test_loss = test_loss / len(testloader.dataset)
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2])

    print(f"Test accuracy: {test_acc:.3f}")
    for cls, acc in class_accuracy.items():
        print(f"Class {cls} Accuracy: {acc:.2f}")
    
    # Group predictions and labels by parcel
    parcel_preds = defaultdict(list)
    parcel_labels = defaultdict(list)

    all_parcel_labels = []
    all_parcel_predictions = []

    for pred, label, parcel in zip(all_predictions, all_labels, all_parcels):
        parcel_preds[parcel.item()].append(pred)
        parcel_labels[parcel.item()].append(label)

    parcel_true_labels = {}
    parcel_final_preds = {}

    for parcel in parcel_preds:
        pred_avg = round(np.mean(parcel_preds[parcel]))  # Voting by average
        label_avg = round(np.mean(parcel_labels[parcel]))

        parcel_final_preds[parcel] = pred_avg
        parcel_true_labels[parcel] = label_avg

    parcel_total = Counter()
    parcel_correct = Counter()

    for parcel in parcel_true_labels:
        true = parcel_true_labels[parcel]
        pred = parcel_final_preds[parcel]
        all_parcel_labels.append(true)
        all_parcel_predictions.append(pred)

        parcel_total[true] += 1
        if true == pred:
            parcel_correct[true] += 1

    # Compute overall parcel-level accuracy
    total_correct = sum(parcel_correct.values())
    total_parcels = sum(parcel_total.values())
    parcel_acc = total_correct / total_parcels

    # Compute per-class parcel-level accuracy
    parcel_class_acc = {
        cls: parcel_correct[cls] / parcel_total[cls] if parcel_total[cls] > 0 else 0.0
        for cls in parcel_total
    }

    print(f"Parcel-level Accuracy: {parcel_acc:.4f}")
    for cls, acc in parcel_class_acc.items():
        print(f"Class {cls} Parcel Accuracy: {acc:.4f}")
    
    cm2 = confusion_matrix(all_parcel_labels, all_parcel_predictions, labels=[0, 1, 2])

    if scores:
        print("Plotting scores")
        with torch.no_grad():
            for idx, (images, labels, parcels) in enumerate(testloader):  # Assuming parcels are there too
                images, labels = images.to(device), labels.to(device)
                
                output = model(images)
                logits = output.logits  # (batch_size, num_classes)
                probs = F.softmax(logits, dim=1)  # Convert to probabilities
                
                for i in range(images.size(0)):
                    print(f"Plotting image {i} of batch {idx}")
                    img = images[i].cpu()
                    label = labels[i].item()
                    prob = probs[i].cpu().numpy()

                    # If your images were normalized, you might want to denormalize them for plotting
                    inv_transform = transforms.Normalize(
                        mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],  # Example for mean=0.5, std=0.5
                        std=[1/0.5, 1/0.5, 1/0.5]
                    )
                    img = inv_transform(img)
                    img = torch.clamp(img, 0, 1)  # Clip values to valid range

                    # Plot
                    plt.figure(figsize=(5, 4))
                    plt.imshow(img.permute(1, 2, 0))  # C x H x W -> H x W x C
                    plt.axis('off')
                    plt.title(f"True Label: {label}, Scores: {prob.round(2)} \n Parcel True Label: {parcel_true_labels[parcels[i]]}, Parcel Predicted Label: {parcel_final_preds[parcels[i]]}")
                    plt.savefig(f"/home/narati/cnr/scores/image_{idx}_{i}.jpg")

    return mean_test_loss, test_acc, class_accuracy, cm, cm2, parcel_acc, parcel_class_acc

def main(labels_file, img_directory, default_parcel, cmPath, fineTuning, k, lora):
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

    parcel_to_files = get_parcel_to_filenames(labels_file)
    splits = get_kfold_image_splits(parcel_to_files, n_splits=k)

    # Load standard dataset just to have info on num_classes and single labels
    dataset = CustomImageDataset(annotations_file=labels_file, img_dir=img_directory, default_parcel=default_parcel, transform=train_transforms)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(set(labels.tolist()))

    run = wandb.init(
        # Set the project where this run will be logged
        project="Computer Vision ViT hf Parcels",
        # Pass a run name
        name=f"ViT_b_16 5-Fold 6-2 Final scores [DELETE]",
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
        "architecture": "ViT_b_16",
        "dataset": "cnr",
        "epochs": 100})

    fold_metrics = {
    "overall_acc": [],
    "class_acc": defaultdict(list),  # class_id -> [acc_fold1, acc_fold2, ...]
    "overall_acc_parcel": [],
    "class_acc_parcel": defaultdict(list)
    }
    confusion_matrices = None
    confusion_matrices_parcel = None

    for fold, (train_files, test_files) in enumerate(splits):
        print(f"\n--- Fold {fold + 1} ---")
        
        train_dataset = CustomImageDataset(labels_file, img_directory, default_parcel, transform=train_transforms, file_filter=train_files)
        test_dataset = CustomImageDataset(labels_file, img_directory, default_parcel, transform=test_transforms, file_filter=test_files)

        trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = "google/vit-base-patch16-224-in21k"
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            id2label={str(i): c for i, c in enumerate(set(labels.tolist()))},
            label2id={c: str(i) for i, c in enumerate(set(labels.tolist()))},
        )
        model.to(device)

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
            optimizer = torch.optim.SGD(model.parameters(), lr=run.config["learning_rate"], momentum=run.config["momentum"], weight_decay=run.config["weight_decay"])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=run.config["milestones"], gamma=run.config["gamma"])
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=run.config["label_smoothing"])
        else:
            for param in model.parameters():
                param.requires_grad = False
            model.classifier.weight.requires_grad = True
            model.classifier.bias.requires_grad = True
            for param in model.vit.encoder.layer[-fineTuning:].parameters():
                param.requires_grad = True
            ch_params = {'params': model.classifier.parameters(), 'lr': run.config["learning_rate"]}
            layer_params = {'params': model.vit.encoder.layer[-fineTuning:].parameters(), 'lr': run.config["learning_rate2"]}
            optimizer = torch.optim.SGD([layer_params, ch_params], momentum=run.config["momentum"], weight_decay=run.config["weight_decay"])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=run.config["milestones"], gamma=run.config["gamma"])
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=run.config["label_smoothing"])

        # Training loop
        train_losses = []
        for epoch in range(1, run.config["epochs"] + 1):
            print(f"Training, epoch {epoch}")
            train_loss = train(model, optimizer, trainloader, criterion, epoch, mixUp=False, alpha=run.config["alpha_mixUp"], device=device)
            train_losses.append(train_loss)
            metrics = {f"Train Loss {fold}": train_loss}
            print(f"Epoch {epoch} mean Loss: {train_loss}")
            scheduler.step()

            val_loss, val_acc, class_acc, _, _, parcel_acc, parcel_class_acc = testFold(model, testloader, criterion, False, device=device)
            val_metrics = {f"Val_accuracy {fold}": val_acc, f"Parcel_accuracy {fold}": parcel_acc}
            wandb.log({**metrics, **val_metrics})
        
        # Validation loop
        torch.save(model.state_dict(), "model.pth")
        if fold == 1:
            print("Test Fold 2")
            test_loss, test_acc, class_acc, cm, cm2, test_parcel_acc, test_parcel_class_acc = testFold(model, testloader, criterion, True, device)
        else:    
            test_loss, test_acc, class_acc, cm, cm2, test_parcel_acc, test_parcel_class_acc = testFold(model, testloader, criterion, False, device)
        test_metrics = {f"Test Accuracy {fold}": test_acc, f"Test Parcel Accuracy {fold}": test_parcel_acc}
        wandb.log({**test_metrics, **{f"Class {cls} Accuracy {fold}": acc for cls, acc in class_acc.items()}, **{f"Class {cls} Parcel Accuracy {fold}": acc for cls, acc in test_parcel_class_acc.items()}})
        print(f"Fold {fold+1} Accuracy: {test_acc:.2f}%")

        fold_metrics["overall_acc"].append(test_acc)
        fold_metrics["overall_acc_parcel"].append(test_parcel_acc)
        for cls, acc in class_acc.items():
            fold_metrics["class_acc"][cls].append(acc)
        for cls, acc in test_parcel_class_acc.items():
            fold_metrics["class_acc_parcel"][cls].append(acc)
        
        if fold == 0:
            confusion_matrices = cm
            confusion_matrices_parcel = cm2
        else:
            confusion_matrices += cm
            confusion_matrices_parcel += cm2

    print("\n=== Cross-Validation Summary ===")

    # Overall Accuracy
    overall_mean = np.mean(fold_metrics["overall_acc"])
    overall_std = np.std(fold_metrics["overall_acc"])
    print(f"Overall Accuracy: {overall_mean:.4f} ± {overall_std:.4f}")

    # Per-Class Accuracy
    for cls in sorted(fold_metrics["class_acc"].keys()):
        class_accs = fold_metrics["class_acc"][cls]
        class_mean = np.mean(class_accs)
        class_std = np.std(class_accs)
        print(f"Class {cls} Accuracy: {class_mean:.4f} ± {class_std:.4f}")

    # Overall Parcel Accuracy
    overall_mean_parcel = np.mean(fold_metrics["overall_acc_parcel"])
    overall_std_parcel = np.std(fold_metrics["overall_acc_parcel"])
    print(f"Overall Parcel Accuracy: {overall_mean_parcel:.4f} ± {overall_std_parcel:.4f}")

    # Per-Class Parcel Accuracy
    for cls in sorted(fold_metrics["class_acc_parcel"].keys()):
        class_accs_parcel = fold_metrics["class_acc_parcel"][cls]
        class_mean_parcel = np.mean(class_accs_parcel)
        class_std_parcel = np.std(class_accs_parcel)
        print(f"Class {cls} Parcel Accuracy: {class_mean_parcel:.4f} ± {class_std_parcel:.4f}")

    # Compute average confusion matrix
    mean_cm = confusion_matrices / k
    mean_cm2 = confusion_matrices_parcel / k
    plot_confusion_matrix(mean_cm.astype(int), class_names=["Class 0", "Class 1", "Class 2"], save_path=cmPath + ".jpg")
    plot_confusion_matrix(mean_cm2.astype(int), class_names=["Class 0", "Class 1", "Class 2"], save_path=cmPath + "_parcel.jpg")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Locations')
    parser.add_argument('--labels_file', metavar='path', required=True)
    parser.add_argument('--img_directory', metavar='path', required=True)
    parser.add_argument('--default_parcel', metavar='int', required=True)
    parser.add_argument('--cmPath', metavar='path', required=True)
    parser.add_argument("--fineTuning", type=int, required=True, help="setup of finetuning")
    parser.add_argument("--kFold", type=int, required=True, help="number of folds for k-fold cross validation")
    parser.add_argument("--loRA", action="store_true", help="performs LoRA")
    parser.add_argument("--noLoRA", action="store_false", dest="loRA", help="does not perform LoRA")
    args = parser.parse_args()
    main(labels_file=args.labels_file, img_directory=args.img_directory, default_parcel=args.default_parcel,
         cmPath=args.cmPath, fineTuning=args.fineTuning, k=args.kFold, lora=args.loRA)