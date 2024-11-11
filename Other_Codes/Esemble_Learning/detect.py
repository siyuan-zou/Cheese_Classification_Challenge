import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from models.dinov2 import DinoV2Finetune
import json

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                for fname in os.listdir(label_path):
                    if fname.endswith('.jpg'):  # Add other extensions if needed
                        self.image_paths.append(os.path.join(label_path, fname))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, image_path

def main(val_dir, model_path):
    # Define the transformation for the validation images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the validation dataset
    val_dataset = CustomDataset(root_dir=val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained model
    model = DinoV2Finetune(num_classes=37, frozen=True, unfreeze_last_layer=True).to(device)
    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint)
    with open("list_of_cheese.txt", "r") as f:
        class_names = sorted(f.read().splitlines())

    # Move model to GPU if available
    model.to(device)

    # Dictionary to store misclassified images
    misclassified = {}

    # Perform classification
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images, image_names, image_paths = batch
            # print(image_paths)
            # print(image_names)
            images = images.to(device)
            preds = model(images)
            preds = preds.argmax(1)

            for j in range(len(image_names)):
                true_label = image_names[j]
                predicted_label = class_names[preds[j]]

                # Collect misclassified images
                if predicted_label != true_label:
                    # print(f"True label: {true_label}, Predicted label: {predicted_label}")

                    img_path = image_paths[j]
                    misclassified[img_path] = (true_label, predicted_label)

    # Output the dictionary of misclassified images
    with open("detect_no_crop.json", "w") as f:
        json.dump(misclassified, f)

if __name__ == '__main__':
    val_dir = 'dataset/val'
    model_path = 'checkpoints/Best_Base_Hyper.pt'
    main(val_dir, model_path)
