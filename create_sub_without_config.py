from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import torch
import torchvision.transforms
from models.dinov2 import DinoV2Finetune

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    def __init__(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        # filter out non-image files
        self.images_list = [image for image in images_list if image.endswith(".jpg")]

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        image = Image.open(image_path)
        image = self.test_transform(image)
        return image, os.path.splitext(image_name)[0]

    def __len__(self):
        return len(self.images_list)


def create_submission():
    test_loader = DataLoader(
        TestDataset(
            "dataset/test", torchvision.transforms.Compose(
            transforms=[
                torchvision.transforms.Resize(size=[224, 224]),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ),
        batch_size=128,
        shuffle=False,
        num_workers=8,
    )
    # Load model and checkpoint
    model = DinoV2Finetune(num_classes=37, frozen=True, unfreeze_last_layer=True).to(device)
    checkpoint = torch.load("checkpoints/DINOV2_simple_prompts_with_production.pt")
    print(f"Loading model from checkpoint: checkpoints/DINOV2_simple_prompts_with_production.pt")
    model.load_state_dict(checkpoint)
    class_names = sorted(os.listdir("dataset/train/simple_prompts"))

    # Create submission.csv
    submission = pd.DataFrame(columns=["id", "label"])

    for i, batch in enumerate(test_loader):
        images, image_names = batch
        images = images.to(device)
        preds = model(images)
        preds = preds.argmax(1)
        preds = [class_names[pred] for pred in preds.cpu().numpy()]
        submission = pd.concat(
            [
                submission,
                pd.DataFrame({"id": image_names, "label": preds}),
            ]
        )
    submission.to_csv(f"submission.csv", index=False)


if __name__ == "__main__":
    create_submission()