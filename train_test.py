import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np

img_width = 840


# Custom Dataset Definition
class CustomDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "labels"))))

    def __getitem__(self, idx):
        # Load images and annotations
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annotation_path = os.path.join(self.root, "labels", self.annotations[idx])

        img = Image.open(img_path).convert("RGB")
        boxes_and_labels = np.loadtxt(annotation_path, delimiter=" ").reshape(
            -1, 5
        )  # [class_id, x_center, y_center, width, height]

        # Extract labels and bounding boxes
        labels = torch.as_tensor(boxes_and_labels[:, 0], dtype=torch.int64)  # labels
        boxes = boxes_and_labels[
            :, 1:
        ]  # bounding boxes in [x_center, y_center, width, height] format

        boxes[:, 0] *= img_width  # x_center
        boxes[:, 1] *= img_width  # y_center
        boxes[:, 2] *= img_width  # width
        boxes[:, 3] *= img_width  # height

        # Convert bounding boxes to [x_min, y_min, x_max, y_max] format
        x_center = boxes[:, 0]
        y_center = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)

        # Filter out invalid boxes
        valid_boxes = (width > 0) & (height > 0)
        boxes = boxes[valid_boxes]
        labels = labels[valid_boxes]

        # Convert to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        image_id = torch.tensor([idx])

        target = {"boxes": boxes, "labels": labels, "image_id": image_id}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


# Define transformations
transform = T.Compose(
    [
        T.ToTensor(),
        T.Resize((img_width, img_width)),
    ]
)

# Load custom dataset
dataset = CustomDataset(root="cm_datasets/train", transforms=transform)
train_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=lambda x: tuple(zip(*x)),
)

# Load a pre-trained model
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

# Replace the classifier with a new one (if your dataset has different classes)
num_classes = 13  # 12 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = (
    torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
)

# Fine-tune the model
model.train()

# Move model to GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Define optimizer and learning rate scheduler
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, targets in train_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    # Step the learning rate scheduler
    lr_scheduler.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    torch.save(model.state_dict(), f"fasterrcnn_custom_{epoch+1}.pth")


print("Training complete.")

# Save the fine-tuned model
