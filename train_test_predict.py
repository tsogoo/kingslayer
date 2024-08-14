import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import os
import cv2
from PIL import Image
import numpy as np

img_width = 840


# Define transformations
transform = T.Compose(
    [
        T.ToTensor(),
        T.Resize((img_width, img_width)),
    ]
)

# Load custom dataset

# Load a pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Replace the classifier with a new one (if your dataset has different classes)
num_classes = 13  # 12 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = (
    torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
)


# Move model to GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# load fasterrcnn_custom.pth model
model.load_state_dict(torch.load("fasterrcnn_custom_5.pth"))
model.eval()
# Define optimizer and learning rate scheduler
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# Function to perform inference on a single image
def predict(image, model, device, threshold=0.6):
    model.eval()
    with torch.no_grad():
        image = [F.to_tensor(image).to(device)]
        output = model(image)
        output = [{k: v.to(device) for k, v in t.items()} for t in output]

    # Filter out low-confidence predictions
    pred_classes = [int(i) for i in output[0]["labels"].cpu().numpy()]
    pred_scores = output[0]["scores"].detach().cpu().numpy()
    pred_boxes = [
        [(i[0], i[1]), (i[2], i[3])] for i in output[0]["boxes"].detach().cpu().numpy()
    ]

    pred_boxes = [
        pred_boxes[i] for i in range(len(pred_boxes)) if pred_scores[i] > threshold
    ]
    pred_classes = [
        pred_classes[i] for i in range(len(pred_classes)) if pred_scores[i] > threshold
    ]

    return pred_boxes, pred_classes, pred_scores


# Example of loading and making predictions on a new image
# image = Image.open("cm_datasets/train/images/0017.png").convert("RGB")
image = Image.open("frameg.jpg").convert("RGB")

boxes, classes, scores = predict(image, model, device)

print("====", boxes)
print("Classes===============", classes)
print("Scores===============", scores)


# function to draw bounding boxes on the image and save it using PIL
def draw_boxes(image, boxes, classes, scores):

    image = np.array(image)
    for box, cls, score in zip(boxes, classes, scores):
        box = [(int(i[0]), int(i[1])) for i in box]
        image = cv2.rectangle(image, tuple(box[0]), tuple(box[1]), (255, 0, 0), 1)
        label = f"{cls}: {score:.2f}"
        image = cv2.putText(
            image, label, tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )

    image = Image.fromarray(image)
    image.save("output_image.jpg")


draw_boxes(image, boxes, classes, scores)
