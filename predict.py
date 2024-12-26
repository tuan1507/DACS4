import torch
from model import ColorCNN
from PIL import Image
import numpy as np
from data import *
import cv2


mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]


def transform(image):
    image = image.resize((64, 64))
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - mean) / std
    image = torch.tensor(image.transpose((2, 0, 1)))
    image = image.float()
    return image

# Hàm dự đoán cho một ảnh mới
def predict_image(image_path, model, transform, classes):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Thêm batch dimension (C, H, W) -> (1, C, H, W)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
        predicted_class_name = classes[predicted_class]
        return predicted_class_name


model = ColorCNN(num_classes=10)
model.load_state_dict(torch.load("color_cnn.pth"))

train_path = './Data/train/'
test_path = './Data/val/'
_, _, classes = get_dataloaders(train_path, test_path)


# image_path = './Data/test/hoa-hong-07.jpg'
# image_path = './Data/test/quabo.jpg'
# image_path = './Data/test/quadautay.jpg'
image_path = './Data/test/nhieutraicay.jpg'
predicted_class_name = predict_image(image_path, model, transform, classes)


image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
window_name = f"Predicted: {predicted_class_name}"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)
cv2.imshow(window_name, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
