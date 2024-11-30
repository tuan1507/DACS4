import torch
from PIL import Image
from model import ColorCNN
from data import get_dataloaders
from torchvision import transforms

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

def predict_color(image_path, model, transform, classes):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return classes[predicted.item()]

def load_model_and_predict(image_path):
    _, _, classes = get_dataloaders('./Data/train/',
                                    './Data/val/')

    model = ColorCNN(num_classes=len(classes))
    model.load_state_dict(torch.load("color_cnn.pth"))


    test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    predicted_color = predict_color(image_path, model, test_transforms, classes)
    print(f"Predicted color: {predicted_color}")

if __name__ == "__main__":
    image_path = "./Data/test/red.jpg"
    load_model_and_predict(image_path)
