import torch
from model import ColorCNN
from data import get_dataloaders

train_path = './Data/train/'
test_path = './Data/val/'

def test_model():
    _, test_loader, classes = get_dataloaders(train_path, test_path)
    num_classes = len(classes)

    # Khởi tạo mô hình và tải trọng số
    model = ColorCNN(num_classes=num_classes)
    model.load_state_dict(torch.load("color_cnn.pth"))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Độ chính xác trên tập kiểm tra: {accuracy:.2f}%")

if __name__ == "__main__":
    test_model()
