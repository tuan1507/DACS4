import torch
import torch.optim as optim
import torch.nn as nn
from model import ColorCNN
from data import get_dataloaders  # Nhập hàm tải dữ liệu

# Các tham số và đường dẫn
train_path = './Data/train/'
test_path = './Data/val/'
num_epochs = 10

# Hàm mất mát và tối ưu hóa
criterion = nn.CrossEntropyLoss()

def train_model():
    # Tải dữ liệu và lấy nhãn lớp
    train_loader, _, classes = get_dataloaders(train_path, test_path)
    num_classes = len(classes)

    # Khởi tạo mô hình
    model = ColorCNN(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Vòng lặp huấn luyện
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images, labels

            # Tiến hành truyền qua mô hình
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Quá trình lan truyền ngược
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Lưu mô hình sau khi huấn luyện
    torch.save(model.state_dict(), "color_cnn.pth")
    print("Mô hình đã được lưu!")

# Chạy hàm huấn luyện
if __name__ == "__main__":
    train_model()
