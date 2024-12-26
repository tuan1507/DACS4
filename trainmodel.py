import torch
import torch.optim as optim
import torch.nn as nn
from model import ColorCNN
from data import get_dataloaders

train_path = './Data/train/'
test_path = './Data/val/'
num_epochs = 30

criterion = nn.CrossEntropyLoss()

def train_model():
    train_loader, _, classes = get_dataloaders(train_path, test_path)
    num_classes = len(classes)

    model = ColorCNN(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "color_cnn.pth")
    print("Mô hình đã được lưu!")

if __name__ == "__main__":
    train_model()
