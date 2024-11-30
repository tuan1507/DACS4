import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Các tham số học máy
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
batch_size = 32

# Định nghĩa các phép biến đổi cho dữ liệu huấn luyện và kiểm tra
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),  # Thay đổi kích thước ảnh về 64x64
    transforms.ToTensor(),
    transforms.Normalize(mean, std)  # Chuẩn hóa ảnh
])

test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),  # Thay đổi kích thước ảnh về 64x64
    transforms.ToTensor(),
    transforms.Normalize(mean, std)  # Chuẩn hóa ảnh
])

# Hàm để tải dữ liệu và trả về DataLoader
def get_dataloaders(train_path, test_path):
    # Tải dữ liệu
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transforms)

    # Tạo DataLoader cho dữ liệu huấn luyện và kiểm tra
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Lấy nhãn của lớp
    classes = train_dataset.classes
    return train_loader, test_loader, classes
