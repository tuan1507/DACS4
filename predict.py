import torch
from model import ColorCNN
from PIL import Image
import numpy as np
from data import *

# Các tham số học máy
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# Hàm biến đổi dữ liệu đầu vào (giống như trong quá trình huấn luyện)
def transform(image):
    image = image.resize((64, 64))  # Đảm bảo ảnh có kích thước 64x64
    image = np.array(image).astype(np.float32) / 255.0  # Chuyển ảnh thành mảng numpy và đổi kiểu thành float32
    image = (image - mean) / std  # Chuẩn hóa theo mean và std
    image = torch.tensor(image.transpose((2, 0, 1)))  # Chuyển từ (H, W, C) sang (C, H, W)
    image = image.float()  # Chuyển sang kiểu float32
    return image

# Hàm dự đoán cho một ảnh mới
def predict_image(image_path, model, transform, classes):
    image = Image.open(image_path).convert('RGB')  # Đọc ảnh
    image = transform(image).unsqueeze(0)  # Thêm batch dimension (C, H, W) -> (1, C, H, W)

    model.eval()  # Chuyển mô hình sang chế độ đánh giá (evaluation)
    with torch.no_grad():  # Tắt tính toán gradient
        outputs = model(image)  # Dự đoán đầu ra
        _, predicted = torch.max(outputs, 1)  # Lấy lớp có xác suất cao nhất
        predicted_class = predicted.item()  # Lớp dự đoán (số)
        predicted_class_name = classes[predicted_class]  # Lấy tên lớp từ danh sách classes
        return predicted_class_name  # Trả về tên lớp

# Đọc trọng số mô hình đã huấn luyện và lấy classes
model = ColorCNN(num_classes=10)  # Đảm bảo số lớp giống như lúc huấn luyện
model.load_state_dict(torch.load("color_cnn.pth"))  # Tải trọng số từ file

# Tải classes từ quá trình huấn luyện (sử dụng lại phần get_dataloaders)
train_path = './Data/train/'
test_path = './Data/val/'
_, _, classes = get_dataloaders(train_path, test_path)

# Dự đoán cho một ảnh mới
image_path = './Data/test/images.png'  # Đường dẫn đến ảnh cần dự đoán
predicted_class_name = predict_image(image_path, model, transform, classes)

# In ra tên lớp dự đoán
print(f"Dự đoán màu: {predicted_class_name}")
