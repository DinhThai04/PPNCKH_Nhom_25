import torch
from torchvision.models import resnet50
import torch.nn as nn

# Danh sách các nhãn cảm xúc (cùng số lớp với mô hình đã train)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_classes = len(emotion_labels)

# Khởi tạo lại mô hình ResNet50 với kiến trúc đã train
model = resnet50(weights=None)  # Không cần tải trọng số ban đầu
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  # Chỉnh sửa lớp đầu ra

# Lưu mô hình nếu đã có biến `model`
torch.save(model.state_dict(), "resnet50_emotion.pth")
print("✅ Mô hình đã được lưu thành công dưới tên resnet50_emotion.pth!")
