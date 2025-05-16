import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Định nghĩa các nhãn cảm xúc
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_classes = len(emotion_labels)

# Đảm bảo mô hình được chuyển sang device phù hợp (CPU hoặc GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Định nghĩa các phép biến đổi cho ảnh (resize, normalize)
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # Kích thước đầu vào cho ResNet50
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Hàm dự đoán cảm xúc từ ảnh
def predict_emotion(image_path, model):
    model.eval()

    # Đọc ảnh từ đường dẫn và chuyển ảnh về RGB nếu cần
    image = Image.open(image_path).convert("RGB")

    # Tiền xử lý ảnh (biến đổi thành tensor và chuẩn hóa)
    image = transform_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Dự đoán cảm xúc
        output = model(image)
        _, predicted = torch.max(output, 1)
        
        # Lấy nhãn cảm xúc tương ứng
        emotion = emotion_labels[predicted.item()]

    return emotion

# Tải mô hình đã huấn luyện (đã lưu)
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  # Điều chỉnh lớp fully connected cho số lớp cảm xúc
model.load_state_dict(torch.load("resnet50_emotion.pth"))  # Load mô hình đã lưu
model.to(device)

# Đường dẫn đến ảnh muốn dự đoán
image_path = r"D:\nhandienResnet\fer2013\test\sad\PublicTest_99688200.jpg"

# Dự đoán cảm xúc của ảnh
predicted_emotion = predict_emotion(image_path, model)
print(f"Predicted Emotion: {predicted_emotion}")
