
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from PIL import Image

# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Lớp biểu cảm
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_classes = len(emotion_labels)

# Tiền xử lý dữ liệu
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Hàm huấn luyện
def train_model():
    # Đường dẫn dữ liệu
    train_dir = r"D:\nhandienResnet\fer2013\train"
    test_dir = r"D:\nhandienResnet\fer2013\test"

    train_dataset = ImageFolder(root=train_dir, transform=transform_train)
    test_dataset = ImageFolder(root=test_dir, transform=transform_test)

    # Chỉnh num_workers về 0 để tránh lỗi trên Windows
    num_workers = 0 if os.name == "nt" else 4

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

    # Load ResNet50
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # Hàm mất mát & tối ưu hóa
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Huấn luyện
    num_epochs = 10
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch+1}/{num_epochs} - Training...")

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Đánh giá trên tập test
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = test_loss / len(test_loader)
        test_losses.append(test_loss)
        accuracy = 100 * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Vẽ biểu đồ loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.show()

    # Đánh giá mô hình
    evaluate_model(model, test_loader)

    torch.save(model.state_dict(), "resnet50_emotion.pth")
    print("Model saved successfully!")

# Hàm đánh giá mô hình
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Ma trận nhầm lẫn
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=emotion_labels, yticklabels=emotion_labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Báo cáo phân loại
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=emotion_labels))

# Hàm dự đoán trên ảnh mới
def predict_emotion(image_path, model):
    model.eval()
    image = Image.open(image_path)  # Sử dụng PIL để mở ảnh
    image = transform_test(image).unsqueeze(0).to(device)  # Chuyển ảnh về tensor và chuẩn hóa

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        emotion = emotion_labels[predicted.item()]

    return emotion

# Chạy chương trình
if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)  # Chống lỗi multiprocessing trên Windows
    
    # Huấn luyện mô hình
    train_model()

    # Dự đoán ảnh mới
    image_path = r"D:\nhandienResnet\fer2013\test\sad\PublicTest_99688200.jpg"  # Đường dẫn ảnh mới
    
    # Tải mô hình đã huấn luyện
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("resnet50_emotion.pth"))  # Nếu đã lưu mô hình
    model.to(device)

    predicted_emotion = predict_emotion(image_path, model)
    print(f"Predicted Emotion: {predicted_emotion}")
