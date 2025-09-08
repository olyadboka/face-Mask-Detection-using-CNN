import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import os
import argparse
import matplotlib.pyplot as plt
import cv2  # OpenCV import

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the model architecture (same as training)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes=2):
        super(ClassifierHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class FaceMaskCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(FaceMaskCNN, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512)
        )
        self.classifier = ClassifierHead(512 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load the trained model


def load_model(model_path, class_names_path):
    """Load the trained model and class names"""
    # Load class names
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)

    # Load model
    model = FaceMaskCNN(num_classes=len(class_names)).to(device)

    # Load state dict
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return model, class_names
    else:
        raise FileNotFoundError(f"Model file {model_path} not found")

# Define transformations (same as training)


def get_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# Prediction function for single image


def predict_image(model, image_path, transform, class_names):
    """Predict mask status for a single image"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            probability = F.softmax(outputs, dim=1)

        # Get results
        predicted_class = class_names[predicted.item()]
        confidence = probability[0][predicted.item()].item()
        confidence_percent = confidence * 100

        return predicted_class, confidence_percent, original_image

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None, None

# Prediction function for OpenCV frame


def predict_frame(model, frame, transform, class_names, face_cascade):
    """Predict mask status for faces in a frame"""
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    results = []

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]

        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))

        # Apply transformations
        image_tensor = transform(pil_image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            probability = F.softmax(outputs, dim=1)

        # Get results
        predicted_class = class_names[predicted.item()]
        confidence = probability[0][predicted.item()].item()
        confidence_percent = confidence * 100

        # Draw rectangle and label
        color = (0, 255, 0) if predicted_class == "with_mask" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = f"{predicted_class}: {confidence_percent:.1f}%"
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        results.append({
            'prediction': predicted_class,
            'confidence': confidence_percent,
            'bbox': (x, y, w, h)
        })

    return frame, results

# Batch prediction function


def predict_batch(model, image_dir, transform, class_names):
    """Predict mask status for all images in a directory"""
    results = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} does not exist!")
        return results

    image_files = [f for f in os.listdir(image_dir)
                   if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"No images found in {image_dir}")
        return results

    print(f"Found {len(image_files)} images to process...")

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        prediction, confidence, image = predict_image(
            model, img_path, transform, class_names)

        if prediction is not None:
            results.append({
                'filename': img_file,
                'prediction': prediction,
                'confidence': confidence,
                'image': image
            })
            print(f"{img_file}: {prediction} ({confidence:.2f}%)")

    return results


def display_results(results, max_images=9):
    """Display prediction results in a grid"""
    if not results:
        print("No results to display")
        return

    num_images = min(len(results), max_images)
    rows = int(np.ceil(num_images / 3))

    fig, axes = plt.subplots(rows, 3, figsize=(15, 5*rows))
    axes = axes.flatten() if rows > 1 else [axes]

    for i, (result, ax) in enumerate(zip(results[:num_images], axes)):
        ax.imshow(result['image'])
        ax.set_title(
            f"{result['filename']}\n{result['prediction']} ({result['confidence']:.1f}%)")
        ax.axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def detect_from_camera(model, transform, class_names):
    """Real-time face mask detection from camera"""
    # Load face detection cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Press 'q' to quit the camera feed")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        frame = cv2.flip(frame, 1)

        processed_frame, results = predict_frame(
            model, frame, transform, class_names, face_cascade)

        cv2.imshow('Face Mask Detection by Olyad Boka', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function


def main():
    parser = argparse.ArgumentParser(
        description='Test Face Mask Detection Model')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--directory', type=str,
                        help='Path to directory containing images')
    parser.add_argument('--camera', action='store_true',
                        help='Use camera for real-time detection')
    parser.add_argument('--model', type=str, default='face_mask_model.pth',
                        help='Path to trained model file')
    parser.add_argument('--classes', type=str, default='class_names.json',
                        help='Path to class names JSON file')
    parser.add_argument('--display', action='store_true',
                        help='Display results with images')

    args = parser.parse_args()

    # Load model and class names
    try:
        model, class_names = load_model(args.model, args.classes)
        transform = get_transforms()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Process camera feed
    if args.camera:
        detect_from_camera(model, transform, class_names)

    # Process single image
    elif args.image:
        if os.path.exists(args.image):
            prediction, confidence, image = predict_image(
                model, args.image, transform, class_names)
            if prediction is not None:
                print(f"\nPrediction for {args.image}:")
                print(f"Class: {prediction}")
                print(f"Confidence: {confidence:.2f}%")

                if args.display and image:
                    plt.figure(figsize=(8, 6))
                    plt.imshow(image)
                    plt.title(f"{prediction} ({confidence:.1f}%)")
                    plt.axis('off')
                    plt.show()
        else:
            print(f"Image file {args.image} not found!")

    # Process directory of images
    elif args.directory:
        results = predict_batch(model, args.directory, transform, class_names)

        if results:
            print(f"\nProcessed {len(results)} images:")
            for result in results:
                print(
                    f"{result['filename']}: {result['prediction']} ({result['confidence']:.2f}%)")

            if args.display:
                display_results(results)
        else:
            print("No images were processed successfully")

    else:
        print("Please specify either --image, --directory, or --camera argument")
        print("Example: python test.py --image test_image.jpg")
        print("Example: python test.py --directory test_images/ --display")
        print("Example: python test.py --camera")


if __name__ == "__main__":
    main()
