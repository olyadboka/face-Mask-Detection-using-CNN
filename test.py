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
import cv2

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


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
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
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
            ConvBlock(128, 256)
        )

        self._initialize_weights()
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 128, 128)
            dummy_output = self.features(dummy_input)
            self.feature_size = dummy_output.view(1, -1).size(1)

        self.classifier = ClassifierHead(self.feature_size, num_classes)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_model(model_path, class_names_path):
    """Load the trained model and class names"""

    with open(class_names_path, 'r') as f:
        class_names = json.load(f)

    model = FaceMaskCNN(num_classes=len(class_names)).to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return model, class_names
    else:
        raise FileNotFoundError(f"Model file {model_path} not found")


def get_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def detect_faces_and_predict_mask(model, image_path, transform, scale_factor=1.1, min_neighbors=5):
    """
    Detect faces in an image and predict if they're wearing masks
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, []

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)

    results = []
    for (x, y, w, h) in faces:

        face_region = image_rgb[y:y+h, x:x+w]

        face_pil = Image.fromarray(face_region)

        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(face_tensor)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        class_names = ['with_mask', 'without_mask']
        predicted_class = class_names[predicted.item()]
        confidence = probs[0][predicted.item()].item()

        color = (0, 255, 0) if predicted_class == 'with_mask' else (255, 0, 0)

        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

        label = f"{predicted_class}: {confidence:.2f}"
        cv2.putText(image, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        results.append({
            'bbox': (x, y, w, h),
            'prediction': predicted_class,
            'confidence': confidence
        })

    # Convert back to BGR for display
    image_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_display, results

# Real-time mask detection using webcam


def real_time_mask_detection(model, transform):
    """
    Real-time mask detection using webcam
    """
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Press 'q' to quit")

    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        # Process each face
        for (x, y, w, h) in faces:
            # Extract face region
            face_region = rgb_frame[y:y+h, x:x+w]

            # Convert to PIL Image for transformation
            face_pil = Image.fromarray(face_region)

            # Apply transformations and predict
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(face_tensor)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

            # Get prediction details
            class_names = ['with_mask', 'without_mask']
            predicted_class = class_names[predicted.item()]
            confidence = probs[0][predicted.item()].item()

            # Determine color based on prediction
            color = (0, 255, 0) if predicted_class == 'with_mask' else (
                0, 0, 255)

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Add label
            label = f"{predicted_class}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display the frame
        cv2.imshow('Face Mask Detection', frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

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

# Function to test the model on custom images


def test_custom_images(model, image_paths, transform, class_names):
    """
    Test the model on custom images
    """
    fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))
    if len(image_paths) == 1:
        axes = [axes]

    for i, image_path in enumerate(image_paths):
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        # Get prediction details
        predicted_class = class_names[predicted.item()]
        confidence = probs[0][predicted.item()].item()

        # Display image
        axes[i].imshow(image)
        axes[i].set_title(
            f"Prediction: {predicted_class}\nConfidence: {confidence:.4f}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

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

# Main function


def main():
    parser = argparse.ArgumentParser(
        description='Test Face Mask Detection Model')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--directory', type=str,
                        help='Path to directory containing images')
    parser.add_argument('--camera', action='store_true',
                        help='Use camera for real-time detection')
    parser.add_argument('--detect-faces', type=str,
                        help='Detect faces in image and predict masks')
    parser.add_argument('--model', type=str, default='complete_face_mask_model.pth',
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
        real_time_mask_detection(model, transform)

    # Detect faces in image and predict masks
    elif args.detect_faces:
        if os.path.exists(args.detect_faces):
            result_image, predictions = detect_faces_and_predict_mask(
                model, args.detect_faces, transform)

            if result_image is not None:
                print(
                    f"\nFound {len(predictions)} faces in {args.detect_faces}:")
                for i, pred in enumerate(predictions):
                    print(
                        f"Face {i+1}: {pred['prediction']} (confidence: {pred['confidence']:.2f})")

                plt.figure(figsize=(10, 10))
                plt.imshow(result_image)
                plt.axis('off')
                plt.title('Face Mask Detection Results')
                plt.show()
        else:
            print(f"Image file {args.detect_faces} not found!")

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
        print("Please specify one of the following arguments:")
        print("  --image: Path to single image file")
        print("  --directory: Path to directory containing images")
        print("  --camera: Use camera for real-time detection")
        print("  --detect-faces: Detect faces in image and predict masks")
        print("\nExamples:")
        print("  python test.py --image test_image.jpg")
        print("  python test.py --directory test_images/ --display")
        print("  python test.py --camera")
        print("  python test.py --detect-faces group_photo.jpg")


if __name__ == "__main__":
    main()
