# Face Mask Detection Model

demo: https://face-mask-detection-using-cnn-1.onrender.com

A deep learning model for detecting whether a person is wearing a face mask or not, built with PyTorch. The system can:

Classify single images

Process batches of images from a directory

Perform real-time face mask detection using webcam

Detect multiple faces in an image and classify each one

The model achieves high accuracy in distinguishing between "with_mask" and "without_mask" classes.

## Model Architecture

The model is a custom convolutional neural network (CNN) with the following architecture:

### Feature Extractor

- **ConvBlock 1**: 3→32 channels, MaxPool (128×128→64×64)
- **ConvBlock 2**: 32→64 channels, MaxPool (64×64→32×32)
- **ConvBlock 3**: 64→128 channels, MaxPool (32×32→16×16)
- **ConvBlock 4**: 128→256 channels, MaxPool (16×16→8×8)
- **ConvBlock 5**: 256→512 channels, MaxPool (8×8→4×4)

### Classifier

- **Fully Connected 1**: 8192→512 units with Dropout (0.5)
- **Fully Connected 2**: 512→2 units (with_mask, without_mask)


https://github.com/user-attachments/assets/317d611d-bf39-4a6a-b3d0-f405acc5a3b2


- 

## Performance

- **Test Accuracy**: 98.35%
- **Training Time**: ~25 epochs
- **Input Size**: 128×128 RGB images
  <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/c5123fdf-4512-400d-b9fe-5f8e8e5925ae" />


## Installation

1. Clone the repository:

```bash
git clone https://github.com/olyadboka/face-mask-detection.git
cd face-mask-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. to Execute

   ```bash
   python test.py --camera
   or
   python test.py --image image1.jpeg 
