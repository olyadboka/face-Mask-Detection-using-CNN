# Face Mask Detection Model

A deep learning model for detecting whether a person is wearing a face mask or not, built with PyTorch.

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
