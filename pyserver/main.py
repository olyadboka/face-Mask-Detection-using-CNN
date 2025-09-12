import io
import os
import json
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

# ---------------- Model definition (copied from test.py) ---------------- #


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
            ConvBlock(128, 256),
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

# ---------------- FastAPI app ---------------- #


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defaults to best_model.pth, with fallbacks
DEFAULT_CHECKPOINTS = [
    'best_model.pth',
    'complete_face_mask_model.pth',
    'face_mask_model.pth',
]

MODEL_ENV = os.environ.get('PYTORCH_MODEL')
CLASSES_JSON = os.environ.get('CLASSES_JSON', 'class_names.json')

with open(os.path.join(os.getcwd(), CLASSES_JSON), 'r') as f:
    class_names: List[str] = json.load(f)

model = FaceMaskCNN(num_classes=len(class_names)).to(device)

# Resolve checkpoint path
search_paths = []
if MODEL_ENV:
    search_paths.append(MODEL_ENV)
else:
    search_paths.extend(DEFAULT_CHECKPOINTS)
# Also check common subfolders
candidates = []
for name in search_paths:
    candidates.append(os.path.join(os.getcwd(), name))
    candidates.append(os.path.join(os.getcwd(), 'server', name))
    candidates.append(os.path.join(os.getcwd(), 'pyserver', name))

ckpt_path = None
for p in candidates:
    if os.path.exists(p):
        ckpt_path = p
        break

if ckpt_path is None:
    raise FileNotFoundError(
        "No checkpoint found. Tried: " + ", ".join(candidates)
    )

print(f"[pyserver] Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device)
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@app.get('/health')
def health():
    return {"ok": True, "device": str(device), "checkpoint": ckpt_path}


@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
        predicted_class = class_names[idx.item()]
        return {
            "prediction": predicted_class,
            "confidence": float(conf.item()),
            "probabilities": [float(p) for p in probs.tolist()],
        }
    except Exception as e:
        return {"error": str(e)}
