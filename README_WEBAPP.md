# Face Mask Detection - Web App

This adds a Node.js (Express) backend serving your ONNX model and a React frontend to upload images or use the webcam for predictions.

## Prerequisites

- Node.js 18+
- The model file `face_mask_model.onnx` exists in the project root (same level as `server/` and `webapp/`).

## Backend (server)

```
cd server
npm install
npm run start
```

- The server by default listens on `http://localhost:5001`.
- It reads the ONNX model from `../face_mask_model.onnx` (project root). Override via env var `MODEL_FILENAME`.

## Frontend (webapp)

```
cd webapp
npm install
npm run dev
```

- Open `http://localhost:5173`.
- Vite dev server proxies `/api/*` to `http://localhost:5001`.

## Predict API

- Endpoint: `POST /api/predict`
- Form field: `image` (file)
- Response:

```json
{
  "prediction": "with_mask",
  "confidence": 0.93,
  "probabilities": [0.93, 0.07]
}
```

## Notes

- Preprocessing mirrors Python: resize to 128x128, CHW float32, ImageNet mean/std normalization.
- The class order used: `["with_mask", "without_mask"]`.

---

## Use the .pth model (PyTorch mode)

If you prefer to use your original `.pth` checkpoint directly, run the Python microservice and set the Node server to proxy:

1. Start Python FastAPI service

```
cd pyserver
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Checkpoint discovery order (auto): best_model.pth, complete_face_mask_model.pth, face_mask_model.pth
# Locations checked: project root, server/, pyserver/
# To force a specific file, set:
#   $env:PYTORCH_MODEL="best_model.pth"
uvicorn main:app --host 127.0.0.1 --port 8000
```

2. Start Node server in PyTorch mode

```
cd server
$env:USE_PYTORCH="1"; $env:PY_URL="http://127.0.0.1:8000"; npm run start
```

- The Node server will forward `/api/predict` to the Python service.
- To customize paths:
  - Python: set `PYTORCH_MODEL` (default auto-discovers `best_model.pth` first) and `CLASSES_JSON` (default `class_names.json`).
  - Node: `PY_URL` if the Python service URL differs.
