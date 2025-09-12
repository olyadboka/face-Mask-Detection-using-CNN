import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import multer from "multer";
import sharp from "sharp";
import fs from "fs";
import path from "path";
import os from "os";
import ort from "onnxruntime-node";
import FormData from "form-data";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({ dest: path.join(os.tmpdir(), "uploads") });

const USE_PYTORCH = process.env.USE_PYTORCH === "1";
const PY_URL = process.env.PY_URL || "http://127.0.0.1:8000";

const MODEL_FILENAME = process.env.MODEL_FILENAME || "face_mask_model.onnx";
const MODEL_PATH = path.resolve(process.cwd(), MODEL_FILENAME);

let session = null;
const classNames = ["with_mask", "without_mask"];

async function loadModel() {
  if (USE_PYTORCH) {
    console.log("Using external PyTorch microservice for predictions.");
    return;
  }
  try {
    console.log(`Loading ONNX model from ${MODEL_PATH}`);
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["cpu"],
    });
    console.log("ONNX model loaded");
  } catch (err) {
    console.error("Failed to load ONNX model:", err.message);
    process.exit(1);
  }
}

function normalizeImageTensor(floatArray) {
  // Normalize using ImageNet mean/std to match test.py
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < 128 * 128; i++) {
      const idx = c * 128 * 128 + i;
      floatArray[idx] = (floatArray[idx] - mean[c]) / std[c];
    }
  }
  return floatArray;
}

async function preprocessToCHWFloat(imagePath) {
  // Resize to 128x128, convert to tensor CHW float32 normalized [0,1]
  const image = sharp(imagePath).resize(128, 128).toFormat("png");
  const { data, info } = await image
    .raw()
    .toBuffer({ resolveWithObject: true });
  if (info.channels !== 3) {
    // Convert to RGB
    const converted = await sharp(data, {
      raw: { width: info.width, height: info.height, channels: info.channels },
    })
      .removeAlpha()
      .toColourspace("rgb")
      .raw()
      .toBuffer({ resolveWithObject: true });
    return toCHWFloat(
      converted.data,
      converted.info.width,
      converted.info.height
    );
  }
  return toCHWFloat(data, info.width, info.height);
}

function toCHWFloat(rgbBuffer, width, height) {
  const size = width * height;
  const chw = new Float32Array(3 * size);
  for (let i = 0; i < size; i++) {
    const r = rgbBuffer[3 * i] / 255;
    const g = rgbBuffer[3 * i + 1] / 255;
    const b = rgbBuffer[3 * i + 2] / 255;
    chw[i] = r;
    chw[size + i] = g;
    chw[2 * size + i] = b;
  }
  return normalizeImageTensor(chw);
}

app.get("/api/health", async (_req, res) => {
  if (USE_PYTORCH) {
    try {
      const r = await fetch(`${PY_URL}/health`);
      const ct = r.headers.get("content-type") || "";
      const body = ct.includes("application/json")
        ? await r.json()
        : await r.text();
      return res.status(r.status).json({ ok: r.ok, pytorch: true, py: body });
    } catch (e) {
      return res
        .status(502)
        .json({ ok: false, pytorch: true, error: String(e) });
    }
  }
  res.json({ ok: true, pytorch: false });
});

app.post("/api/predict", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image uploaded" });
    }

    if (USE_PYTORCH) {
      try {
        const form = new FormData();
        form.append("image", fs.createReadStream(req.file.path), {
          filename: req.file.originalname || "upload.jpg",
          contentType: req.file.mimetype || "application/octet-stream",
        });
        const r = await fetch(`${PY_URL}/predict`, {
          method: "POST",
          body: form,
        });
        const ct = r.headers.get("content-type") || "";
        const body = ct.includes("application/json")
          ? await r.json()
          : await r.text();
        fs.unlink(req.file.path, () => {});
        if (!r.ok) {
          return res
            .status(r.status)
            .json({ error: "Upstream error", details: body });
        }
        if (typeof body === "string") {
          return res
            .status(502)
            .json({ error: "Non-JSON response from Python", details: body });
        }
        return res.json(body);
      } catch (e) {
        fs.unlink(req.file.path, () => {});
        return res
          .status(502)
          .json({ error: "Proxy to Python failed", details: String(e) });
      }
    }

    const chw = await preprocessToCHWFloat(req.file.path);
    const input = new ort.Tensor("float32", chw, [1, 3, 128, 128]);

    const inputName = session.inputNames[0];
    const output = await session.run({ [inputName]: input });
    const outputName = session.outputNames[0];
    const logits = output[outputName].data; // Float32Array of shape [1,2]

    const probs = softmax(Array.from(logits));
    const predictedIndex = probs.indexOf(Math.max(...probs));
    const prediction = classNames[predictedIndex] || `class_${predictedIndex}`;
    const confidence = probs[predictedIndex];

    fs.unlink(req.file.path, () => {});

    res.json({ prediction, confidence, probabilities: probs });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Inference failed", details: err.message });
  }
});

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

const PORT = process.env.PORT || 5001;

loadModel().then(() => {
  app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
    if (USE_PYTORCH) {
      console.log(`Proxying predictions to ${PY_URL}`);
    }
  });
});
