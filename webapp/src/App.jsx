import React, { useMemo, useRef, useState } from "react";
import { predictImage } from "./api";

function UploadTab() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!image) return;
    setLoading(true);
    setResult(null);
    try {
      const data = await predictImage(image);
      setResult(data);
    } catch (err) {
      setResult({ error: err.message, details: err.details });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card">
      <h2>Upload Image</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setImage(e.target.files[0] || null)}
        />
        <button disabled={!image || loading} type="submit">
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>
      {image && (
        <div className="preview">
          <img src={URL.createObjectURL(image)} alt="preview" />
        </div>
      )}
      {result && (
        <div className="result">
          {result.error ? (
            <p className="error">{result.error}</p>
          ) : (
            <>
              <p>
                <strong>Prediction:</strong> {result.prediction}
              </p>
              <p>
                <strong>Confidence:</strong>{" "}
                {(result.confidence * 100).toFixed(1)}%
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function CameraTab() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [result, setResult] = useState(null);
  const [streaming, setStreaming] = useState(false);
  const [busy, setBusy] = useState(false);

  async function start() {
    if (streaming) return;
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current.srcObject = stream;
    await videoRef.current.play();
    setStreaming(true);
  }

  function stop() {
    const stream = videoRef.current?.srcObject;
    stream?.getTracks().forEach((t) => t.stop());
    setStreaming(false);
  }

  async function snapAndPredict() {
    if (!streaming || busy) return;
    setBusy(true);
    setResult(null);
    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0);
      const blob = await new Promise((r) =>
        canvas.toBlob(r, "image/jpeg", 0.9)
      );
      const data = await predictImage(blob);
      setResult(data);
    } catch (err) {
      setResult({ error: err.message, details: err.details });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h2>Webcam</h2>
      <div className="camera">
        <video ref={videoRef} autoPlay playsInline muted />
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>
      <div className="row">
        <button onClick={start} disabled={streaming}>
          Start
        </button>
        <button onClick={stop} disabled={!streaming}>
          Stop
        </button>
        <button onClick={snapAndPredict} disabled={!streaming || busy}>
          {busy ? "Predicting..." : "Snap & Predict"}
        </button>
      </div>
      {result && (
        <div className="result">
          {result.error ? (
            <p className="error">{result.error}</p>
          ) : (
            <>
              <p>
                <strong>Prediction:</strong> {result.prediction}
              </p>
              <p>
                <strong>Confidence:</strong>{" "}
                {(result.confidence * 100).toFixed(1)}%
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("upload");
  return (
    <div className="container">
      <header>
        <h1>Face Mask Detection</h1>
        <nav>
          <button
            className={tab === "upload" ? "active" : ""}
            onClick={() => setTab("upload")}
          >
            Upload
          </button>
          <button
            className={tab === "camera" ? "active" : ""}
            onClick={() => setTab("camera")}
          >
            Webcam
          </button>
        </nav>
      </header>
      {tab === "upload" ? <UploadTab /> : <CameraTab />}
      <footer>
        <p>Model served by ONNX Runtime on Node.js</p>
        <p>Copyright &copy; 2025 reserved to Olyad Boka</p>
      </footer>
    </div>
  );
}
