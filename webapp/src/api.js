import axios from "axios";

const apiBase = import.meta.env.VITE_API_BASE || ""; // e.g., https://your-domain.com

export const api = axios.create({
  baseURL: apiBase,
  timeout: 20000,
  headers: { "X-Requested-With": "XMLHttpRequest" },
});

export async function predictImage(fileOrBlob) {
  const form = new FormData();
  form.append("image", fileOrBlob);
  try {
    const res = await api.post(
      "https://face-mask-detection-using-cnn.onrender.com/api/predict",
      form,
      {
        headers: { "Content-Type": "multipart/form-data" },
      }
    );
    return res.data;
  } catch (err) {
    if (err.response) {
      // Server responded with non-2xx
      const data = err.response.data;
      return Promise.reject({
        message:
          (data && (data.error || data.message)) ||
          `HTTP ${err.response.status}`,
        details: data && data.details,
        status: err.response.status,
      });
    }
    if (err.request) {
      return Promise.reject({
        message: "Network error or no response from server",
      });
    }
    return Promise.reject({ message: err.message || "Unknown error" });
  }
}
