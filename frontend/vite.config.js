import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: proxy /api sang backend Flask (:5000) để tránh vấn đề CORS/URL cứng.
// Prod: build tĩnh, Nginx (xem Dockerfile) tự proxy /api sang service backend.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_BASE_URL || "http://localhost:5000",
        changeOrigin: true,
      },
    },
  },
});
