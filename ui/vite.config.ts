import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// In dev, /api is proxied to the equate service (`python -m equate.service serve`).
// Point elsewhere with EQUATE_API_TARGET, or skip the proxy in a build with VITE_EQUATE_API.
const target = process.env.EQUATE_API_TARGET ?? 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: { '/api': { target, changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, '') } },
  },
  test: { include: ['src/**/*.test.ts'] },
});
