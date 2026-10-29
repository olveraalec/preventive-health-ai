import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [react()],
    server: {
      host: '127.0.0.1',
      proxy: { '/api': { target: env.API_PROXY_TARGET || 'http://127.0.0.1:8003', changeOrigin: false } },
    },
    preview: { host: '127.0.0.1' },
  }
})
