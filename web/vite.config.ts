import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const backendUrl = process.env.VITE_BACKEND_PROXY_TARGET
  || process.env.VITE_BACKEND_URL
  || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    chunkSizeWarningLimit: 800,
    rollupOptions: {
      output: {
        manualChunks(id) {
          const normalized = id.replace(/\\/g, '/')
          if (!normalized.includes('/node_modules/')) return undefined
          if (/(\/node_modules\/(?:react|react-dom|scheduler)\/)/.test(normalized)) return 'vendor-react'
          if (normalized.includes('/node_modules/@ant-design/') || normalized.includes('/node_modules/antd/')) return 'vendor-antd'
          if (normalized.includes('/node_modules/rc-')) return 'vendor-rc'
          if (/(\/node_modules\/(?:react-markdown|remark-|rehype-|unified|mdast-|hast-|micromark|vfile|unist-)\/)/.test(normalized)) return 'vendor-markdown'
          if (normalized.includes('/node_modules/katex/')) return 'vendor-katex'
          if (normalized.includes('/node_modules/highlight.js/')) return 'vendor-highlight'
          if (normalized.includes('/node_modules/zustand/')) return 'vendor-state'
          return undefined
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: backendUrl,
        changeOrigin: true,
      },
    },
  },
})
