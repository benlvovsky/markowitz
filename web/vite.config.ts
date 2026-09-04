import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// `base: './'` rather than '/markowitz/'. GitHub Pages serves a project site from a
// subpath, and a hardcoded absolute base breaks the moment the repo is renamed or the
// build is opened from disk. Relative asset URLs work under any prefix, so the same
// artifact serves from /, from /markowitz/, and from `vite preview`. Data files are
// fetched through `import.meta.env.BASE_URL` for the same reason -- see src/data.ts.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: { outDir: 'dist', sourcemap: true },
})
