# MLB Simulation Dashboard Frontend

This package contains the React single-page application that powers the MLB DFS simulation dashboard. The app was bootstrapped with Vite and styled with Tailwind CSS.

## Available scripts

All commands should be run from this directory:

```bash
npm install        # Install dependencies
npm run dev        # Start the Vite development server
npm run build      # Produce an optimized production build in dist/
npm run lint       # Run ESLint against the source tree
```

During local development the Vite dev server proxies requests from `/api/*` to `http://localhost:8000`, which matches the FastAPI server defined in `src/dashboard_api.py`.
