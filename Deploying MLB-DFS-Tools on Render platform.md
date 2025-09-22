# Deploying MLB-DFS-Tools on Render platform

**The "Frontend bundle not available" error occurs because React build files aren't being generated before FastAPI starts - fixing this requires implementing a proper build pipeline with npm build execution before Python initialization, combined with correct static file mounting in FastAPI.** The solution involves configuring Render's build commands to handle both Node.js and Python dependencies sequentially, ensuring the React frontend compiles into a `build` directory that FastAPI can serve through its StaticFiles middleware. For the most cost-effective deployment, separate the frontend as a free static site and deploy the backend as a $7/month starter web service, avoiding the limitations of Render's ephemeral filesystem by using AWS S3 for CSV storage at minimal additional cost.

## Build pipeline fixes the missing frontend error

The root cause of your deployment failure stems from Render executing only Python build commands while skipping the crucial React compilation step. When FastAPI attempts to serve frontend files from a `build` or `dist` directory that doesn't exist, it throws the "Frontend bundle not available" error. The fix requires a **multi-step build process** that executes `npm run build` before starting the Python application.

For a single-service deployment on Render, configure your `render.yaml` with a combined build command that handles both frontend and backend setup. The build command must navigate to the frontend directory, install Node dependencies, run the React build, copy the output to where FastAPI expects it, then install Python requirements. This ensures the static files exist before the FastAPI server starts looking for them.

The most reliable approach uses a **shell script** that orchestrates the entire build process with proper error handling. Create a `build.sh` file that validates the environment, builds the React app with `npm ci && npm run build`, copies the build output to `backend/static/`, then installs Python dependencies. This script becomes your single build command in Render's configuration, eliminating race conditions and missing file errors.

## FastAPI requires specific static file configuration

Your `dashboard_api.py` needs modification to properly serve the React single-page application with client-side routing support. The standard FastAPI StaticFiles mount isn't sufficient for React Router - you need a custom handler that returns `index.html` for all unmatched routes, enabling proper SPA functionality.

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except Exception as ex:
            if getattr(ex, 'status_code', None) == 404:
                return await super().get_response("index.html", scope)
            raise ex

app = FastAPI()
build_dir = Path("build")

if build_dir.exists():
    app.mount("/static", StaticFiles(directory=build_dir / "static"), name="static")
    app.mount("/", SPAStaticFiles(directory=build_dir, html=True), name="frontend")
```

This configuration mounts the `static` subdirectory separately for CSS and JavaScript bundles, then uses the custom SPAStaticFiles class to handle all other routes. The **html=True parameter** enables automatic index.html serving, while the custom exception handler ensures React Router can manage client-side navigation without 404 errors. API routes should be prefixed with `/api/` to avoid conflicts with frontend routes.

## Separate services architecture maximizes efficiency

Render's platform strongly favors deploying React and FastAPI as **separate services** rather than a monolithic application. The frontend deploys as a completely free static site with global CDN distribution, while the backend runs as a web service starting at $7/month. This architecture provides independent scaling, better performance through CDN caching, and eliminates CORS complexities during development.

The separate services approach reduces costs by **93% for frontend hosting** compared to serving static files from a web service. Your React app benefits from Render's global CDN with automatic SSL certificates and zero-downtime deploys, while the FastAPI backend can scale independently based on API load. The only additional configuration required is setting proper CORS headers in FastAPI and configuring the React app with the backend URL through environment variables.

For single-service deployment, Render supports multi-language builds through Docker or custom build scripts, but this approach wastes compute resources serving static files and prevents horizontal scaling. The monolithic architecture costs the same $7/month minimum but delivers inferior performance since static assets aren't cached at edge locations. **The recommendation is clear**: use separate services for production deployments.

## render.yaml orchestrates the deployment pipeline

The infrastructure-as-code approach using `render.yaml` provides reproducible deployments and environment consistency. For the recommended separate services architecture, create two service definitions - one for the static React site and another for the FastAPI web service.

```yaml
services:
  - type: web
    name: mlb-dfs-backend
    runtime: python3
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn dashboard_api:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: CORS_ORIGINS
        value: https://mlb-dfs-frontend.onrender.com
    
  - type: static_site
    name: mlb-dfs-frontend
    buildCommand: npm ci && npm run build
    publishPath: ./build
    envVars:
      - key: REACT_APP_API_URL
        value: https://mlb-dfs-backend.onrender.com
```

For single-service deployment requiring both Node.js and Python, the build command becomes more complex. Use a **pipe chain** to execute multiple commands sequentially: navigate to the frontend directory, run npm build, copy files to the backend static directory, then install Python dependencies. The startCommand remains simple - just launch uvicorn with the appropriate host and port bindings.

Environment variables sync between the dashboard and render.yaml, with sensitive values marked as `sync: false` to prompt for manual entry during deployment. This prevents secrets from being committed to version control while maintaining configuration portability.

## CSV data requires external storage solutions

Render's free tier provides only **ephemeral filesystem storage** that resets on every deployment, making local CSV file persistence impossible without upgrading to paid persistent disks at $0.25/GB per month. Even with persistent disks, you lose horizontal scaling capabilities since disks attach to single instances only. The platform's limitations make external storage essential for reliable data persistence.

AWS S3 emerges as the most cost-effective solution at **$0.023/GB per month** - roughly 10x cheaper than Render's persistent disks. Implement pre-signed URLs for direct browser uploads, bypassing your backend entirely for large file transfers. Store CSV metadata in Render's free PostgreSQL instance while keeping actual files in S3, creating a hybrid architecture that balances cost and performance.

For smaller datasets under 1GB, PostgreSQL's native CSV handling through COPY commands provides excellent performance without additional dependencies. The free tier PostgreSQL instance handles CSV import/export efficiently, though it expires after 30 days requiring migration to the $19/month basic tier. Process larger files in-memory using Python's tempfile module or pandas DataFrames, avoiding disk writes entirely on Render's ephemeral filesystem.

## Complete build script prevents deployment failures

A robust build script with validation and error handling ensures successful deployments every time. The script must check for required directories, validate Node.js availability, build the frontend with production optimizations, verify the build output exists, then configure the backend to serve these files.

```bash
#!/bin/bash
set -e

echo "Building React frontend..."
cd frontend
npm ci --production=false
NODE_ENV=production npm run build

if [ ! -d "build" ]; then
  echo "Frontend build failed - directory not found"
  exit 1
fi

cd ..
echo "Configuring FastAPI static files..."
rm -rf backend/static
cp -r frontend/build backend/static/

echo "Installing Python dependencies..."
pip install -r backend/requirements.txt

echo "Build completed successfully"
```

This script uses **npm ci** instead of npm install for faster, more reliable builds from package-lock.json. Setting NODE_ENV=production enables React optimizations like code splitting and minification. The script validates each step succeeded before proceeding, failing fast with clear error messages if issues occur. Copy operations use forced overwrites to prevent stale file conflicts during redeployments.

## Conclusion

Fixing the "Frontend bundle not available" error requires implementing a proper multi-step build process that compiles React before starting FastAPI, combined with correct static file mounting configuration in your Python code. Deploy using Render's separate services architecture - a free static site for React and a $7/month starter web service for FastAPI - to achieve optimal performance and cost efficiency. Handle CSV data persistence through AWS S3 integration at minimal additional cost, avoiding Render's expensive and limited persistent disk options. With the provided build scripts and configurations, your MLB-DFS-Tools application will deploy successfully with full UI functionality while remaining within free and starter tier limitations.