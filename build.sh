#!/bin/bash
set -e

echo "🚀 Starting MLB-DFS-Tools build process..."

# Validate environment
echo "📋 Validating build environment..."
if ! command -v node &> /dev/null; then
    echo "❌ ERROR: Node.js not found. Please install Node.js."
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ ERROR: npm not found. Please install npm."
    exit 1
fi

if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ ERROR: Python not found. Please install Python 3."
    exit 1
fi

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo "❌ ERROR: Frontend directory not found"
    exit 1
fi

echo "✅ Environment validation complete"

# Build React frontend
echo "🎨 Building React frontend..."
cd frontend

# Clean any existing build
rm -rf dist build

# Install dependencies with clean install for production
echo "📦 Installing frontend dependencies..."
npm ci --production=false

# Build with production optimizations
echo "🔨 Building frontend with Vite..."
NODE_ENV=production npm run build

# Validate build output
if [ ! -d "dist" ]; then
    echo "❌ ERROR: Frontend build failed - dist directory not found"
    echo "📂 Contents of frontend directory:"
    ls -la
    exit 1
fi

if [ ! -f "dist/index.html" ]; then
    echo "❌ ERROR: Frontend build failed - dist/index.html not found"
    echo "📂 Contents of dist directory:"
    ls -la dist/ || echo "dist directory is empty or inaccessible"
    exit 1
fi

echo "✅ Frontend build successful - dist/index.html exists"
echo "📂 Frontend build contents:"
ls -la dist/

# Return to project root
cd ..

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Build completed successfully!"
echo "🎯 Frontend assets available at: frontend/dist/"
echo "🚀 Ready for deployment!"