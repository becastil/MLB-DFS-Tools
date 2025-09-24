#!/bin/bash
set -e

echo "🚀 Starting MLB-DFS-Tools build process..."
echo "🌍 Environment info:"
echo "  - PWD: $(pwd)"
echo "  - USER: $USER"
echo "  - HOME: $HOME"
echo "  - PATH: $PATH"

# Validate environment with detailed info
echo "📋 Validating build environment..."

# Check Node.js with multiple possible locations
echo "🔍 Checking for Node.js..."
if command -v node &> /dev/null; then
    echo "✅ Node.js found: $(which node) ($(node --version))"
elif command -v /usr/bin/node &> /dev/null; then
    echo "✅ Node.js found at /usr/bin/node"
    export PATH="/usr/bin:$PATH"
elif command -v /opt/node/bin/node &> /dev/null; then
    echo "✅ Node.js found at /opt/node/bin"
    export PATH="/opt/node/bin:$PATH"
else
    echo "❌ ERROR: Node.js not found in any expected location"
    echo "🔍 Searching for node in filesystem..."
    find /usr /opt -name "node" -type f 2>/dev/null | head -5 || echo "No node binary found"
    exit 1
fi

# Check npm
echo "🔍 Checking for npm..."
if command -v npm &> /dev/null; then
    echo "✅ npm found: $(which npm) ($(npm --version))"
elif [ -f "$(dirname $(which node))/npm" ]; then
    export PATH="$(dirname $(which node)):$PATH"
    echo "✅ npm found alongside node"
else
    echo "❌ ERROR: npm not found. Node: $(which node)"
    ls -la "$(dirname $(which node))/" || echo "Can't list node directory"
    exit 1
fi

# Check Python
echo "🔍 Checking for Python..."
if command -v python3 &> /dev/null; then
    echo "✅ Python3 found: $(which python3) ($(python3 --version))"
elif command -v python &> /dev/null; then
    echo "✅ Python found: $(which python) ($(python --version))"
else
    echo "❌ ERROR: Python not found"
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
echo "📂 Current directory: $(pwd)"
echo "📂 Contents: $(ls -la)"

cd frontend
echo "📂 Entered frontend directory: $(pwd)"
echo "📂 Frontend contents: $(ls -la)"

# Clean any existing build
echo "🧹 Cleaning previous builds..."
rm -rf dist build
echo "📂 After cleanup: $(ls -la)"

# Install dependencies with clean install for production
echo "📦 Installing frontend dependencies..."
echo "🔍 Node/npm versions inside frontend:"
echo "  - Node: $(node --version)"
echo "  - NPM: $(npm --version)"
echo "  - NPM config: $(npm config get registry)"

# Clean install first, fallback to regular install if it fails
if ! npm ci --production=false; then
    echo "⚠ npm ci failed, trying regular install with legacy peer deps..."
    rm -rf node_modules package-lock.json
    npm install --legacy-peer-deps --production=false
fi

echo "📦 Dependencies installed. Node_modules size:"
du -sh node_modules 2>/dev/null || echo "Could not check node_modules size"

# Build with production optimizations
echo "🔨 Building frontend with Vite..."
echo "📄 Package.json scripts:"
cat package.json | grep -A 10 '"scripts"' || echo "Could not read package.json scripts"

NODE_ENV=production npm run build

echo "🏗️ Build completed. Checking output..."
echo "📂 Frontend directory after build: $(ls -la)"

# Validate build output
echo "🔍 Validating build output..."
if [ ! -d "dist" ]; then
    echo "❌ ERROR: Frontend build failed - dist directory not found"
    echo "📂 Contents of frontend directory:"
    ls -la
    echo "🔍 Looking for any build output directories:"
    find . -type d -name "*build*" -o -name "*dist*" 2>/dev/null || echo "No build directories found"
    echo "📝 Vite config:"
    cat vite.config.js || echo "Could not read vite config"
    exit 1
fi

if [ ! -f "dist/index.html" ]; then
    echo "❌ ERROR: Frontend build failed - dist/index.html not found"
    echo "📂 Contents of dist directory:"
    ls -la dist/ || echo "dist directory is empty or inaccessible"
    echo "🔍 Searching for index.html files:"
    find . -name "index.html" 2>/dev/null || echo "No index.html found anywhere"
    exit 1
fi

echo "✅ Frontend build successful - dist/index.html exists"
echo "📂 Frontend build contents:"
ls -la dist/
echo "📊 Build output size:"
du -sh dist/ 2>/dev/null || echo "Could not check dist size"
echo "🎯 Key files check:"
[ -f "dist/index.html" ] && echo "  ✅ index.html" || echo "  ❌ index.html"
[ -d "dist/assets" ] && echo "  ✅ assets/" || echo "  ❌ assets/"

# Return to project root
cd ..
echo "📂 Back to project root: $(pwd)"

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
echo "🔍 Python environment:"
echo "  - Python: $(python3 --version 2>/dev/null || python --version 2>/dev/null || echo 'Python not found')"
echo "  - Pip: $(pip --version 2>/dev/null || pip3 --version 2>/dev/null || echo 'Pip not found')"
echo "📄 Requirements file:"
head -5 requirements.txt 2>/dev/null || echo "Could not read requirements.txt"

# Use python3 -m pip for better compatibility
if command -v python3 &> /dev/null; then
    python3 -m pip install -r requirements.txt
elif command -v python &> /dev/null; then
    python -m pip install -r requirements.txt
else
    pip install -r requirements.txt
fi

echo "✅ Build completed successfully!"
echo "🎯 Final verification:"
echo "  - Frontend assets: frontend/dist/"
[ -f "frontend/dist/index.html" ] && echo "    ✅ Frontend built" || echo "    ❌ Frontend missing"
echo "  - Python packages installed"
echo "🚀 Ready for deployment!"

# Final directory structure for debugging
echo "📂 Final project structure:"
ls -la
echo "📂 Frontend dist structure:"
ls -la frontend/dist/ 2>/dev/null || echo "No frontend/dist directory"