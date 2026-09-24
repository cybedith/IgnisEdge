#!/bin/bash
TARGET=~/Desktop/IgnisEdge_Monorepo
cd $TARGET

mkdir -p edge_core backend frontend

# Copy edge core
echo "Copying edge_core..."
rsync -a --exclude='venv' --exclude='__pycache__' --exclude='models' --exclude='dataset*' --exclude='capturas' --exclude='.git' ~/IgnisEdge/ edge_core/

# Copy backend
echo "Copying backend..."
rsync -a --exclude='venv' --exclude='__pycache__' --exclude='.git' --exclude='credenciales.json' ~/ignis-edge-backend/ backend/

# Copy frontend
echo "Copying frontend..."
rsync -a --exclude='node_modules' --exclude='dist' --exclude='.git' --exclude='.env.local' ~/Descargas/H1/frontend/ignis-edge-frontend/ frontend/

echo "Creating .gitignore..."
cat << 'IGNORE' > .gitignore
venv/
node_modules/
dist/
__pycache__/
*.pyc
.env
.env.local
credenciales.json
models/
dataset*/
capturas/
.DS_Store
IGNORE

echo "Done!"
