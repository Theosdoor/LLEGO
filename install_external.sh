#!/bin/bash

echo "Starting installation of external libraries..."

# Check if src/external directory exists
if [ ! -d "src/external" ]; then
  echo "Error: src/external directory not found"
  exit 1
fi

# Install bonsai-dt
# echo ""
# echo "===================================="
# echo "Installing bonsai-dt..."
# if [ -d "src/external/bonsai-dt" ]; then
#   (cd "src/external/bonsai-dt" && uv pip install -e .)
#   echo "✓ Finished bonsai-dt"
# else
#   echo "× Directory src/external/bonsai-dt not found"
# fi

# Install gatree
echo ""
echo "===================================="
echo "Installing gatree..."
if [ -d "src/external/gatree" ]; then
  cd "src/external/gatree"
  uv pip install -e . --no-build-isolation
  if [ $? -eq 0 ]; then
    echo "✓ Finished gatree"
  else
    echo "× Failed to install gatree"
    exit 1
  fi
  cd ../../..
else
  echo "× Directory src/external/gatree not found"
  exit 1
fi

# Install pydl8.5
# echo ""
# echo "===================================="
# echo "Installing pydl8.5..."
# if [ -d "src/external/pydl8.5" ]; then
#   (cd "src/external/pydl8.5" && uv pip install -e .)
#   echo "✓ Finished pydl8.5"
# else
#   echo "× Directory src/external/pydl8.5 not found"
# fi

echo ""
echo "===================================="
echo "Installation process complete"