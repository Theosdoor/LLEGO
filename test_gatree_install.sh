#!/bin/bash
# Quick test script to verify gatree installation

echo "Testing gatree installation..."
echo ""

# Setup environment
echo "1. Setting up base environment..."
uv sync
echo ""

# Install gatree
echo "2. Installing gatree..."
bash install_external.sh
echo ""

# Test import
echo "3. Testing gatree import..."
uv run python -c "
try:
    from gatree.ga.crossover import Crossover
    print('✓ gatree import successful!')
except Exception as e:
    print(f'✗ gatree import failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "===================================="
    echo "✓ All tests passed!"
    echo "===================================="
else
    echo ""
    echo "===================================="
    echo "✗ Tests failed"
    echo "===================================="
    exit 1
fi
