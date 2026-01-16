#!/bin/bash
# Installation script for Counter Manufacturing QA System
# Powered by HP ZGX Nano AI Station

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "Counter Manufacturing Quality Assurance System"
echo "Installation Script"
echo "Powered by HP ZGX Nano AI Station"
echo "============================================================"
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Found Python: ${PYTHON_VERSION}"

# Create virtual environment if it doesn't exist
if [ ! -d "manu-env" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv manu-env
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source counter-env/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA support..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# Install other dependencies
echo ""
echo "Installing dependencies..."
pip install -r backend/requirements.txt

echo ""
echo "============================================================"
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Start the demo: ./start_demo.sh"
echo "   (Models will be downloaded on first run - ~17GB)"
echo ""
echo "2. For remote access: ./start_demo_remote.sh"
echo ""
echo "Optional: Configure email alerts by setting environment variables:"
echo "  export SMTP_HOST=smtp.your-server.com"
echo "  export SMTP_PORT=587"
echo "  export SMTP_USER=your-email@domain.com"
echo "  export SMTP_PASS=your-password"
echo "  export ALERT_EMAIL=production.lead@counter.com"
echo "============================================================"
