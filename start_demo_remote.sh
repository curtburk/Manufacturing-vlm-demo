#!/bin/bash
# Script to start the Counter QA demo on a remote Linux device
# with access from a Windows laptop

echo "================================================"
echo "Counter Manufacturing Quality Assurance System"
echo "Powered by HP ZGX Nano AI Station - Remote Start"
echo "Using BLIP-2 FLAN-T5-XL for Visual Product Inspection"
echo "================================================"
echo ""

# Get the hostname/IP of the Linux server
SERVER_IP=$(hostname -I | awk '{print $1}')

echo "Server Information:"
echo "  Hostname/IP: $SERVER_IP"
echo ""

# Kill any existing processes on the ports
echo "Cleaning up old processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 2

# Check if virtual environment exists
if [ ! -d "counter-env" ]; then
    echo "✗ Virtual environment not found!"
    echo "  Please run install.sh first"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source counter-env/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Check if main.py exists
if [ ! -f "backend/main.py" ]; then
    echo "✗ backend/main.py not found!"
    exit 1
fi

# Check if frontend exists
if [ ! -f "frontend/index.html" ]; then
    echo "⚠ frontend/index.html not found - UI may not load"
fi

echo ""
echo "======================================"
echo "✅ Demo is running!"
echo "======================================"
echo ""
echo "Access the demo from your Windows laptop:"
echo "👉 http://${SERVER_IP}:8000"
echo ""
echo "Backend API endpoints:"
echo "  - Status: http://${SERVER_IP}:8000/"
echo "  - Health: http://${SERVER_IP}:8000/api/health"
echo "  - Product Lines: http://${SERVER_IP}:8000/api/product-lines"
echo "  - Severity Levels: http://${SERVER_IP}:8000/api/severity-levels"
echo ""
echo "Instructions:"
echo "1. Open the web interface in your browser"
echo "2. Upload a product image for inspection"
echo "3. Select the appropriate product line"
echo "4. Click 'Run Quality Inspection'"
echo ""
echo "⚠️  Note: First run will download BLIP-2 model (~17GB)"
echo ""
echo "Email Alert Configuration (optional):"
echo "  Set SMTP_HOST, SMTP_USER, SMTP_PASS, ALERT_EMAIL"
echo "  to enable automatic alerts for critical/major defects"
echo ""
echo "Press Ctrl+C to stop the demo"
echo "======================================"

# Start the FastAPI server
python3 backend/main.py
