#!/bin/bash
# Lambda Labs Environment Setup Script for GraphMERT Training
# Run this script immediately after SSH'ing into your Lambda Labs instance

set -e  # Exit on error

echo "========================================================================"
echo "GraphMERT Training Environment Setup"
echo "========================================================================"

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running on Lambda Labs (optional)
echo -e "\n${BLUE}[1/7]${NC} Checking system..."
echo "Hostname: $(hostname)"
echo "OS: $(lsb_release -d | cut -f2)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo -e "${GREEN}✓${NC} System check complete"

# Update system packages
echo -e "\n${BLUE}[2/7]${NC} Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq tmux htop git
echo -e "${GREEN}✓${NC} System packages updated"

# Install Python dependencies
echo -e "\n${BLUE}[3/7]${NC} Setting up Python environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Created virtual environment"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Install PyTorch with CUDA support (for A100)
echo -e "\n${BLUE}[4/7]${NC} Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
echo -e "${GREEN}✓${NC} PyTorch installed"

# Install other requirements
echo -e "\n${BLUE}[5/7]${NC} Installing dependencies from requirements.txt..."
# Filter out potentially problematic packages
pip install -r requirements.txt -q
echo -e "${GREEN}✓${NC} Dependencies installed"

# Test GPU availability
echo -e "\n${BLUE}[6/7]${NC} Testing GPU availability..."
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
EOF
echo -e "${GREEN}✓${NC} GPU test passed"

# Verify dataset exists
echo -e "\n${BLUE}[7/7]${NC} Verifying dataset..."
if [ -f "data/python_chain_graphs_1024.pt" ]; then
    SIZE=$(du -h data/python_chain_graphs_1024.pt | cut -f1)
    echo -e "${GREEN}✓${NC} Dataset found: python_chain_graphs_1024.pt ($SIZE)"
else
    echo -e "${RED}✗${NC} Dataset not found: data/python_chain_graphs_1024.pt"
    echo "Please upload the dataset using:"
    echo "  scp data/python_chain_graphs_1024.pt ubuntu@<instance-ip>:~/graphmert/data/"
fi

# Print summary
echo ""
echo "========================================================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. (Optional) Set up W&B: wandb login"
echo "  3. Start training: ./start_training.sh"
echo ""
echo "Quick test (verify everything works):"
echo "  python test_training.py"
echo ""
echo "========================================================================"
