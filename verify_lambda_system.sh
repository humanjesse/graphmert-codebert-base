#!/bin/bash
# System Verification Script for Lambda Labs
# Run this FIRST when you SSH into the Lambda instance

echo "========================================================================="
echo "Lambda Labs System Verification"
echo "========================================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Track what's missing
MISSING_ITEMS=()

echo "1. Operating System"
echo "-------------------"
if [ -f /etc/os-release ]; then
    cat /etc/os-release | grep -E "PRETTY_NAME|VERSION"
    echo -e "${GREEN}✓${NC} OS info available"
else
    echo -e "${RED}✗${NC} Cannot determine OS"
fi
echo ""

echo "2. NVIDIA Driver & GPU"
echo "----------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo -e "${GREEN}✓${NC} nvidia-smi found"
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo -e "${GREEN}✓${NC} Detected $GPU_COUNT GPU(s)"
else
    echo -e "${RED}✗${NC} nvidia-smi not found - NVIDIA drivers not installed"
    MISSING_ITEMS+=("nvidia-drivers")
fi
echo ""

echo "3. CUDA Toolkit"
echo "---------------"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
    echo -e "${GREEN}✓${NC} CUDA toolkit found"
else
    echo -e "${RED}✗${NC} nvcc not found - CUDA toolkit not installed"
    MISSING_ITEMS+=("cuda-toolkit")
fi
echo ""

echo "4. Python"
echo "---------"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo "$PYTHON_VERSION"
    echo -e "${GREEN}✓${NC} Python3 found"

    # Check if it's Python 3.10+
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
    if [ "$PYTHON_MINOR" -ge 10 ]; then
        echo -e "${GREEN}✓${NC} Python version is 3.10+ (compatible)"
    else
        echo -e "${YELLOW}!${NC} Python version < 3.10 (may need upgrade)"
        MISSING_ITEMS+=("python3.10")
    fi
else
    echo -e "${RED}✗${NC} Python3 not found"
    MISSING_ITEMS+=("python3")
fi
echo ""

echo "5. pip"
echo "------"
if command -v pip3 &> /dev/null; then
    pip3 --version
    echo -e "${GREEN}✓${NC} pip3 found"
else
    echo -e "${RED}✗${NC} pip3 not found"
    MISSING_ITEMS+=("python3-pip")
fi
echo ""

echo "6. Python venv"
echo "--------------"
if python3 -m venv --help &> /dev/null; then
    echo -e "${GREEN}✓${NC} venv module available"
else
    echo -e "${RED}✗${NC} venv module not available"
    MISSING_ITEMS+=("python3-venv")
fi
echo ""

echo "7. System Utilities"
echo "-------------------"
UTILS=("git" "wget" "curl" "tmux" "tar")
for util in "${UTILS[@]}"; do
    if command -v $util &> /dev/null; then
        echo -e "${GREEN}✓${NC} $util found"
    else
        echo -e "${RED}✗${NC} $util not found"
        MISSING_ITEMS+=("$util")
    fi
done
echo ""

echo "8. Disk Space"
echo "-------------"
df -h / | tail -1 | awk '{print "Total: "$2"  Used: "$3"  Available: "$4"  ("$5" used)"}'
AVAILABLE_GB=$(df -BG / | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -gt 50 ]; then
    echo -e "${GREEN}✓${NC} Sufficient disk space (need ~50GB for training)"
else
    echo -e "${YELLOW}!${NC} Low disk space (have ${AVAILABLE_GB}GB, need ~50GB)"
fi
echo ""

echo "9. Memory"
echo "---------"
free -h | grep "Mem:" | awk '{print "Total: "$2"  Used: "$3"  Available: "$7}'
AVAILABLE_MEM=$(free -g | grep "Mem:" | awk '{print $7}')
if [ "$AVAILABLE_MEM" -gt 50 ]; then
    echo -e "${GREEN}✓${NC} Sufficient RAM"
else
    echo -e "${YELLOW}!${NC} Limited RAM (${AVAILABLE_MEM}GB available)"
fi
echo ""

echo "10. Sudo Access"
echo "---------------"
if sudo -n true 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Passwordless sudo available"
else
    if sudo -v 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Sudo available (requires password)"
    else
        echo -e "${RED}✗${NC} No sudo access"
    fi
fi
echo ""

echo "========================================================================="
echo "Summary"
echo "========================================================================="
echo ""

if [ ${#MISSING_ITEMS[@]} -eq 0 ]; then
    echo -e "${GREEN}✅ All required components are installed!${NC}"
    echo ""
    echo "You can proceed with:"
    echo "  1. Transfer project files"
    echo "  2. Create Python virtual environment"
    echo "  3. Install dependencies"
    echo "  4. Start training"
else
    echo -e "${YELLOW}⚠ Missing components:${NC}"
    for item in "${MISSING_ITEMS[@]}"; do
        echo "  - $item"
    done
    echo ""
    echo "Installation order:"
    echo "  1. System packages: sudo apt-get update && sudo apt-get install -y ${MISSING_ITEMS[*]}"

    if [[ " ${MISSING_ITEMS[@]} " =~ " nvidia-drivers " ]]; then
        echo "  2. NVIDIA drivers: sudo ubuntu-drivers autoinstall && sudo reboot"
    fi

    if [[ " ${MISSING_ITEMS[@]} " =~ " cuda-toolkit " ]]; then
        echo "  3. CUDA toolkit: See LAMBDA_BARE_BONES_SETUP.md Phase 3"
    fi
fi

echo ""
echo "For detailed installation instructions, see:"
echo "  LAMBDA_BARE_BONES_SETUP.md"
echo ""
echo "========================================================================="
