# Lambda Labs Bare-Bones Setup Guide
## Assuming NOTHING is pre-installed

This guide assumes your Lambda Labs instance is completely bare-bones and may not have GPU drivers, CUDA, or even basic tools installed.

---

## Phase 1: Initial Connection and System Verification

### Step 1.1: Get your Lambda Labs SSH key

1. Go to Lambda Labs dashboard
2. Navigate to your instance
3. Download SSH key (save as `~/.ssh/graphmert-lambda`)
4. Set proper permissions:
   ```bash
   chmod 600 ~/.ssh/graphmert-lambda
   ```

### Step 1.2: First connection and system check

```bash
# Connect to Lambda instance
ssh -i ~/.ssh/graphmert-lambda ubuntu@150.136.213.255

# Once connected, check what we have:
```

Run these verification commands **one by one** and note the results:

```bash
# Check OS version
cat /etc/os-release

# Check if nvidia-smi exists
which nvidia-smi
nvidia-smi  # May fail if drivers not installed

# Check if CUDA exists
which nvcc
nvcc --version  # May fail if CUDA not installed

# Check Python
which python3
python3 --version

# Check pip
which pip3
pip3 --version

# Check available disk space
df -h

# Check RAM
free -h

# Check if we have sudo access
sudo echo "Sudo works"
```

**IMPORTANT:** Note down which commands succeed and which fail. We'll install missing components next.

---

## Phase 2: Install System Dependencies

### Step 2.1: Update system packages

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### Step 2.2: Install basic utilities

```bash
# Install essential build tools
sudo apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    vim \
    tmux \
    htop

# Verify installation
which wget curl git tmux
```

### Step 2.3: Install Python 3.10+ (if not present)

```bash
# Check current Python version
python3 --version

# If Python < 3.10 or not installed, install Python 3.10
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip

# Verify
python3.10 --version
pip3 --version
```

---

## Phase 3: Install NVIDIA Drivers and CUDA Toolkit

### Step 3.1: Check if NVIDIA drivers are installed

```bash
# This will fail if drivers not installed
nvidia-smi
```

### Step 3.2: If nvidia-smi FAILS, install NVIDIA drivers

**Option A: Using Ubuntu package manager (easier)**

```bash
# Check available NVIDIA drivers
ubuntu-drivers devices

# Install recommended driver
sudo ubuntu-drivers autoinstall

# OR install specific version (e.g., 535)
sudo apt-get install -y nvidia-driver-535

# Reboot required after driver installation
sudo reboot
```

After reboot, reconnect and verify:
```bash
ssh -i ~/.ssh/graphmert-lambda ubuntu@150.136.213.255
nvidia-smi
```

**Option B: Using NVIDIA's official installer (if Option A fails)**

```bash
# Download NVIDIA driver (check latest at https://www.nvidia.com/Download/index.aspx)
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/535.154.05/NVIDIA-Linux-x86_64-535.154.05.run

# Make executable
chmod +x NVIDIA-Linux-x86_64-535.154.05.run

# Install (this may take 5-10 minutes)
sudo ./NVIDIA-Linux-x86_64-535.154.05.run

# Reboot
sudo reboot
```

### Step 3.3: Install CUDA Toolkit 11.8

```bash
# Download CUDA 11.8 installer
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# Make executable
chmod +x cuda_11.8.0_520.61.05_linux.run

# Install CUDA (deselect driver installation if you already installed drivers)
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH (add to ~/.bashrc for persistence)
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
nvidia-smi
```

**Expected nvidia-smi output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 11.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        ...          | ...                  | ...                  |
|   0  NVIDIA A100-SXM...       | ...                  | ...                  |
|   1  NVIDIA A100-SXM...       | ...                  | ...                  |
|   ...                         | ...                  | ...                  |
|   7  NVIDIA A100-SXM...       | ...                  | ...                  |
+-----------------------------------------------------------------------------+
```

---

## Phase 4: Transfer Project Files

### Step 4.1: Create project directory on Lambda

```bash
# On Lambda instance
mkdir -p ~/graphmert
mkdir -p ~/graphmert/data
cd ~/graphmert
```

### Step 4.2: Transfer files from local machine

**Open a NEW terminal on your LOCAL machine** (don't close Lambda SSH session):

```bash
# Navigate to your project directory
cd /home/wassie/Desktop/graphmert

# Create a deployment tarball (excludes large unnecessary files)
tar czf graphmert_deploy.tar.gz \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='checkpoints*' \
    --exclude='logs' \
    --exclude='cloud_deploy' \
    graphmert/ \
    train_cloud.py \
    start_training_v2.sh \
    requirements.txt \
    data/python_chain_graphs_1024_v2.pt

# Check tarball size (should be ~85MB)
ls -lh graphmert_deploy.tar.gz

# Transfer to Lambda (this will take 1-3 minutes depending on connection)
scp -i ~/.ssh/graphmert-lambda graphmert_deploy.tar.gz ubuntu@150.136.213.255:~/graphmert/

# Verify transfer
ssh -i ~/.ssh/graphmert-lambda ubuntu@150.136.213.255 "ls -lh ~/graphmert/graphmert_deploy.tar.gz"
```

### Step 4.3: Extract files on Lambda

**Back to Lambda SSH terminal:**

```bash
cd ~/graphmert

# Extract
tar xzf graphmert_deploy.tar.gz

# Verify structure
ls -la
# Should see: graphmert/, train_cloud.py, start_training_v2.sh, requirements.txt, data/

# Verify dataset
ls -lh data/python_chain_graphs_1024_v2.pt
# Should show ~83MB file

# Clean up tarball
rm graphmert_deploy.tar.gz
```

---

## Phase 5: Setup Python Environment

### Step 5.1: Create virtual environment

```bash
cd ~/graphmert

# Create venv using Python 3.10
python3.10 -m venv venv

# Activate
source venv/bin/activate

# You should see (venv) in your prompt
# Verify:
which python
python --version  # Should show Python 3.10.x
```

### Step 5.2: Upgrade pip and install wheel

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Verify pip version (should be 23.x or higher)
pip --version
```

### Step 5.3: Install PyTorch with CUDA 11.8 support

```bash
# Install PyTorch with CUDA 11.8
# This will download ~2GB, takes 5-10 minutes
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

**Expected output:**
```
PyTorch version: 2.0.1+cu118
CUDA available: True
CUDA version: 11.8
GPU count: 8
```

**If CUDA available shows False:**
- Check nvidia-smi works
- Check CUDA toolkit is in PATH
- Reinstall PyTorch with correct CUDA version

### Step 5.4: Install transformers and core dependencies

```bash
# Install transformers and HuggingFace ecosystem
pip install \
    transformers==4.30.0 \
    tokenizers==0.13.3 \
    accelerate==0.20.0 \
    datasets==2.14.0

# Install scientific computing libraries
pip install \
    numpy==1.24.3 \
    scipy==1.10.1 \
    scikit-learn==1.3.0 \
    pandas==2.0.3

# Install utilities
pip install \
    tqdm==4.65.0 \
    wandb==0.15.5
```

### Step 5.5: Install PyTorch Geometric (for graph operations)

```bash
# Install PyG dependencies (this is tricky, must match PyTorch version)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Install PyG
pip install torch-geometric==2.3.1

# Verify installation
python -c "import torch_geometric; print(f'PyG version: {torch_geometric.__version__}')"
```

### Step 5.6: Verify all dependencies

```bash
# Test all imports
python << 'EOF'
import torch
import transformers
import torch_geometric
import numpy as np
import wandb
import tqdm

print("✓ PyTorch:", torch.__version__)
print("✓ CUDA:", torch.cuda.is_available())
print("✓ GPUs:", torch.cuda.device_count())
print("✓ Transformers:", transformers.__version__)
print("✓ PyG:", torch_geometric.__version__)
print("✓ NumPy:", np.__version__)
print("✓ All dependencies loaded successfully!")
EOF
```

---

## Phase 6: Configure Weights & Biases

### Step 6.1: Get W&B API key

1. Go to https://wandb.ai/authorize
2. Copy your API key

### Step 6.2: Login to W&B

```bash
# Login to W&B
wandb login

# Paste your API key when prompted
# Should see: "Successfully logged in to Weights & Biases!"
```

---

## Phase 7: Pre-Training Verification

### Step 7.1: Test dataset loading

```bash
cd ~/graphmert
source venv/bin/activate

# Test loading the dataset
python << 'EOF'
import torch
from graphmert.chain_graph_dataset import ChainGraphDataset

print("Loading dataset...")
dataset = ChainGraphDataset("data/python_chain_graphs_1024_v2.pt")

print(f"✓ Dataset loaded successfully!")
print(f"  Total examples: {len(dataset)}")

# Test loading one sample
sample = dataset[0]
print(f"  Sample keys: {sample.keys()}")
print(f"  Input IDs shape: {sample['input_ids'].shape}")
print(f"  Graph structure shape: {sample['graph_structure'].shape}")

print("\n✓ Dataset is ready for training!")
EOF
```

### Step 7.2: Test model initialization

```bash
python << 'EOF'
import torch
from graphmert.models.graphmert import GraphMERT

print("Initializing model...")
model = GraphMERT(
    num_relations=12,
    hidden_size=768,
    num_attention_heads=12,
    num_layers=12
)

print(f"✓ Model initialized successfully!")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Test moving to GPU
if torch.cuda.is_available():
    model = model.cuda()
    print(f"✓ Model moved to GPU: {torch.cuda.get_device_name(0)}")

print("\n✓ Model is ready for training!")
EOF
```

### Step 7.3: Check disk space for checkpoints

```bash
# Check available space (need ~10GB for checkpoints)
df -h ~

# Create checkpoint directory
mkdir -p ~/graphmert/checkpoints_v2
mkdir -p ~/graphmert/logs
```

---

## Phase 8: Start Training

### Step 8.1: Make training script executable

```bash
cd ~/graphmert
chmod +x start_training_v2.sh

# Review the script contents
cat start_training_v2.sh
```

### Step 8.2: Start training in tmux session

```bash
# Create tmux session
tmux new -s graphmert_training

# Inside tmux, activate environment
cd ~/graphmert
source venv/bin/activate

# Start training (test run with 5 epochs first)
python train_cloud.py \
    --data_path data/python_chain_graphs_1024_v2.pt \
    --output_dir checkpoints_v2 \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --weight_decay 0.02 \
    --lambda_mlm 0.6 \
    --use_wandb \
    --wandb_project graphmert-v2

# Training will start!
# To detach from tmux: Ctrl+B, then D
# To reattach: tmux attach -t graphmert_training
```

### Step 8.3: Monitor training

**In a separate terminal (or after detaching from tmux):**

```bash
# SSH into Lambda
ssh -i ~/.ssh/graphmert-lambda ubuntu@150.136.213.255

# Monitor GPU usage
watch -n 1 nvidia-smi

# View training logs (in another pane)
tail -f ~/graphmert/logs/training_v2_*.log

# Reattach to tmux to see live output
tmux attach -t graphmert_training
```

### Step 8.4: Monitor on W&B dashboard

1. Go to https://wandb.ai
2. Navigate to your project: `graphmert-v2`
3. Watch live training metrics:
   - Total loss
   - MLM loss (should decrease from ~7.8 to ~6.5)
   - MNM loss (should decrease from ~9.1 to ~6.8, NO SPIKE!)
   - GPU utilization
   - Memory usage

---

## Phase 9: Expected Training Progress

### First 5 epochs (test run ~30-45 min, ~$6-10)

```
Epoch 0:
  Train Loss: 8.30 (MLM: 7.80, MNM: 9.10)
  Val Loss:   7.10 (MLM: 7.10, MNM: 7.20)

Epoch 1:
  Train Loss: 7.20 (MLM: 7.10, MNM: 7.50)
  Val Loss:   6.90 (MLM: 7.05, MNM: 7.00)  ✓ NO SPIKE!

Epoch 2:
  Train Loss: 6.95 (MLM: 6.90, MNM: 7.05)
  Val Loss:   6.70 (MLM: 6.85, MNM: 6.80)  ✓ IMPROVING!

Epoch 3:
  Train Loss: 6.80 (MLM: 6.75, MNM: 6.90)
  Val Loss:   6.60 (MLM: 6.75, MNM: 6.70)

Epoch 4:
  Train Loss: 6.70 (MLM: 6.65, MNM: 6.80)
  Val Loss:   6.55 (MLM: 6.70, MNM: 6.65)
```

**Key Success Indicators:**
- ✅ MNM loss should steadily decrease (NOT spike to 16+)
- ✅ Validation loss should be close to training loss
- ✅ Both MLM and MNM should converge together

**If you see this, continue to full 15 epochs!**

### Full training (15 epochs, ~1.5-2.5 hours, ~$18-35)

```bash
# If test run looks good, stop and restart with 15 epochs
tmux attach -t graphmert_training
# Ctrl+C to stop

# Start full training
python train_cloud.py \
    --data_path data/python_chain_graphs_1024_v2.pt \
    --output_dir checkpoints_v2 \
    --num_epochs 15 \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --weight_decay 0.02 \
    --lambda_mlm 0.6 \
    --use_wandb \
    --wandb_project graphmert-v2 \
    --resume_from checkpoints_v2/checkpoint_latest.pt  # Resume from epoch 5

# Detach: Ctrl+B, then D
```

---

## Phase 10: Post-Training

### Step 10.1: Verify checkpoints

```bash
cd ~/graphmert/checkpoints_v2
ls -lh

# Should see:
# checkpoint_epoch_0.pt
# checkpoint_epoch_1.pt
# ...
# checkpoint_epoch_14.pt
# checkpoint_best.pt
# checkpoint_latest.pt
```

### Step 10.2: Download checkpoints to local machine

**On your LOCAL machine:**

```bash
# Create local directory
mkdir -p /home/wassie/Desktop/graphmert/checkpoints_v2_lambda

# Download all checkpoints
scp -i ~/.ssh/graphmert-lambda -r \
    ubuntu@150.136.213.255:~/graphmert/checkpoints_v2/ \
    /home/wassie/Desktop/graphmert/checkpoints_v2_lambda/

# This will take 5-10 minutes (checkpoints are ~500MB each)
```

### Step 10.3: Download logs

```bash
# Download training logs
scp -i ~/.ssh/graphmert-lambda -r \
    ubuntu@150.136.213.255:~/graphmert/logs/ \
    /home/wassie/Desktop/graphmert/logs_lambda/
```

### Step 10.4: Terminate Lambda instance

**IMPORTANT:** Don't forget to terminate the instance to stop billing!

1. Go to Lambda Labs dashboard
2. Find your instance
3. Click "Terminate"
4. Confirm termination

---

## Troubleshooting Common Issues

### Issue 1: nvidia-smi not found

```bash
# Reinstall NVIDIA drivers
sudo ubuntu-drivers autoinstall
sudo reboot
```

### Issue 2: CUDA out of memory

```bash
# Reduce batch size
python train_cloud.py ... --batch_size 4  # Instead of 8
```

### Issue 3: Connection lost during training

```bash
# Training continues in tmux! Just reconnect:
ssh -i ~/.ssh/graphmert-lambda ubuntu@150.136.213.255
tmux attach -t graphmert_training
```

### Issue 4: PyTorch can't find CUDA

```bash
# Check CUDA paths
echo $PATH
echo $LD_LIBRARY_PATH

# Add to ~/.bashrc if missing
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

### Issue 5: torch-scatter installation fails

```bash
# Install dependencies first
pip install ninja

# Then try installing with explicit versions
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

### Issue 6: W&B not logging

```bash
# Re-login
wandb login --relogin

# Check W&B is enabled in training
# Should see: --use_wandb flag in training command
```

### Issue 7: Training diverges (loss becomes NaN)

```bash
# Lower learning rate
python train_cloud.py ... --learning_rate 1e-4  # Instead of 2e-4
```

---

## Quick Command Reference

```bash
# Connect to Lambda
ssh -i ~/.ssh/graphmert-lambda ubuntu@150.136.213.255

# Activate environment
cd ~/graphmert && source venv/bin/activate

# Check GPU
nvidia-smi

# Start training
tmux new -s graphmert_training
python train_cloud.py --data_path data/python_chain_graphs_1024_v2.pt --output_dir checkpoints_v2 --num_epochs 15 --batch_size 8 --learning_rate 2e-4 --weight_decay 0.02 --lambda_mlm 0.6 --use_wandb --wandb_project graphmert-v2

# Detach from tmux
Ctrl+B, then D

# Reattach to tmux
tmux attach -t graphmert_training

# Monitor GPU
watch -n 1 nvidia-smi

# Download checkpoints (from local machine)
scp -i ~/.ssh/graphmert-lambda -r ubuntu@150.136.213.255:~/graphmert/checkpoints_v2/ ./
```

---

## Checklist Summary

- [ ] Phase 1: Connect to Lambda and verify system
- [ ] Phase 2: Install system dependencies (build tools, git, etc)
- [ ] Phase 3: Install NVIDIA drivers and CUDA 11.8
- [ ] Phase 4: Transfer project files (83MB dataset)
- [ ] Phase 5: Setup Python venv and install dependencies
- [ ] Phase 6: Configure W&B
- [ ] Phase 7: Pre-training verification (test dataset & model)
- [ ] Phase 8: Start training (5 epoch test run)
- [ ] Phase 9: Full training (15 epochs if test successful)
- [ ] Phase 10: Download checkpoints and terminate instance

**Estimated total setup time:** 30-60 minutes
**Estimated training time:** 1.5-2.5 hours
**Total cost:** ~$24-42 (setup + training)

Good luck! 🚀
