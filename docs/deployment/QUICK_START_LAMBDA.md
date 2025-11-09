# Lambda Labs Training - Quick Start Commands

## Phase 1: First Connection (LOCAL machine)

```bash
# Set SSH key permissions
chmod 600 ~/.ssh/graphmert-lambda

# Connect to Lambda
ssh -i ~/.ssh/graphmert-lambda ubuntu@YOUR_INSTANCE_IP

# Transfer verification script (from local machine, new terminal)
scp -i ~/.ssh/graphmert-lambda verify_lambda_system.sh ubuntu@YOUR_INSTANCE_IP:~/
```

## Phase 2: System Check (ON LAMBDA)

```bash
# Run verification script
chmod +x ~/verify_lambda_system.sh
~/verify_lambda_system.sh

# This will tell you what's missing
```

## Phase 3: Install Missing Components (ON LAMBDA)

### If nvidia-smi is missing:

```bash
sudo apt-get update
sudo ubuntu-drivers autoinstall
sudo reboot
# Wait 2-3 min, then reconnect
```

### If CUDA is missing:

```bash
# Download CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
chmod +x cuda_11.8.0_520.61.05_linux.run

# Install (deselect driver if already installed)
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
nvidia-smi
```

### If basic tools missing:

```bash
sudo apt-get install -y git wget curl tmux python3.10 python3.10-venv python3-pip
```

## Phase 4: Transfer Project Files (LOCAL machine)

```bash
# Navigate to project
cd /home/wassie/Desktop/graphmert

# Create deployment package
tar czf graphmert_deploy.tar.gz \
    --exclude='venv' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.git' --exclude='checkpoints*' --exclude='logs' \
    graphmert/ train_cloud.py start_training_v2.sh requirements.txt \
    data/python_chain_graphs_1024_v2.pt

# Transfer (~1-3 min for 85MB)
scp -i ~/.ssh/graphmert-lambda graphmert_deploy.tar.gz ubuntu@YOUR_INSTANCE_IP:~/
```

## Phase 5: Setup Environment (ON LAMBDA)

```bash
# Extract files
mkdir -p ~/graphmert && cd ~/graphmert
tar xzf ~/graphmert_deploy.tar.gz
rm ~/graphmert_deploy.tar.gz

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA (~5-10 min)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Install dependencies (~5 min)
pip install transformers==4.30.0 tokenizers==0.13.3 accelerate==0.20.0
pip install numpy scipy scikit-learn pandas tqdm wandb

# Install PyTorch Geometric (~3 min)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install torch-geometric==2.3.1

# Verify all imports
python -c "import torch, transformers, torch_geometric, wandb; print('✓ All dependencies ready!')"
```

## Phase 6: Configure W&B (ON LAMBDA)

```bash
# Login
wandb login
# Paste your API key from https://wandb.ai/authorize
```

## Phase 7: Start Training (ON LAMBDA)

### Test Run (5 epochs, ~30 min, ~$6-10)

```bash
cd ~/graphmert
source venv/bin/activate

# Start tmux session
tmux new -s graphmert_training

# Start training
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

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t graphmert_training
```

### Full Run (15 epochs, ~1.5-2.5 hrs, ~$18-35)

```bash
# If test run looks good (no MNM spike), run full training
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
    --resume_from checkpoints_v2/checkpoint_latest.pt
```

## Phase 8: Monitor Training

```bash
# Monitor GPU (new terminal)
watch -n 1 nvidia-smi

# View logs
tail -f ~/graphmert/logs/training_v2_*.log

# Reattach to training
tmux attach -t graphmert_training

# W&B dashboard
# https://wandb.ai/your-username/graphmert-v2
```

## Phase 9: Download Results (LOCAL machine)

```bash
# Download checkpoints
scp -i ~/.ssh/graphmert-lambda -r \
    ubuntu@YOUR_INSTANCE_IP:~/graphmert/checkpoints_v2/ \
    /home/wassie/Desktop/graphmert/checkpoints_v2_lambda/

# Download logs
scp -i ~/.ssh/graphmert-lambda -r \
    ubuntu@YOUR_INSTANCE_IP:~/graphmert/logs/ \
    /home/wassie/Desktop/graphmert/logs_lambda/
```

## Phase 10: Terminate Instance

**IMPORTANT:** Go to Lambda Labs dashboard and terminate instance to stop billing!

---

## Expected Training Output

### Good Training (v2 dataset):
```
Epoch 0: Train Loss 8.30 (MLM: 7.80, MNM: 9.10), Val Loss 7.10
Epoch 1: Train Loss 7.20 (MLM: 7.10, MNM: 7.50), Val Loss 6.90  ✓ NO SPIKE
Epoch 2: Train Loss 6.95 (MLM: 6.90, MNM: 7.05), Val Loss 6.70  ✓ IMPROVING
```

### Bad Training (would indicate problems):
```
Epoch 0: Train Loss 8.30, Val Loss 7.10
Epoch 1: Train Loss 7.20, Val Loss 18.19  ✗ SPIKE! STOP TRAINING
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| nvidia-smi not found | `sudo ubuntu-drivers autoinstall && sudo reboot` |
| CUDA not found | Install CUDA 11.8 (see Phase 3) |
| CUDA OOM | Reduce batch size: `--batch_size 4` |
| Connection lost | Reconnect: `ssh -i ~/.ssh/graphmert-lambda ubuntu@IP` |
| tmux lost | Reattach: `tmux attach -t graphmert_training` |
| torch-scatter fails | `pip install ninja` then retry |
| W&B not logging | `wandb login --relogin` |
| Training diverges (NaN) | Lower LR: `--learning_rate 1e-4` |

---

## One-Liner Commands

```bash
# Quick connect
ssh -i ~/.ssh/graphmert-lambda ubuntu@YOUR_INSTANCE_IP

# Quick setup (after file transfer)
cd ~/graphmert && python3.10 -m venv venv && source venv/bin/activate && pip install --upgrade pip && pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Quick verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Quick train (test)
cd ~/graphmert && source venv/bin/activate && tmux new -s graphmert_training

# Quick monitor
watch -n 1 nvidia-smi
```

---

## Files You Need

| File | Size | Purpose |
|------|------|---------|
| verify_lambda_system.sh | ~5KB | Check what's installed |
| graphmert_deploy.tar.gz | ~85MB | All code + data |
| LAMBDA_BARE_BONES_SETUP.md | - | Detailed guide |

---

## Estimated Timeline

| Phase | Time | Cost |
|-------|------|------|
| Setup (1-6) | 30-60 min | ~$6-15 |
| Test training (5 epochs) | 30-45 min | ~$6-10 |
| Full training (15 epochs) | 1.5-2.5 hrs | ~$18-35 |
| **Total** | **2.5-4 hrs** | **~$30-60** |

---

## Success Checklist

- [ ] Lambda instance running and accessible
- [ ] `nvidia-smi` shows 8x A100 GPUs
- [ ] `nvcc --version` shows CUDA 11.8
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` returns True
- [ ] Dataset loaded: `ls -lh ~/graphmert/data/python_chain_graphs_1024_v2.pt`
- [ ] W&B logged in: `wandb login`
- [ ] Training started in tmux
- [ ] Epoch 1-2 validation loss decreasing (no spike)
- [ ] Checkpoints being saved
- [ ] Downloaded checkpoints to local machine
- [ ] Lambda instance terminated

Good luck! 🚀
