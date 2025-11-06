# Lambda Labs Training Guide for GraphMERT

Complete guide for training GraphMERT on Lambda Labs cloud GPUs.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Step 1: Lambda Labs Account Setup](#step-1-lambda-labs-account-setup)
- [Step 2: Launch GPU Instance](#step-2-launch-gpu-instance)
- [Step 3: Connect to Instance](#step-3-connect-to-instance)
- [Step 4: Upload Dataset](#step-4-upload-dataset)
- [Step 5: Environment Setup](#step-5-environment-setup)
- [Step 6: Start Training](#step-6-start-training)
- [Step 7: Monitor Training](#step-7-monitor-training)
- [Step 8: Download Checkpoints](#step-8-download-checkpoints)
- [Cost Tracking](#cost-tracking)
- [Troubleshooting](#troubleshooting)

---

## Overview

**Training Configuration:**
- **Model:** GraphMERT (127M parameters)
- **Dataset:** 16,482 code samples (193 MB)
- **GPU:** A100 40GB (recommended) or V100 16GB
- **Training Time:** 6-9 hours (A100) or 10-13 hours (V100)
- **Estimated Cost:** $15-17 (A100) or $6-8 (V100)

---

## Prerequisites

- Local machine with SSH client
- Dataset file: `data/python_chain_graphs_1024.pt` (193 MB)
- Credit/debit card for Lambda Labs account
- (Optional) Weights & Biases account for monitoring

---

## Step 1: Lambda Labs Account Setup

### 1.1 Create Account

1. Go to https://lambdalabs.com/service/gpu-cloud
2. Click "Sign Up" in the top right
3. Fill in your details (email, name, password)
4. Verify your email address

### 1.2 Add Payment Method

1. Log in to your Lambda Labs account
2. Go to "Billing" in the left sidebar
3. Click "Add Payment Method"
4. Enter credit/debit card details
5. Add initial credits ($20-50 recommended)

### 1.3 Generate SSH Key (First Time Only)

On your **local machine**:

```bash
# Generate SSH key if you don't have one
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub
# Copy the output (starts with "ssh-ed25519")
```

In **Lambda Labs dashboard**:

1. Go to "SSH Keys" in left sidebar
2. Click "Add SSH Key"
3. Paste your public key
4. Give it a name (e.g., "my-laptop")
5. Click "Add Key"

---

## Step 2: Launch GPU Instance

### 2.1 Choose Instance Type

1. Go to "Instances" in Lambda Labs dashboard
2. Click "Launch New Instance"
3. **Recommended:** Select **"1x A100 (40 GB SXM)"**
   - Fastest training (~6-9 hours)
   - Cost: ~$1.29/hour = $15-17 total
4. **Budget Option:** Select **"1x V100 (16 GB)"**
   - Slower but cheaper (~10-13 hours)
   - Cost: ~$0.55/hour = $6-8 total

### 2.2 Configure Instance

- **Region:** Select any available region (closest to you for lower latency)
- **File System:** Default (200 GB) is sufficient
- **SSH Keys:** Select your SSH key from dropdown
- Click **"Launch Instance"**

### 2.3 Wait for Instance to Start

- Status will change from "Booting" to "Running" (1-2 minutes)
- Note the **Instance IP address** (e.g., 67.213.XXX.XXX)
- Instance hostname will be something like `ssh ubuntu@67.213.XXX.XXX`

---

## Step 3: Connect to Instance

### 3.1 SSH Connection

From your **local machine terminal**:

```bash
# Replace with your actual instance IP
ssh ubuntu@67.213.XXX.XXX
```

First time connecting:
- You'll see "The authenticity of host... can't be established"
- Type `yes` and press Enter

You should now see a prompt like:
```
ubuntu@lambda-instance:~$
```

### 3.2 Verify GPU

```bash
nvidia-smi
```

You should see your GPU (A100 or V100) with memory info.

---

## Step 4: Upload Dataset

You need to upload the pre-built dataset to the Lambda instance.

### Option A: SCP Upload (Recommended)

From your **local machine** (in the graphmert directory):

```bash
# Upload dataset (193 MB, takes ~30 seconds)
scp data/python_chain_graphs_1024.pt ubuntu@67.213.XXX.XXX:~/
```

Then on the **Lambda instance**:

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/graphmert.git
cd graphmert

# Move dataset to data folder
mkdir -p data
mv ~/python_chain_graphs_1024.pt data/

# Verify
ls -lh data/python_chain_graphs_1024.pt
```

### Option B: Git LFS (If Dataset is in Git)

If your dataset is tracked with Git LFS:

```bash
# On Lambda instance
git clone https://github.com/YOUR_USERNAME/graphmert.git
cd graphmert
git lfs pull  # Download large files
```

### Option C: Download from Cloud Storage

If you uploaded to Google Drive, Dropbox, or S3:

```bash
# Example for Google Drive (requires gdrive CLI or direct link)
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=FILE_ID' -O data/python_chain_graphs_1024.pt

# Example for public URL
curl -L https://your-storage.com/dataset.pt -o data/python_chain_graphs_1024.pt
```

---

## Step 5: Environment Setup

On the **Lambda instance** (in the graphmert directory):

### 5.1 Run Setup Script

```bash
# Make scripts executable
chmod +x setup_lambda.sh start_training.sh

# Run setup (takes ~5 minutes)
./setup_lambda.sh
```

This will:
- Update system packages
- Create Python virtual environment
- Install PyTorch with CUDA support
- Install all dependencies
- Test GPU availability
- Verify dataset exists

### 5.2 Activate Environment

```bash
source venv/bin/activate
```

### 5.3 (Optional) Setup Weights & Biases

For remote monitoring:

```bash
# Install wandb (already in requirements.txt)
# Login with your W&B account
wandb login
```

You'll be prompted to paste your API key (get it from https://wandb.ai/authorize).

### 5.4 Quick Test (Optional but Recommended)

Verify everything works before starting full training:

```bash
python test_training.py
```

This runs a quick test (~2 minutes) to ensure:
- Model loads correctly
- Dataset loads correctly
- Forward/backward pass works
- GPU is being used

---

## Step 6: Start Training

### 6.1 Basic Training (No Monitoring)

```bash
./start_training.sh
```

### 6.2 With Weights & Biases Monitoring (Recommended)

```bash
./start_training.sh --wandb
```

### 6.3 Resume from Checkpoint

If training was interrupted:

```bash
./start_training.sh --resume checkpoints/checkpoint_latest.pt
```

### 6.4 Custom Configuration

```bash
./start_training.sh \
    --wandb \
    --batch-size 48 \
    --epochs 25
```

### What Happens:

1. Training starts in a **tmux session** named `graphmert_training`
2. You can safely disconnect - training continues in background
3. Logs are saved to `logs/training_TIMESTAMP.log`
4. Checkpoints saved to `checkpoints/` every 5 epochs

---

## Step 7: Monitor Training

### 7.1 Attach to Training Session

```bash
tmux attach -t graphmert_training
```

**To detach** without stopping training: Press `Ctrl+B`, then press `D`

### 7.2 View Logs

```bash
# Follow log in real-time
tail -f logs/training_*.log

# View last 100 lines
tail -n 100 logs/training_*.log
```

### 7.3 Monitor GPU Usage

In tmux, the second window shows GPU monitoring. Or run:

```bash
# One-time check
nvidia-smi

# Continuous monitoring (refreshes every 1 second)
watch -n 1 nvidia-smi
```

### 7.4 Check System Resources

```bash
htop  # Press 'q' to quit
```

### 7.5 Weights & Biases Dashboard

If you used `--wandb`:

1. Go to https://wandb.ai
2. Navigate to your project: `graphmert-pretraining`
3. View real-time metrics:
   - Training loss (MLM + MNM)
   - Validation loss
   - Learning rate schedule
   - GPU utilization

### 7.6 Check if Training is Running

```bash
# List tmux sessions
tmux list-sessions

# Check if Python is running
ps aux | grep train_cloud.py
```

---

## Step 8: Download Checkpoints

### 8.1 Wait for Training to Complete

Training is complete when:
- Tmux session shows "Training finished"
- Logs show "Training Complete!"
- 25 epochs have finished

### 8.2 List Checkpoints

On the **Lambda instance**:

```bash
ls -lh checkpoints/
```

You should see:
- `checkpoint_epoch_5.pt`
- `checkpoint_epoch_10.pt`
- `checkpoint_epoch_15.pt`
- `checkpoint_epoch_20.pt`
- `checkpoint_epoch_25.pt`
- `checkpoint_best.pt` (best validation loss)
- `checkpoint_latest.pt` (for resuming)

### 8.3 Download to Local Machine

From your **local machine**:

```bash
# Download all checkpoints (~500 MB per checkpoint)
scp -r ubuntu@67.213.XXX.XXX:~/graphmert/checkpoints ./checkpoints_cloud

# Or download just the best checkpoint
scp ubuntu@67.213.XXX.XXX:~/graphmert/checkpoints/checkpoint_best.pt ./checkpoint_best.pt
```

### 8.4 Download Logs

```bash
# Download training logs
scp -r ubuntu@67.213.XXX.XXX:~/graphmert/logs ./training_logs
```

---

## Step 9: Cleanup

### 9.1 Terminate Instance

**IMPORTANT:** Always terminate your instance when done to stop billing!

**Option A: From Dashboard (Recommended)**
1. Go to Lambda Labs dashboard
2. Click "Instances"
3. Find your running instance
4. Click "Terminate"
5. Confirm termination

**Option B: From Command Line**

Before terminating, make sure you've downloaded everything you need!

```bash
# On Lambda instance
sudo shutdown now
```

Then confirm termination in the dashboard.

### 9.2 Verify Termination

- Dashboard shows instance as "Terminated"
- Billing should stop immediately
- You can check "Billing" tab to verify

---

## Cost Tracking

### Estimated Costs

**A100 40GB:**
- **Hourly rate:** $1.29/hour
- **Training time:** 6-9 hours
- **Total cost:** $7.74 - $11.61
- **With setup overhead:** ~$15-17

**V100 16GB:**
- **Hourly rate:** $0.55/hour
- **Training time:** 10-13 hours
- **Total cost:** $5.50 - $7.15
- **With setup overhead:** ~$6-8

### Monitor Your Spending

1. Go to Lambda Labs dashboard
2. Click "Billing" in left sidebar
3. View:
   - Current balance
   - Recent charges
   - Instance uptime

### Cost Saving Tips

1. **Terminate immediately after training:** Don't leave instance running idle
2. **Use V100 for budget training:** Cheaper but slower
3. **Resume interrupted training:** Use checkpoints to avoid wasting progress
4. **Download checkpoints quickly:** Don't pay for instance while downloading (or use background scp)

---

## Troubleshooting

### Issue: Dataset Not Found

**Error:** `Error: Dataset not found at data/python_chain_graphs_1024.pt`

**Solution:**
```bash
# Verify dataset location
ls -lh data/

# Re-upload if missing
# (From local machine)
scp data/python_chain_graphs_1024.pt ubuntu@67.213.XXX.XXX:~/graphmert/data/
```

### Issue: Out of Memory (OOM)

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```bash
# Reduce batch size
./start_training.sh --batch-size 16  # Instead of 32

# Or enable gradient accumulation in train_cloud.py
# Add: --gradient_accumulation_steps 2
```

### Issue: Training Stopped Unexpectedly

**Check if tmux session still exists:**
```bash
tmux list-sessions
```

**Resume training:**
```bash
./start_training.sh --resume checkpoints/checkpoint_latest.pt --wandb
```

**Check logs for errors:**
```bash
tail -n 100 logs/training_*.log
```

### Issue: Cannot Connect via SSH

**Solution:**
1. Verify instance is "Running" in dashboard (not "Booting")
2. Check IP address is correct
3. Verify SSH key is added to Lambda Labs
4. Try: `ssh -v ubuntu@IP` for verbose debugging

### Issue: Slow Dataset Upload

**Large file upload taking too long?**

**Solutions:**
1. Use `rsync` instead of `scp` (can resume):
   ```bash
   rsync -avz --progress data/python_chain_graphs_1024.pt ubuntu@IP:~/graphmert/data/
   ```

2. Upload to cloud storage first (Google Drive, S3), then download on instance:
   ```bash
   # On Lambda instance
   wget https://your-storage-url/dataset.pt -O data/python_chain_graphs_1024.pt
   ```

### Issue: tmux Not Responding

**Cannot attach to tmux session:**
```bash
# Kill stuck session
tmux kill-session -t graphmert_training

# Restart training
./start_training.sh --wandb
```

### Issue: W&B Login Issues

**Error:** `wandb: ERROR authentication failed`

**Solution:**
```bash
# Re-login
wandb login

# Or use offline mode (no W&B)
./start_training.sh  # Without --wandb flag
```

---

## Advanced Topics

### Using Multiple GPUs

If you launch a multi-GPU instance (e.g., 8x A100):

```bash
# Modify train_cloud.py to use DataParallel or DistributedDataParallel
# Or use accelerate library

accelerate config  # Follow prompts for multi-GPU
accelerate launch train_cloud.py --data_path data/python_chain_graphs_1024.pt --use_wandb
```

### Hyperparameter Tuning

Create a grid search script:

```bash
for LR in 1e-4 2e-4 4e-4 8e-4; do
    for LAMBDA in 0.5 0.6 0.7; do
        python train_cloud.py \
            --data_path data/python_chain_graphs_1024.pt \
            --learning_rate $LR \
            --lambda_mlm $LAMBDA \
            --output_dir checkpoints/lr${LR}_lambda${LAMBDA} \
            --use_wandb
    done
done
```

### Scheduled Training

Use `cron` or `at` for scheduled training:

```bash
# Start training at specific time
echo "./start_training.sh --wandb" | at 10:00 PM
```

---

## Quick Reference Commands

### Connect
```bash
ssh ubuntu@<IP>
```

### Start Training
```bash
./start_training.sh --wandb
```

### Monitor Training
```bash
tmux attach -t graphmert_training  # Ctrl+B, D to detach
tail -f logs/training_*.log
watch -n 1 nvidia-smi
```

### Download Results
```bash
scp -r ubuntu@<IP>:~/graphmert/checkpoints ./checkpoints_cloud
```

### Terminate
Lambda Labs Dashboard → Instances → Terminate

---

## Support

- **Lambda Labs Support:** support@lambdalabs.com
- **Lambda Labs Docs:** https://lambdalabs.com/blog/getting-started-with-lambda-cloud
- **GraphMERT Issues:** [Your GitHub Issues Page]

---

## Summary Checklist

- [ ] Lambda Labs account created and payment added
- [ ] SSH key generated and added to Lambda Labs
- [ ] GPU instance launched (A100 or V100)
- [ ] Connected via SSH
- [ ] Dataset uploaded to instance
- [ ] Environment setup completed (`./setup_lambda.sh`)
- [ ] (Optional) W&B logged in
- [ ] Training started (`./start_training.sh --wandb`)
- [ ] Training monitored (tmux/W&B)
- [ ] Training completed (all 25 epochs)
- [ ] Checkpoints downloaded
- [ ] Instance terminated

**Estimated total time:** 7-10 hours (mostly training)
**Estimated total cost:** $6-17 depending on GPU choice

---

Good luck with your training! 🚀
