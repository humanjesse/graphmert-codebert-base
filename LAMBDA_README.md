# Lambda Labs Training Guide - START HERE

## 🎯 Which Guide Should I Use?

**Use this decision tree:**

```
Do you have a Lambda Labs instance running?
│
├─ No → Go to Lambda Labs dashboard and request instance first
│
└─ Yes → Have you SSH'd into it yet?
    │
    ├─ No → Start with: QUICK_START_LAMBDA.md (Phase 1-2)
    │
    └─ Yes → Does `nvidia-smi` work?
        │
        ├─ Don't know → Run: ./verify_lambda_system.sh
        │
        ├─ No → Follow: LAMBDA_BARE_BONES_SETUP.md (complete guide)
        │
        └─ Yes → Does `nvcc --version` show CUDA 11.8+?
            │
            ├─ No → Follow: LAMBDA_BARE_BONES_SETUP.md (Phase 3+)
            │
            └─ Yes → Skip to: QUICK_START_LAMBDA.md (Phase 4+)
```

## 📚 Documentation Files

### Active Documentation (Use These!)

| File | When to Use |
|------|-------------|
| **QUICK_START_LAMBDA.md** | Quick reference card with copy-paste commands |
| **LAMBDA_BARE_BONES_SETUP.md** | Complete guide assuming nothing pre-installed (30-60 min) |
| **verify_lambda_system.sh** | Run this FIRST to check what's installed |
| **start_training_v2.sh** | After setup, use this to launch training |

### Archived Documentation

Old scripts that assumed pre-installed components are in: `archive/lambda_old_deployment/`

**Don't use these unless** you've verified nvidia-smi and CUDA are already working.

## 🚀 Recommended Workflow

### Step 1: Request Instance
- Instance type: **8x A100** (or smaller for testing)
- Wait for status: **"Running"**
- Download SSH key → `~/.ssh/graphmert-lambda`
- Note the IP address

### Step 2: First Connection
```bash
# From your local machine
chmod 600 ~/.ssh/graphmert-lambda
ssh -i ~/.ssh/graphmert-lambda ubuntu@YOUR_IP
```

### Step 3: System Verification
```bash
# Transfer verification script
scp -i ~/.ssh/graphmert-lambda verify_lambda_system.sh ubuntu@YOUR_IP:~/

# On Lambda instance
chmod +x ~/verify_lambda_system.sh
~/verify_lambda_system.sh
```

**The script will tell you exactly what's missing!**

### Step 4: Follow Appropriate Guide

**If everything is installed:**
- Jump to QUICK_START_LAMBDA.md (Phase 4: Transfer Files)

**If things are missing:**
- Follow LAMBDA_BARE_BONES_SETUP.md step-by-step
- Install only what the verification script says is missing

### Step 5: Train
```bash
# After setup complete
cd ~/graphmert
source venv/bin/activate

# Test run (5 epochs, ~$6-10)
python train_cloud.py \
    --data_path data/python_chain_graphs_1024_v2.pt \
    --output_dir checkpoints_v2 \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --use_wandb \
    --wandb_project graphmert-v2
```

## 📊 Expected Costs & Timeline

| Phase | Time | Cost |
|-------|------|------|
| Setup (if bare-bones) | 30-60 min | ~$6-15 |
| Setup (if pre-installed) | 10-15 min | ~$2-4 |
| Test training (5 epochs) | 30-45 min | ~$6-10 |
| Full training (15 epochs) | 1.5-2.5 hrs | ~$18-35 |

**Total: ~$30-60** (depending on what needs installation)

## ⚠️ Common Issues

| Problem | Solution |
|---------|----------|
| Can't SSH | Instance still booting (wait 2-3 min) |
| nvidia-smi not found | Need to install drivers (Phase 3 of bare-bones guide) |
| CUDA not found | Need to install CUDA toolkit (Phase 3 of bare-bones guide) |
| torch.cuda.is_available() = False | Check nvidia-smi and CUDA installation |
| Out of memory | Reduce batch size: `--batch_size 4` |

## 📞 Need Help?

1. Run `verify_lambda_system.sh` and share the output
2. Check the troubleshooting section in LAMBDA_BARE_BONES_SETUP.md
3. Review QUICK_START_LAMBDA.md for common issues

## ✅ Success Checklist

- [ ] Lambda instance running (status: "Running")
- [ ] SSH connection works
- [ ] `nvidia-smi` shows 8x A100 GPUs
- [ ] `nvcc --version` shows CUDA 11.8
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → True
- [ ] Dataset transferred: `~/graphmert/data/python_chain_graphs_1024_v2.pt`
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (PyTorch, transformers, PyG)
- [ ] W&B logged in
- [ ] Training started successfully
- [ ] Epoch 1-2 validation loss decreasing (NO spike to 16+)

---

**Quick Start:** `./verify_lambda_system.sh` → then follow the appropriate guide!
