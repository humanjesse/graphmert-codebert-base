#!/bin/bash
# Start GraphMERT training with improved v2 dataset
# Optimized hyperparameters for better data quality

set -e

# Configuration
SESSION_NAME="graphmert_training_v2"
DATA_PATH="data/python_chain_graphs_1024_v2.pt"
OUTPUT_DIR="./checkpoints_v2"
WANDB_PROJECT="graphmert-v2"

# Optimized training hyperparameters for v2 dataset
# (Better data quality → fewer epochs, smaller batch, slower LR)
NUM_EPOCHS=15          # Reduced from 25 (better data = less overfitting)
BATCH_SIZE=8           # Reduced from 32 (smaller dataset, better gradients)
LEARNING_RATE=2e-4     # Reduced from 4e-4 (more stable learning)
WEIGHT_DECAY=0.02      # Increased from 0.01 (more regularization)
LAMBDA_MLM=0.6         # Keep same (60% MLM, 40% MNM)
NUM_RELATIONS=12

# Parse command line arguments
USE_WANDB=false
RESUME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--wandb] [--resume CHECKPOINT] [--batch-size N] [--epochs N] [--lr RATE]"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "Starting GraphMERT Training (v2 - Improved Dataset)"
echo "========================================================================"
echo "Session: $SESSION_NAME"
echo "Dataset: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS (optimized for v2)"
echo "Batch size: $BATCH_SIZE (optimized for v2)"
echo "Learning rate: $LEARNING_RATE (optimized for v2)"
echo "Weight decay: $WEIGHT_DECAY (increased regularization)"
echo "W&B enabled: $USE_WANDB"
if [ -n "$RESUME" ]; then
    echo "Resuming from: $RESUME"
fi
echo ""
echo "Dataset Improvements:"
echo "  - Entity linking: 83.9% (was 58.8%)"
echo "  - Balanced relations (calls: 25%, declares: 25%)"
echo "  - Quality filtered (removed 36% poor examples)"
echo "  - Expected: NO epoch 2 validation spike!"
echo "========================================================================"

# Check if dataset exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: v2 dataset not found at $DATA_PATH"
    echo ""
    echo "The v2 dataset should have been created by the data improvement pipeline."
    echo "Check if data/python_chain_graphs_1024_v2.pt exists."
    echo ""
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo ""
    echo "Training session '$SESSION_NAME' already exists!"
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill and restart: tmux kill-session -t $SESSION_NAME && $0"
    echo ""
    exit 1
fi

# Build training command with optimized hyperparameters
TRAIN_CMD="source venv/bin/activate && python train_cloud.py \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --lambda_mlm $LAMBDA_MLM \
    --num_relations $NUM_RELATIONS \
    --checkpoint_every 1"

if [ "$USE_WANDB" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use_wandb --wandb_project $WANDB_PROJECT"
fi

if [ -n "$RESUME" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME"
fi

# Add logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_v2_${TIMESTAMP}.log"
TRAIN_CMD="$TRAIN_CMD 2>&1 | tee $LOG_FILE"

# Create tmux session and start training
echo ""
echo "Creating tmux session and starting training..."
echo "Log file: $LOG_FILE"
echo ""

tmux new-session -d -s "$SESSION_NAME" bash -c "$TRAIN_CMD; echo 'Training finished. Press Enter to exit.'; read"

echo "========================================================================"
echo "Training started successfully with v2 dataset!"
echo "========================================================================"
echo ""
echo "Expected improvements vs v1:"
echo "  ✅ NO validation spike at epoch 2 (was 16.35 → 7.16)"
echo "  ✅ Faster convergence (~10-12 epochs vs 25)"
echo "  ✅ Better final performance (cleaner training signal)"
echo "  ✅ More stable MNM loss (83.9% entity linking vs 58.8%)"
echo ""
echo "To monitor training:"
echo "  - Attach to tmux: tmux attach -t $SESSION_NAME"
echo "  - Detach from tmux: Ctrl+B, then D"
echo "  - View logs: tail -f $LOG_FILE"
echo "  - Check GPU: nvidia-smi"
echo "  - Monitor with htop: htop"
if [ "$USE_WANDB" = true ]; then
    echo "  - View W&B dashboard: https://wandb.ai/$WANDB_PROJECT"
fi
echo ""
echo "To check if training is still running:"
echo "  tmux list-sessions"
echo ""
echo "To stop training:"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
echo "========================================================================"

# Start GPU monitoring in another tmux window
tmux new-window -t "$SESSION_NAME" -n "gpu-monitor" "watch -n 1 nvidia-smi"

# Show session info
tmux list-sessions

echo ""
echo "TIP: Run 'tmux attach -t $SESSION_NAME' to view training progress"
echo "     Compare with v1 results to see improvements!"
echo ""
