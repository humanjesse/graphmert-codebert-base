#!/bin/bash
# Start GraphMERT training in a tmux session
# This allows training to continue even if you disconnect

set -e

# Configuration
SESSION_NAME="graphmert_training"
DATA_PATH="data/python_chain_graphs_1024.pt"
OUTPUT_DIR="./checkpoints"
WANDB_PROJECT="graphmert-pretraining"

# Training hyperparameters (from paper)
NUM_EPOCHS=25
BATCH_SIZE=32
LEARNING_RATE=4e-4
LAMBDA_MLM=0.6
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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--wandb] [--resume CHECKPOINT] [--batch-size N] [--epochs N]"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "Starting GraphMERT Training"
echo "========================================================================"
echo "Session: $SESSION_NAME"
echo "Dataset: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "W&B enabled: $USE_WANDB"
if [ -n "$RESUME" ]; then
    echo "Resuming from: $RESUME"
fi
echo "========================================================================"

# Check if dataset exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Dataset not found at $DATA_PATH"
    echo "Please upload the dataset first:"
    echo "  scp data/python_chain_graphs_1024.pt ubuntu@<instance-ip>:~/graphmert/data/"
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

# Build training command
TRAIN_CMD="source venv/bin/activate && python train_cloud.py \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
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
LOG_FILE="logs/training_${TIMESTAMP}.log"
TRAIN_CMD="$TRAIN_CMD 2>&1 | tee $LOG_FILE"

# Create tmux session and start training
echo ""
echo "Creating tmux session and starting training..."
echo "Log file: $LOG_FILE"
echo ""

tmux new-session -d -s "$SESSION_NAME" bash -c "$TRAIN_CMD; echo 'Training finished. Press Enter to exit.'; read"

echo "========================================================================"
echo "Training started successfully!"
echo "========================================================================"
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
echo ""
