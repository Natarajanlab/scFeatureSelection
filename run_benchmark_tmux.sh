#!/bin/bash

SESSION_NAME=$1
SCRIPT="runner_scRNA_Seq.py"

if [ -z "$SESSION_NAME" ]; then
    echo "❌ Error: Please provide a tmux session name."
    echo "Usage: ./run_benchmark_tmux.sh <session_name>"
    exit 1
fi

LOGFILE="logs/${SESSION_NAME}_output.log"
ERRFILE="logs/${SESSION_NAME}_error.log"

mkdir -p logs

if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  Tmux session '$SESSION_NAME' already exists. Attaching instead."
    tmux attach -t $SESSION_NAME
    exit 0
else
    echo "🚀 Starting new tmux session '$SESSION_NAME'"
    
    # tmux new-session -d -s $SESSION_NAME
    # tmux send-keys -t $SESSION_NAME "source fs_env/bin/activate" C-m
    # tmux send-keys -t $SESSION_NAME "python3 $SCRIPT > >(tee $LOGFILE) 2> >(tee $ERRFILE >&2)" C-m
    
    tmux new-session -d -s "$SESSION_NAME" \
    "bash -lc 'cd $(pwd) && \
    source fs_env/bin/activate && \
    python3 $SCRIPT 2>&1 | tee \"$LOGFILE\"'"

    echo "📄 Logging to: $LOGFILE"
    echo "🖥️  Reattach with: tmux attach -t $SESSION_NAME"
fi

echo "✅ Benchmark launched in tmux session '$SESSION_NAME'"
echo "📄 Output is being logged to '$LOGFILE'"

echo "To Detach (Leave it Running in Background)"
echo "Ctrl + B, then D"

# How to attach later:
echo "🖥️  Reattach anytime with: tmux attach -t $SESSION_NAME"

# How to kill the session after completion:
echo "Kill session with: tmux kill-session -t $SESSION_NAME"