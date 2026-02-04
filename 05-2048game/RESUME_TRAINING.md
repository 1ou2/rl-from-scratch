# How to Resume Training from Checkpoint

## Quick Start

### Option 1: Resume from latest checkpoint (easiest)
```bash
python resume_training.py
```

### Option 2: Resume from specific checkpoint
```bash
python resume_training.py --checkpoint models/dqn_2048_ep500.pth
```

### Option 3: Use train_2048.py directly
```bash
python train_2048.py --resume models/dqn_2048_ep500.pth
```

## List Available Checkpoints

```bash
python resume_training.py --list
```

## What Gets Restored?

When you resume training, the following are restored:
- ✅ Policy network weights
- ✅ Target network weights  
- ✅ Optimizer state (learning rate, momentum, etc.)
- ✅ Episode count
- ✅ Step count
- ✅ Epsilon value (exploration rate)
- ✅ Training history (scores, rewards, losses)

## Your Checkpoints

Based on your training, you should have:
- `models/dqn_2048_ep500.pth` - After 500 episodes
- `models/dqn_2048_ep1000.pth` - After 1000 episodes
- `models/dqn_2048_ep1500.pth` - After 1500 episodes
- ... and so on every 500 episodes
- `models/dqn_2048.pth` - Final model after 5000 episodes

## Example Workflow

### Continue training for 5000 more episodes:

```bash
# Resume from final checkpoint
python resume_training.py --checkpoint models/dqn_2048.pth
```

The agent will continue from episode 5000, keeping the same epsilon and learned patterns.

### Start fresh training vs Resume:

**Start Fresh:**
```bash
python train_2048.py
# Starts at episode 0, epsilon=1.0, empty replay buffer
```

**Resume:**
```bash
python resume_training.py
# Continues from last episode, keeps epsilon, keeps replay buffer empty but model trained
```

## Note on Replay Buffer

⚠️ The replay buffer is NOT saved (it would be huge - 100k transitions).
- Model weights ARE preserved (the learned knowledge)
- When resuming, buffer starts empty and refills during training
- This is normal and doesn't significantly impact continued learning

## Checking Training Progress

After resuming, you can:

1. **Check the plots:**
   ```bash
   ls -lh plots/training_curves.png
   ```

2. **Check saved stats:**
   ```bash
   cat models/final_stats.json
   ```

3. **Watch the agent play:**
   ```bash
   python play_2048.py --model models/dqn_2048.pth
   ```

## Tips

1. **To reach higher tiles (512, 1024, 2048):**
   - Resume and train for 5000-10000 more episodes
   - The agent will improve gradually

2. **If training seems stuck:**
   - Try lowering learning rate in train_2048.py
   - Or start fresh with different hyperparameters

3. **Save disk space:**
   - You can delete intermediate checkpoints (ep500, ep1000, etc.)
   - Keep the final model and maybe every 2000 episodes
