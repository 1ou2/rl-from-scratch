# 2048 Game with Deep Reinforcement Learning

This directory contains a complete implementation of the 2048 game as a Gymnasium environment, along with deep reinforcement learning agents to solve it.

## 📁 Files Overview

- **`game_2048.py`** - 2048 game as a Gymnasium environment
- **`neural_network.py`** - Deep Q-Network (DQN) architecture
- **`train_2048.py`** - ⭐ Main DQN training script (RECOMMENDED)
- **`train_cem.py`** - Cross-Entropy Method (for comparison)
- **`play_2048.py`** - Play the game or watch trained agent
- **`TRAINING_STRATEGIES.md`** - Detailed strategy comparison and guide

## 🎯 Quick Start

### 1. Train an agent (DQN - Recommended)

```bash
python train_2048.py
```

This will:
- Train a DQN agent for 5000 episodes
- Save checkpoints every 500 episodes to `models/`
- Generate training curves in `plots/`
- Print evaluation results every 100 episodes

### 2. Watch a trained agent play

```bash
python play_2048.py --model models/dqn_2048.pth
```

### 3. Play manually

```bash
python play_2048.py --human
```

## 🧠 Training Strategies

### Deep Q-Learning (DQN) - ⭐ RECOMMENDED

**Why DQN for 2048:**
- ✅ Sample efficient - learns from every move
- ✅ Handles long games (500+ moves)
- ✅ Can penalize invalid moves
- ✅ Proven for similar games (Atari)

**Key Features:**
- Experience replay buffer (100k transitions)
- Target network for stability
- Epsilon-greedy exploration with decay
- Double DQN to reduce overestimation
- Invalid move penalties

**Expected Results:**
- 1000 episodes: reach 256-512 tiles
- 3000 episodes: occasionally reach 1024
- 5000+ episodes: sometimes reach 2048

### Cross-Entropy Method (CEM)

Simpler but less efficient alternative. Good for:
- Quick prototyping
- Educational comparison
- Understanding different RL paradigms

Run with: `python train_cem.py`

## 📊 Architecture

### DQN Network
```
Input (4x4 grid) → Flatten (16)
    ↓
FC(16 → 256) → ReLU
    ↓
FC(256 → 128) → ReLU
    ↓
FC(128 → 64) → ReLU
    ↓
FC(64 → 4) → Q-values
```

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Learning Rate | 1e-4 | Stable learning |
| Gamma | 0.99 | Future reward discount |
| Epsilon | 1.0 → 0.01 | Exploration → Exploitation |
| Batch Size | 64 | Training batch |
| Buffer | 100k | Experience replay |
| Target Update | 1000 | Stabilization frequency |
| Invalid Penalty | -10 | Discourage invalid moves |

## 🎮 Environment Details

### State Space
- 4x4 grid with tile values as log₂
- Empty: 0, tile "2": 1, tile "4": 2, etc.
- Observation shape: (4, 4) dtype=int32

### Action Space
- 0: Up
- 1: Down
- 2: Left
- 3: Right

### Rewards
- Reward = sum of merged tile values
- Invalid move penalty: configurable (default -10)
- Game ends when no moves possible

## 📈 Monitoring Training

Training produces:
1. **Console output** - Progress every 100 episodes
2. **Model checkpoints** - Saved to `models/dqn_2048_ep*.pth`
3. **Training curves** - Plots in `plots/training_curves.png`
4. **Final stats** - JSON in `models/final_stats.json`

## 🔧 Customization

### Adjust Training

Edit hyperparameters in `train_2048.py`:

```python
trainer = DQNTrainer(
    env=env,
    learning_rate=1e-4,        # ← Adjust
    gamma=0.99,                # ← Adjust
    epsilon_decay=0.995,       # ← Adjust
    invalid_move_penalty=-10.0 # ← Adjust
)
```

### Modify Network

Edit architecture in `neural_network.py`:

```python
self.fc1 = nn.Linear(input_size, 256)  # ← Change layer sizes
self.fc2 = nn.Linear(256, 128)
```

## 🚀 Advanced Improvements

After basic DQN works, try:

1. **Prioritized Experience Replay** - Focus on important transitions
2. **Dueling DQN** - Separate value and advantage streams
3. **Noisy Networks** - Learnable exploration
4. **Multi-step Returns** - n-step TD learning
5. **Rainbow DQN** - Combine all improvements

## 📚 Strategy Comparison

See [TRAINING_STRATEGIES.md](TRAINING_STRATEGIES.md) for detailed comparison of:
- Deep Q-Learning (DQN)
- Cross-Entropy Method (CEM)
- Policy Gradient (REINFORCE)
- Actor-Critic (A2C/PPO)

**TL;DR:** Use DQN for best results with 2048.

## 🎓 Learning Resources

- **DQN Paper**: Mnih et al. (2015) "Human-level control through deep RL"
- **Double DQN**: van Hasselt et al. (2016) "Deep RL with Double Q-learning"
- **Gymnasium Docs**: https://gymnasium.farama.org/

## 💡 Tips

1. **Be patient** - 2048 is challenging, needs 5000+ episodes
2. **Monitor curves** - Look for smooth improvement
3. **Check invalid moves** - Should decrease over time
4. **Save checkpoints** - Training takes time
5. **Tune penalties** - Adjust if too many invalid moves

## 🐛 Troubleshooting

**Training is unstable?**
- Lower learning rate (try 5e-5)
- Increase target update frequency
- Check for NaN in losses

**Too many invalid moves?**
- Increase invalid_move_penalty (try -20)
- Increase epsilon_decay (slower exploration)
- Check network is learning (view loss curves)

**Not improving?**
- Train longer (5000+ episodes)
- Increase buffer size
- Try different random seed

## 📝 Example Training Run

```bash
$ python train_2048.py

Starting training for 5000 episodes...
Policy Network: 37636 parameters
Hyperparameters:
  - Learning rate: 0.0001
  - Gamma: 0.99
  - Batch size: 64
  - Epsilon: 1.0 -> 0.01 (decay: 0.995)
  - Buffer capacity: 100000
  - Target update freq: 1000
  - Double DQN: True
  - Invalid move penalty: -10.0

Training: 100%|██████████| 5000/5000 [2:15:30<00:00, 1.63s/it]

Episode 5000/5000
  Avg Reward (last 100): 2456.32
  Avg Score (last 100): 12843.21
  Avg Max Tile (last 100): 512
  Best Score: 24680
  Epsilon: 0.0100
  Buffer size: 100000
  Eval Score: 11234.56 ± 3421.89
  Eval Max Tile: 512

Training complete!
Final best score: 24680
Model saved to models/dqn_2048.pth
Training curves saved to plots/training_curves.png
```

---

**Ready to train? Run:** `python train_2048.py` 🚀
