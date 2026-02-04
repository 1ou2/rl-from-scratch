# Training Strategies for 2048 with Deep RL

## Strategy Comparison

### 1. **Deep Q-Learning (DQN)** ⭐ **RECOMMENDED**

**How it works:**
- Uses epsilon-greedy to select actions
- Learns Q-values: Q(s, a) = expected future reward for action `a` in state `s`
- Loss: Temporal Difference (TD) error with Huber/MSE loss
- Trains on individual transitions via experience replay

**Advantages for 2048:**
✅ Sample efficient - learns from every transition
✅ Handles long episodes well (games can take 500+ moves)
✅ Can penalize invalid moves immediately
✅ Target network stabilizes learning
✅ Experience replay breaks temporal correlation
✅ Proven to work well for games like this

**Formula:**
```
Loss = (Q(s,a) - [r + γ * max_a' Q_target(s', a')])²
```

**Implementation:** See `train_2048.py`

---

### 2. **Cross-Entropy Method (CEM)**

**How it works:**
- Generate N random episodes with current policy
- Select top K% episodes (highest rewards)
- Train network to imitate actions from elite episodes
- Loss: Cross-entropy loss (classification)

**Advantages:**
✅ Simple to implement
✅ No need for value estimation
✅ Works well for simple environments

**Disadvantages for 2048:**
❌ Needs many complete episodes per iteration (slow)
❌ Wastes data from bad episodes
❌ Can't learn from failed moves
❌ Hard to handle invalid moves
❌ Requires high variance in episode outcomes
❌ Less sample efficient

**Formula:**
```
Loss = -Σ log(π(a_elite | s_elite))
```

**When to use:** Simple environments with short episodes, or when you need a simple baseline.

---

### 3. **Policy Gradient (REINFORCE)**

**How it works:**
- Directly optimize policy π(a|s)
- Use episode returns to weight gradient updates
- Loss: Negative log-likelihood weighted by return

**Advantages:**
✅ Can learn stochastic policies
✅ Works well in continuous action spaces
✅ Theoretically sound

**Disadvantages for 2048:**
❌ High variance (needs many episodes)
❌ Only learns at episode end
❌ Sample inefficient
❌ Discrete actions (could use DQN instead)

**Formula:**
```
Loss = -Σ log(π(a|s)) * G_t
```

**When to use:** Continuous actions, stochastic policies needed, or in combination with actor-critic.

---

### 4. **Actor-Critic (A2C/PPO)**

**How it works:**
- Actor network: learns policy π(a|s)
- Critic network: learns value V(s)
- Combines policy gradient with value estimation

**Advantages:**
✅ Lower variance than REINFORCE
✅ More stable than pure policy gradient
✅ Can handle continuous and discrete actions

**Disadvantages for 2048:**
≈ More complex than DQN
≈ Similar performance to DQN for discrete actions
≈ Requires tuning two networks

**When to use:** After DQN works, as an advanced alternative or for comparison.

---

## Recommended Progression

### Phase 1: Start with DQN ⭐
1. Implement basic DQN (already in `train_2048.py`)
2. Tune hyperparameters:
   - Learning rate: 1e-4 to 1e-3
   - Gamma: 0.95 to 0.99
   - Epsilon decay: 0.99 to 0.995
   - Invalid move penalty: -5 to -20

### Phase 2: Enhancements
1. **Double DQN** (already implemented) - reduces overestimation
2. **Prioritized Experience Replay** - focus on important transitions
3. **Dueling DQN** - separate value and advantage streams
4. **Noisy Networks** - learnable exploration

### Phase 3: Advanced (Optional)
1. Try Actor-Critic (A2C/PPO)
2. Multi-step returns (n-step DQN)
3. Rainbow DQN (combines all improvements)

---

## Key Hyperparameters for 2048

| Parameter | Recommended | Purpose |
|-----------|-------------|---------|
| Learning Rate | 1e-4 | Too high → unstable, too low → slow |
| Gamma (γ) | 0.99 | Discount future rewards |
| Epsilon Start | 1.0 | Start with full exploration |
| Epsilon End | 0.01 | Always keep 1% exploration |
| Epsilon Decay | 0.995 | Gradual shift to exploitation |
| Batch Size | 64 | Balance speed vs. stability |
| Buffer Size | 100k | Store enough diverse experiences |
| Target Update | 1000 steps | Stabilize learning |
| Invalid Penalty | -10 | Discourage invalid moves |

---

## Why DQN is Best for 2048

1. **Discrete Actions**: DQN is optimal for discrete action spaces
2. **Deterministic Environment**: No need for stochastic policies
3. **Sample Efficiency**: Learn from every transition, not just episodes
4. **Proven Track Record**: Successfully used for Atari games, which are similar
5. **Invalid Moves**: Can immediately penalize bad actions
6. **Long Episodes**: Experience replay handles this well

---

## Expected Results

With proper tuning, you should see:
- **By 1000 episodes**: Consistently reach 256-512 tiles
- **By 3000 episodes**: Occasionally reach 1024 tile
- **By 5000+ episodes**: Sometimes reach 2048 tile

Performance depends heavily on:
- Network architecture (current architecture is good)
- Reward shaping (invalid move penalties)
- Exploration strategy (epsilon decay)
- Training time and compute

---

## Quick Start

```bash
# Train with DQN
python train_2048.py

# Training will:
# - Save models to models/dqn_2048_ep*.pth
# - Save plots to plots/training_curves.png
# - Print progress every 100 episodes
# - Evaluate every 100 episodes
```

## Tips for Success

1. **Monitor training curves** - look for smooth improvement
2. **Check invalid move rate** - should decrease over time
3. **Watch max tile progression** - should steadily increase
4. **Be patient** - 2048 is hard, needs 5000+ episodes
5. **Adjust penalties** - if too many invalid moves, increase penalty
6. **Save checkpoints** - training is long, save frequently

---

## References

- DQN Paper: Mnih et al. (2015) "Human-level control through deep reinforcement learning"
- Double DQN: van Hasselt et al. (2016) "Deep Reinforcement Learning with Double Q-learning"
- 2048 AI: Various implementations on GitHub show DQN works well
