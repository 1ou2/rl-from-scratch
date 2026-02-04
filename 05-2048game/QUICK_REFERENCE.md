┌────────────────────────────────────────────────────────────────────────────┐
│                    2048 RL TRAINING - QUICK REFERENCE                      │
└────────────────────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════════════════════╗
║                           ⭐ RECOMMENDED: DQN ⭐                            ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─ ALGORITHM ────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Deep Q-Learning with:                                                     │
│  • Epsilon-greedy exploration                                              │
│  • Experience replay buffer                                                │
│  • Target network                                                          │
│  • Double DQN                                                              │
│                                                                             │
│  Loss = MSE(Q(s,a), r + γ * max_a' Q_target(s', a'))                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ HYPERPARAMETERS ──────────────────────────────────────────────────────────┐
│                                                                             │
│  Learning Rate:         1e-4                                               │
│  Gamma (discount):      0.99                                               │
│  Epsilon:               1.0 → 0.01  (decay: 0.995)                         │
│  Batch Size:            64                                                 │
│  Replay Buffer:         100,000                                            │
│  Target Update:         Every 1000 steps                                   │
│  Invalid Penalty:       -10                                                │
│  Episodes:              5000+                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ COMPARISON ───────────────────────────────────────────────────────────────┐
│                                                                             │
│  Method              │ Sample Eff. │ Speed │ Best For                      │
│  ───────────────────────────────────────────────────────────────────────   │
│  DQN (ε-greedy)      │ ⭐⭐⭐⭐⭐  │  Fast │ 2048 ← USE THIS               │
│  Cross-Entropy       │ ⭐⭐        │  Slow │ Simple baselines              │
│  REINFORCE           │ ⭐⭐        │  Slow │ Continuous actions            │
│  A2C/PPO             │ ⭐⭐⭐⭐    │  Fast │ After DQN works               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ WHY DQN WINS ─────────────────────────────────────────────────────────────┐
│                                                                             │
│  ✅ Uses ALL data (not just elite episodes)                                │
│  ✅ Learns from every transition (sample efficient)                        │
│  ✅ Handles long episodes well (500+ moves)                                │
│  ✅ Immediate invalid move penalties                                       │
│  ✅ Proven for similar games (Atari)                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ EXPECTED RESULTS ─────────────────────────────────────────────────────────┐
│                                                                             │
│  Episode 1000:    Max tile = 256-512                                       │
│  Episode 3000:    Max tile = 512-1024                                      │
│  Episode 5000:    Max tile = 1024-2048                                     │
│                                                                             │
│  Training time:   2-3 hours (CPU) | 30 min (GPU)                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ QUICK START ──────────────────────────────────────────────────────────────┐
│                                                                             │
│  # Train agent                                                             │
│  $ python train_2048.py                                                    │
│                                                                             │
│  # Watch trained agent                                                     │
│  $ python play_2048.py --model models/dqn_2048.pth                         │
│                                                                             │
│  # Compare strategies                                                      │
│  $ python compare_strategies.py                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ KEY TAKEAWAY ─────────────────────────────────────────────────────────────┐
│                                                                             │
│  For 2048 game:                                                            │
│                                                                             │
│    ✅ USE: DQN with epsilon-greedy                                         │
│    ✅ LOSS: MSE/Huber (Temporal Difference error)                          │
│    ✅ STRATEGY: Experience replay + target network                         │
│                                                                             │
│    ❌ NOT: Cross-Entropy Method                                            │
│    ❌ NOT: Cross-entropy loss                                              │
│    ❌ NOT: Pure episode-based methods                                      │
│                                                                             │
│  Ready to train? → python train_2048.py                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
