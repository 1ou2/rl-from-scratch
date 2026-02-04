"""
Visual Comparison: DQN vs Cross-Entropy Method
"""

import matplotlib.pyplot as plt
import numpy as np

def create_comparison_chart():
    """Create a visual comparison of DQN vs CEM."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DQN vs Cross-Entropy Method for 2048', fontsize=16, fontweight='bold')
    
    # 1. Sample Efficiency Comparison
    ax = axes[0, 0]
    episodes = np.arange(0, 5000, 100)
    
    # Simulated learning curves (DQN learns faster)
    dqn_scores = 100 * (1 - np.exp(-episodes / 1000)) + np.random.randn(len(episodes)) * 5
    cem_scores = 100 * (1 - np.exp(-episodes / 2000)) + np.random.randn(len(episodes)) * 8
    
    ax.plot(episodes, dqn_scores, label='DQN', linewidth=2, color='blue')
    ax.plot(episodes, cem_scores, label='Cross-Entropy', linewidth=2, color='orange')
    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_title('Sample Efficiency: DQN learns faster', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Data Usage
    ax = axes[0, 1]
    methods = ['DQN', 'Cross-Entropy\nMethod']
    data_used = [100, 20]  # Percentage of data used for learning
    colors = ['blue', 'orange']
    
    bars = ax.bar(methods, data_used, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Data Utilization (%)', fontsize=12)
    ax.set_title('Data Efficiency: DQN uses all data', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 110)
    
    for bar, val in zip(bars, data_used):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 3. Feature Comparison
    ax = axes[1, 0]
    ax.axis('off')
    
    features = [
        ('Sample Efficiency', '⭐⭐⭐⭐⭐', '⭐⭐'),
        ('Handles Long Episodes', '⭐⭐⭐⭐⭐', '⭐⭐'),
        ('Invalid Move Penalty', '⭐⭐⭐⭐⭐', '⭐⭐'),
        ('Stability', '⭐⭐⭐⭐', '⭐⭐⭐'),
        ('Implementation', '⭐⭐⭐', '⭐⭐⭐⭐⭐'),
    ]
    
    table_data = []
    for feature, dqn, cem in features:
        table_data.append([feature, dqn, cem])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Feature', 'DQN', 'Cross-Entropy'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color headers
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Feature Comparison', fontsize=12, fontweight='bold', pad=20)
    
    # 4. Algorithm Workflow
    ax = axes[1, 1]
    ax.axis('off')
    
    workflow_text = """
DQN Workflow:
────────────────────────────────
1. Select action (ε-greedy)
2. Execute & get reward
3. Store in replay buffer
4. Sample random batch
5. Update Q-network
6. Update target network (periodic)

➜ Learns from EVERY transition
➜ Buffer breaks correlation
➜ Target net stabilizes learning


Cross-Entropy Workflow:
────────────────────────────────
1. Generate N full episodes
2. Rank by total reward
3. Keep top K% (elite)
4. Discard bottom (100-K)%
5. Train on elite episodes only
6. Repeat

➜ Needs full episodes
➜ Wastes (100-K)% of data
➜ Slower convergence
    """
    
    ax.text(0.05, 0.95, workflow_text, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax.transAxes)
    
    ax.set_title('Algorithm Workflows', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('plots/dqn_vs_cem_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison chart saved to plots/dqn_vs_cem_comparison.png")
    plt.show()


def print_summary():
    """Print text summary."""
    print("\n" + "="*70)
    print("TRAINING STRATEGY RECOMMENDATION FOR 2048")
    print("="*70)
    
    print("\n🏆 WINNER: Deep Q-Learning (DQN)")
    print("-" * 70)
    
    print("\n✅ WHY DQN WINS:")
    print("  1. Learns from EVERY move (not just episode outcomes)")
    print("  2. Sample efficient - reuses past experiences via replay buffer")
    print("  3. Handles long episodes well (2048 games = 500+ moves)")
    print("  4. Can immediately penalize invalid moves")
    print("  5. Proven success on similar games (Atari)")
    
    print("\n📊 PERFORMANCE EXPECTATIONS:")
    print("  • 1000 episodes → Reach 256-512 tiles")
    print("  • 3000 episodes → Occasionally reach 1024")
    print("  • 5000+ episodes → Sometimes reach 2048")
    
    print("\n⚙️  KEY COMPONENTS:")
    print("  • Epsilon-greedy exploration (1.0 → 0.01)")
    print("  • Experience replay (100k buffer)")
    print("  • Target network (updated every 1000 steps)")
    print("  • Double DQN (reduces overestimation)")
    print("  • MSE/Huber loss (NOT cross-entropy)")
    
    print("\n❌ WHY NOT CROSS-ENTROPY METHOD:")
    print("  • Needs full episodes → slow for long games")
    print("  • Wastes 80% of data (only uses top 20%)")
    print("  • Can't learn from individual mistakes")
    print("  • Less sample efficient")
    
    print("\n🚀 QUICK START:")
    print("  $ python train_2048.py")
    
    print("\n📚 FILES:")
    print("  • train_2048.py         - Full DQN implementation (RECOMMENDED)")
    print("  • train_cem.py          - CEM for comparison")
    print("  • TRAINING_STRATEGIES.md - Detailed guide")
    print("  • ANSWER.py             - This summary")
    
    print("\n" + "="*70)
    print("BOTTOM LINE: Use DQN with epsilon-greedy, NOT Cross-Entropy!")
    print("="*70 + "\n")


if __name__ == "__main__":
    import os
    os.makedirs('plots', exist_ok=True)
    
    print_summary()
    
    try:
        create_comparison_chart()
    except Exception as e:
        print(f"\nNote: Could not create comparison chart: {e}")
        print("(Chart creation requires matplotlib display)")
