# Assignment: Multi-Armed Bandit Problem

## Problem Description

You are a data scientist working for an online advertising company. The company runs multiple ad campaigns simultaneously and needs to determine which campaigns generate the highest click-through rates (CTR). However, you don't know the true CTR of each campaign in advance - you must learn through experimentation.

This is a classic **multi-armed bandit problem**:
- Each "arm" represents an advertising campaign
- "Pulling an arm" means showing that campaign's ad to a user
- The "reward" is whether the user clicks (1) or doesn't click (0) the ad
- Your goal is to maximize total clicks over time

The challenge is the **exploration vs exploitation dilemma**:
- **Exploitation**: Show ads from campaigns you think perform best
- **Exploration**: Test other campaigns to potentially find better ones

## Mathematical Foundation

### Key Concepts You Need to Understand:

1. **Action Value**: Q(a) = expected reward when taking action a
2. **Regret**: The difference between optimal performance and your performance
3. **Confidence Intervals**: Measure of uncertainty in your estimates
4. **Beta Distribution**: Used for modeling binary outcomes (click/no-click)

### Algorithms to Implement:

1. **Epsilon-Greedy**: With probability ε, explore randomly; otherwise exploit best known option
2. **Upper Confidence Bound (UCB)**: Select action with highest upper confidence bound
3. **Thompson Sampling**: Bayesian approach using probability distributions

## Progressive Exercises

### Exercise 1: Environment Setup
**Objective**: Create the bandit environment and basic infrastructure

**Requirements**:
- Create a `BanditEnvironment` class that simulates ad campaigns
- Each campaign has a hidden true CTR (between 0 and 1)
- When you "pull" a campaign, return 1 (click) with probability = true CTR, 0 otherwise
- Track which campaign is actually optimal
- Include at least 4 different campaigns with CTRs: [0.05, 0.12, 0.08, 0.15]

**Methods to implement**:
- `__init__(self, true_ctrs)`: Initialize with list of true click-through rates
- `pull_arm(self, campaign_id)`: Return 1 or 0 based on campaign's true CTR
- `get_optimal_arm(self)`: Return ID of best campaign

### Exercise 2: Epsilon-Greedy Algorithm
**Objective**: Implement the simplest exploration strategy

**Requirements**:
- Create `EpsilonGreedy` class
- Maintain estimated CTR for each campaign
- With probability ε (e.g., 0.1), choose random campaign
- Otherwise, choose campaign with highest estimated CTR
- Update estimates using incremental averaging: new_avg = old_avg + (reward - old_avg) / count

**Methods to implement**:
- `__init__(self, n_arms, epsilon)`: Initialize with number of campaigns and exploration rate
- `select_arm(self)`: Return campaign ID to show
- `update(self, arm, reward)`: Update estimates after observing result
- `get_estimates(self)`: Return current CTR estimates

### Exercise 3: Upper Confidence Bound (UCB)
**Objective**: Implement confidence-based exploration

**Requirements**:
- Create `UCB` class
- Use UCB1 formula: UCB(a) = Q(a) + c × √(ln(t) / N(a))
  - Q(a) = estimated value of arm a
  - c = confidence parameter (try c=2)
  - t = total number of pulls so far
  - N(a) = number of times arm a was pulled
- Always pull each arm at least once initially
- Select arm with highest UCB value

**Methods to implement**:
- `__init__(self, n_arms, c)`: Initialize with confidence parameter
- `select_arm(self)`: Return campaign with highest UCB
- `update(self, arm, reward)`: Update estimates and counts
- `get_ucb_values(self)`: Return current UCB values for analysis

### Exercise 4: Thompson Sampling
**Objective**: Implement Bayesian approach using Beta distributions

**Requirements**:
- Create `ThompsonSampling` class
- Model each campaign's CTR using Beta distribution
- Start with Beta(1,1) for each campaign (uniform prior)
- For each decision: sample from each campaign's Beta distribution, choose highest sample
- Update: if click → add 1 to alpha parameter, if no click → add 1 to beta parameter
- Beta(α,β) has mean α/(α+β)

**Methods to implement**:
- `__init__(self, n_arms)`: Initialize Beta parameters
- `select_arm(self)`: Sample from distributions and return best
- `update(self, arm, reward)`: Update Beta parameters
- `get_posterior_means(self)`: Return current estimated CTRs

### Exercise 5: Experiment Runner
**Objective**: Create framework to test and compare algorithms

**Requirements**:
- Create `run_experiment(algorithm, environment, n_steps)` function
- Run algorithm for specified number of steps
- Track: rewards received, arms selected, whether optimal arm was chosen
- Return cumulative reward, regret over time, and percentage of optimal actions

**Data to collect**:
- Total reward at each step
- Cumulative regret: optimal_reward_so_far - actual_reward_so_far
- Binary indicator: was optimal arm chosen?
- Current estimates of each algorithm

### Exercise 6: Statistical Analysis
**Objective**: Compare algorithms across multiple runs and environments

**Requirements**:
- Run each algorithm multiple times (at least 100 runs) on same environment
- Test on different environments:
  - Easy: CTRs = [0.1, 0.5, 0.2, 0.15] (clear winner)
  - Hard: CTRs = [0.12, 0.13, 0.11, 0.125] (very close)
  - Deceptive: CTRs = [0.45, 0.1, 0.1, 0.5] (good arm looks bad initially)

**Analysis to perform**:
- Average performance across runs
- Confidence intervals (mean ± 1.96 × std/√n)
- Statistical significance tests between algorithms
- Convergence analysis: how quickly do algorithms find optimal arm?

### Exercise 7: Visualization and Reporting
**Objective**: Create comprehensive performance analysis

**Requirements**:
- Plot cumulative average reward over time for each algorithm
- Plot percentage of optimal actions over time
- Plot regret over time (should decrease and flatten)
- Create summary table with final performance metrics
- Include error bars showing confidence intervals

**Visualizations needed**:
- Line plots comparing all algorithms
- Separate plots for each test environment
- Box plots showing distribution of final performance
- Heatmap showing algorithm performance across different environments

## Success Criteria

Your implementation should demonstrate:
1. **Correct algorithm implementation**: Each algorithm behaves as theoretically expected
2. **Proper experimental design**: Multiple runs, different environments, statistical rigor
3. **Clear performance differences**: UCB and Thompson Sampling should generally outperform Epsilon-Greedy
4. **Convergence**: All algorithms should eventually identify the optimal campaign
5. **Professional code quality**: Clean, documented, modular code

## Expected Insights

After completing this assignment, you should observe:
- Thompson Sampling often performs best on binary reward problems
- UCB provides good balance of exploration and exploitation
- Epsilon-Greedy is simple but can be inefficient
- Performance differences are more pronounced in "hard" environments
- All algorithms eventually converge, but at different rates

## Bonus Challenges (Optional)

1. **Decaying Epsilon**: Make ε decrease over time in Epsilon-Greedy
2. **Non-stationary Environment**: CTRs change over time
3. **Contextual Information**: Add user demographics that affect CTR
4. **Budget Constraints**: Limited number of ad impressions per day

This assignment will give you deep hands-on experience with the exploration-exploitation tradeoff that underlies all of reinforcement learning.
