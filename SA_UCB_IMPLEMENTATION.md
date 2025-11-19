# SA-UCB Solver Implementation

## Overview

This implementation adds a new solver called **SA-UCB (Semantic Aware UCB)** to the AIRA-Dojo framework. SA-UCB extends the standard MCTS solver with semantic awareness by incorporating two specialized LLM scorers that evaluate nodes based on:

1. **Promise Score (P)**: How promising/effective a solution approach appears
2. **Novelty Score (U)**: How novel/diverse a solution is compared to its siblings

## Mathematical Formulation

The SA-UCB formula is:

```
SA-UCB(v) = Q_v/N_v + C·√(ln(N_parent)/N_v) + λ·P_v + γ·U_v
            └────────────────────┬────────────────────┘   └─────┬─────┘
                    Standard UCB                        Semantic Bonus
```

Where:
- `Q_v/N_v`: Exploitation term (average reward)
- `C·√(ln(N_parent)/N_v)`: Exploration term (UCB exploration)
- `λ·P_v`: Promise bonus (weighted promise score)
- `γ·U_v`: Novelty bonus (weighted novelty score)

## Implementation Files

### Core Solver
- **`src/dojo/solvers/mcts/sa_ucb.py`**: Main SA-UCB solver implementation
  - `SA_UCBNode`: Extended node class with promise/novelty scores
  - `SA_UCB`: Main solver class extending MCTS
  - `sa_uct_value()`: SA-UCB value calculation function

### Configuration
- **`src/dojo/config_dataclasses/solver/sa_ucb.py`**: Config dataclass with λ and γ hyperparameters
- **`src/dojo/configs/solver/sa_ucb.yaml`**: Base solver config
- **`src/dojo/configs/solver/mlebench/sa_ucb.yaml`**: MLEBench-specific config

### Operators
- **`src/dojo/configs/solver/operators/mlebench/aira_operators/promise_scorer.yaml`**: Promise scorer LLM config
- **`src/dojo/configs/solver/operators/mlebench/aira_operators/novelty_scorer.yaml`**: Novelty scorer LLM config

### Experiment Config
- **`src/dojo/configs/_exp/run_three_tasks_sa_ucb.yaml`**: Example experiment config for testing

## Key Features

### 1. Semantic Scoring LLMs

**Promise Scorer**:
- Evaluates how promising a solution approach appears
- Considers: method appropriateness, theoretical foundation, evaluation strategy, data handling
- Returns: score ∈ [0, 1] + reasoning

**Novelty Scorer**:
- Evaluates how novel/diverse a solution is vs siblings
- Considers: different methods, unique features, complementary strategies
- Returns: score ∈ [0, 1] + reasoning

### 2. Comprehensive Logging

All scorer outputs are logged to `{checkpoint_path}/scorer_logs/`:
- `promise_scores.jsonl`: Promise scorer logs
- `novelty_scores.jsonl`: Novelty scorer logs

Each log entry includes:
- Timestamp
- Node step
- Score and reasoning
- System prompt used
- User prompt
- LLM completion
- Usage statistics

### 3. Node-Level Metrics

Each `SA_UCBNode` tracks:
- `promise_score`: Numeric promise score
- `novelty_score`: Numeric novelty score
- `promise_reasoning`: LLM's explanation for promise score
- `novelty_reasoning`: LLM's explanation for novelty score

## Hyperparameters

### Default Values (in `mlebench/sa_ucb.yaml`)

- `lambda_promise: 0.3` - Weight for promise bonus
- `gamma_novelty: 0.2` - Weight for novelty bonus
- `uct_c: 0.25` - Standard UCB exploration constant

### Recommended Ranges

- `lambda_promise`: 0.1 - 0.5 (higher values favor promising approaches)
- `gamma_novelty`: 0.1 - 0.5 (higher values favor diverse exploration)

### Tuning Guidelines

- **Increase λ** if you want to focus more on exploiting good-looking approaches
- **Increase γ** if you want to explore more diverse solutions
- **Balance both** for a mix of exploitation (promise) and exploration (novelty)

## Usage

### Running with SA-UCB

```bash
# Use the provided experiment config
python run_three_tasks.py --config-name run_three_tasks_sa_ucb

# Or override the solver in an existing config
python run_three_tasks.py solver=mlebench/sa_ucb
```

### Customizing Hyperparameters

In your experiment config or via command line:

```yaml
solver:
  lambda_promise: 0.4  # Adjust promise weight
  gamma_novelty: 0.3   # Adjust novelty weight
```

Or via CLI:
```bash
python run_three_tasks.py solver=mlebench/sa_ucb solver.lambda_promise=0.4 solver.gamma_novelty=0.3
```

## Implementation Details

### Scoring Workflow

1. When a new node is created (draft/improve/debug):
   - Node is created with code and plan
   - Promise scorer evaluates the approach quality
   - Novelty scorer compares against siblings
   - Scores are stored in the node

2. During search policy (node selection):
   - SA-UCB value is calculated for each child
   - Node with highest SA-UCB value is selected
   - Path continues until a leaf is reached

### Error Handling

- If scorer LLMs fail, default score of 0.5 is used
- Scores are clamped to [0, 1] range
- Errors are logged but don't stop execution

### Performance Considerations

- Scorers use `temperature=0.3` for consistent scoring
- Code is truncated to 2000 chars to avoid context limits
- Only first 5 siblings are compared for novelty (to avoid context overflow)

## Logging and Analysis

### Scorer Logs Location

```
{checkpoint_path}/scorer_logs/
├── promise_scores.jsonl    # All promise scorer calls
└── novelty_scores.jsonl    # All novelty scorer calls
```

### Log Entry Format

```json
{
  "timestamp": 1234567890.123,
  "node_step": 5,
  "scorer_type": "promise",
  "score": 0.75,
  "reasoning": "This approach uses XGBoost which is well-suited for tabular data...",
  "system_prompt": "You are an expert ML engineer...",
  "user_prompt": "# TASK DESCRIPTION...",
  "completion": "{\"promise_score\": 0.75, \"reasoning\": \"...\"}",
  "usage": {"prompt_tokens": 1234, "completion_tokens": 56, ...}
}
```

### Main Logger Integration

Scores are also logged to the main logger with tags:
- `SCORER_PROMISE`: Promise scorer outputs
- `SCORER_NOVELTY`: Novelty scorer outputs

## Differences from Standard MCTS

| Aspect | MCTS | SA-UCB |
|--------|------|--------|
| Node Class | `MCTSNode` | `SA_UCBNode` (extends MCTSNode) |
| Selection Formula | Standard UCB | SA-UCB (UCB + semantic bonuses) |
| Operators | draft, improve, debug, analyze | + promise_scorer, novelty_scorer |
| Hyperparameters | `uct_c` | `uct_c`, `lambda_promise`, `gamma_novelty` |
| Logging | Standard metrics | + scorer logs in `scorer_logs/` |

## Testing

### Quick Test

```bash
# Run with step_limit=2 for quick testing
python run_three_tasks.py --config-name run_three_tasks_sa_ucb solver.step_limit=2
```

### Full Experiment

```bash
# Use default step_limit=2500
python run_three_tasks.py --config-name run_three_tasks_sa_ucb
```

### Analyzing Results

Check the scorer logs to see how nodes were evaluated:

```bash
# View promise scores
cat outputs/{experiment_dir}/scorer_logs/promise_scores.jsonl | jq .

# View novelty scores
cat outputs/{experiment_dir}/scorer_logs/novelty_scores.jsonl | jq .
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure `SA_UCB` is registered in:
   - `src/dojo/solvers/mcts/__init__.py`
   - `src/dojo/config_dataclasses/solver/__init__.py`

2. **Config not found**: Check that YAML files are in correct locations:
   - `configs/solver/sa_ucb.yaml`
   - `configs/solver/mlebench/sa_ucb.yaml`
   - `configs/solver/operators/mlebench/aira_operators/promise_scorer.yaml`
   - `configs/solver/operators/mlebench/aira_operators/novelty_scorer.yaml`

3. **Scorer LLM failures**: Check that LLM clients are configured:
   - `solver.operators.promise_scorer.llm.client`
   - `solver.operators.novelty_scorer.llm.client`

## Future Improvements

Potential enhancements to consider:

1. **Adaptive Weights**: Dynamically adjust λ and γ based on search progress
2. **Multi-Objective Scoring**: Combine multiple scoring dimensions
3. **Hierarchical Novelty**: Score novelty at different tree levels
4. **Score Caching**: Cache scores for similar nodes to reduce LLM calls
5. **Score Calibration**: Calibrate scores across different tasks/domains

## References

- Base MCTS implementation: `src/dojo/solvers/mcts/mcts.py`
- UCB formula: `uct_value()` in `mcts.py:51-66`
- Node structure: `src/dojo/core/solvers/utils/journal.py`
