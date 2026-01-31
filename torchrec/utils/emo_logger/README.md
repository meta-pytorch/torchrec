# EMO Decision Logger

Lightweight logger for tracking EMO decisions in TorchRec.

## Usage

```python
import logging
from torchrec.utils.emo_logger import DecisionCategory, log_emo_decision

log_emo_decision(DecisionCategory.CLF, logging.DEBUG, "calculated load_factor=0.05", table_name="user_embedding")
log_emo_decision(DecisionCategory.KERNEL, logging.INFO, "promoted to HBM", table_name="item_embedding", clf=0.97)
```

## Decision Categories

| Category | Description |
| -------- | ----------- |
| `CLF` | Cache Load Factor calculations |
| `KERNEL` | Kernel selection decisions |
| `PROPOSER` | Proposer decisions |

## Integration Points

| Component          | File                   | Integration Point            |
| ------------------ | ---------------------- | ---------------------------- |
| Constraint builder | `sparsenn_configs.py`  | `update_cache_params_helper` |
| Stats matching     | `embedding_stats.py`   | `EmbeddingStatsAccessor`     |
| Proposers          | `proposers.py`         | `EmbeddingOffloadCacheScalingProposer` |
