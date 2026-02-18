# Predictive Process Monitoring Benchmarks

## Benchmark 1: Teinemaa et al. (2017)

**Paper**: ["Outcome-Oriented Predictive Process Monitoring: Review and Benchmark"](https://arxiv.org/abs/1707.06766)  
**Repository**: https://github.com/irhete/predictive-monitoring-benchmark  

### Datasets Used

The Teinemaa benchmark provides labeled datasets for:

- BPIC 2011, 2012, 2015, 2017
- Sepsis Cases
- Hospital Billing
- Road Traffic Fine Management
- Production log

### Usage in This Project

Teinemaa labels are stored in `datasets/labels/benchmarks/Teinemaa_2017/` and serve as baseline comparisons for outcome prediction tasks. These labels follow a straightforward case-level outcome approach without temporal considerations.

---

## Benchmark 2: Weytjens & vanden Broucke (2021)

**Paper**: ["Creating Unbiased Public Benchmark Datasets with Data Leakage Prevention for Predictive Process Monitoring"](https://arxiv.org/abs/2107.01905)  
**Repository**: https://github.com/hansweytjens/predictive-process-monitoring-benchmarks

### Datasets Used

Weytjens provides scripts for:

- **BPIC 2012** (variants A, O, W): Classification targets (approved, declined, canceled)
- **BPIC 2015** (5 municipalities): Remaining time prediction
- **BPIC 2017**: Classification targets (approved, declined, canceled)
- **BPIC 2019**: Purchase order handling
- **BPIC 2020** (Payments, Permits, Travel Costs): Remaining time prediction

### Using Different Label Sets

When training models, you can specify which label file to use:

```python
from utils.log_datasets import OutcomeDataset

# Use primary labels (from our labeling.py)
dataset = OutcomeDataset(
    dataset_name='BPIC_12_O',
    dataset_folder='datasets/raw',
    labels_folder='datasets/labels',
    label_filename='BPIC_12_O'  # Matches dataset name by default
)

# Use Weytjens benchmark labels
dataset = OutcomeDataset(
    dataset_name='BPIC_12_O',
    dataset_folder='datasets/raw',
    labels_folder='datasets/labels/benchmarks/Weytjens_2021/labels',
    label_filename='BPIC_12_O'  # Or a different filename if exists
)

# Use binary classification variant
dataset = OutcomeDataset(
    dataset_name='BPIC_12_O',
    dataset_folder='datasets/raw',
    labels_folder='datasets/labels/binary',
    label_filename='BPIC_12_O'
)
```
