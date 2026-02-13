# Outcome-Oriented Predictive Process Monitoring: Review and Benchmark (2017)

**Source:** https://dl.acm.org/doi/pdf/10.1145/3301300  
**Repository:** https://github.com/irhete/predictive-monitoring-benchmark

## Overview

This benchmark folder contains labeled datasets derived from the paper above, mapped to the project's core datasets for outcome-oriented predictive process monitoring tasks.

## Dataset Mapping

Most datasets align directly with their counterparts in the `labels/` directory:

- BPIC_12, BPIC_15_1-5, BPIC_17, Hospital_Billing, Sepsis, Traffic_Fines

Some benchmark variants include outcome-specific labels not present in the main dataset collection:

- `BPIC_12_O_CANCELLED.csv` - Cancelled orders
- `BPIC_12_O_DECLINED.csv` - Declined orders
- `BPIC_17_O_Cancelled.csv` - Cancelled applications
- `BPIC_17_O_Refused.csv` - Refused applications
- `Hospital_Billing_2.csv`, `Sepsis_2.csv`, `Sepsis_3.csv` - Alternative label configurations

These outcome-specific variants can be utilized when required for specialized prediction tasks, despite not having direct counterparts in the main dataset collection.
