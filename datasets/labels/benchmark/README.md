# Outcome-Oriented Predictive Process Monitoring: Review and Benchmark (2017)

**Source:** https://dl.acm.org/doi/pdf/10.1145/3301300  
**Repository:** https://github.com/irhete/predictive-monitoring-benchmark

## Overview

This benchmark folder contains labeled datasets derived from the paper above, mapped to the project's core datasets for outcome-oriented predictive process monitoring tasks.

## Dataset Mapping

Most datasets align directly with their counterparts in the `labels/` directory:

- BPIC_12, BPIC_15_1-5, BPIC_17, Hospital_Billing, Sepsis, Traffic_Fines

Some benchmark variants include outcome-specific labels not present in the main dataset collection.

These outcome-specific variants can be utilized when required for specialized prediction tasks, despite not having direct counterparts in the main dataset collection.

Important consideration: The benchmark datasets may have undergone specific preprocessing steps (e.g., event filtering, handling of missing values, encoding of categorical variables) that differ from the raw datasets.
