# Qonductor

## Project Description

Qonductor is a cloud orchestrator for hybrid quantum-classical applications running on heterogeneous, hybrid resources. It abstracts away the complexity of hybrid programming and resource management through a hardware-agnostic Qonductor API. Qonductor will appear in The International Conference for High Performance Computing, Networking, Storage, and Analysis (SC) ‘25, at St. Louis, MO, USA.

## Key design goal:

- Scalable & Load-Balancing – Achieve improved job completion times (JCTs), increased quantum resource utilization, and maintain performance under growing system size and workload.

## Key components:

1. Resource Estimator – Predicts fidelity and execution times to systematically generate resource plans that use hybrid quantum-classical resources.

2. Hybrid Scheduler – Automates job scheduling across hybrid resources, trading off between Quality of Service (QoS) objectives and cloud operator resource efficiency.


### Performance Highlights (from our evaluation with >7000 real IBM Quantum runs):

1. Up to 54% lower JCTs with only 3% execution quality loss.

2. Balanced QPU utilization, improving usage by up to 66%.

3. Scales to larger system sizes and workloads without bottlenecks.

## Project Structure

### `data/`:  Datasets, benchmarks, trained models, results
### `src/`: Source code 
- `analysis/`: Scripts to reproduce paper figures
    - `ibm_status_analysis.py`: C1 (Figure 2) – IBM QPU status analysis
    - `e2e_performance.py`: C2 (Figure 6) – End-to-end performance plots
    - `estimator_analysis.py`: C3 (Figure 7) – Resource estimator evaluation
    - `scheduler_analysis.py`: C4 (Figure 10) – Scheduler performance
    - `scheduling_manager_analysis.py`: C4, C5 (Figures 8, 9) – More scheduler performance and scalability
- `scheduler`: Scheduling logic and hybrid orchestration
- `optimization`: Pareto-optimal optimization code for scheduling
- `execution_time`: Execution time prediction models
- `scheduling_manager`: Scheduling simulation framework
- `utils`: Helper functions and common utilities
### `install.sh`: Installation script for dependencies
### `requirements.txt`: Python dependencies

## **Installation**
```bash
# Clone repository
git clone https://github.com/manosgior/Qonductor-SC25.git
cd Qonductor-SC25

# Install dependencies
bash install.sh

# Set Python path
export PYTHONPATH=.
```

## Examples
### Reproduce Figure 6 – E2E performance
```bash
python src/analysis/e2e_performance.py
```

### Run resource estimator analysis (Figure 7)
```bash
python src/analysis/estimator_analysis.py
```
