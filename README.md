# HACO: Hedged Agent Computing for Reliable LLM Systems

**NeurIPS 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2607.19215-b31b1b.svg)](https://arxiv.org/abs/2607.19215)
[![Conference](https://img.shields.io/badge/NeurIPS-2026-blue.svg)](https://openreview.net/forum?id=zNkQ67Ps8c)

Official implementation of **HACO: Hedged Agent Computing for Reliable LLM Systems**.

[Enhan Li](https://openreview.net/profile?id=~Enhan_Li1), [Hongyang Du](https://hongyangdu.github.io/)  
The University of Hong Kong

**[Paper](https://arxiv.org/abs/2607.19215) · [OpenReview](https://openreview.net/forum?id=zNkQ67Ps8c)**

## Overview

HACO is a runtime control framework for reliable LLM agent systems. For each role invocation, it selects a small set of candidate agent instances, runs them in parallel, and learns from their execution traces. Each candidate combines a role, an LLM, and an execution environment, allowing allocation to adapt to changing service and network conditions.

The framework has four stages:

1. **Role abstraction** identifies the next role request from the task and workflow context.
2. **Redundancy allocation** combines optimistic candidate ranking with conservative reliability accumulation to select a hedge set.
3. **Hedged execution** executes the selected candidates in parallel and selects an output.
4. **Experience harvesting** updates candidate and network profiles using quality, success, latency, and network observations from executed candidates.

The paper evaluates HACO on MatplotBench, InfiAgent-Bench, and DSBench, including runtime degradation and Azure deployment experiments.

## Repository Structure

```text
.
├── configs/                      # HACO, baseline, and agent-pool configurations
├── datawiseagent/
│   ├── agents/os/
│   │   ├── process_management/    # HACO, baselines, and ablation variants
│   │   ├── experience/            # Experience collection and profile updates
│   │   └── network/               # Network models and latency measurement
│   ├── coding/                    # Code execution and Jupyter sandbox
│   ├── llms/                      # LLM interfaces
│   └── tools/                     # Benchmark tools
├── evaluation/                    # Benchmark runners and scoring scripts
├── scripts/                       # Experience summary utilities
├── .env.example                   # Environment configuration template
└── main.py                        # Backend entry point
```

The main allocation algorithm is in [`haco.py`](datawiseagent/agents/os/process_management/haco.py); experience collection is implemented in [`experience_collecting.py`](datawiseagent/agents/os/experience/experience_collecting.py).

## Setup

### Python Environment

Use Python 3.10 or later:

```bash
git clone https://github.com/NICE-HKU/haco.git
cd haco
conda create -n haco python=3.10 -y
conda activate haco
```

The runtime uses FastAPI, OpenAI's Python SDK, Pydantic, PyYAML, Docker, Jupyter, and data-processing libraries. Install the dependencies required by the modules you use. A version-pinned dependency specification is not currently included.

### API Configuration

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` in your local `.env`. The default configuration entry is:

```dotenv
CUSTOM_CONFIG=configs/default_config_haco.yaml
```

Keep credentials in local files or environment variables. `.env` and local credential files are excluded from version control.

For Azure experiments, configure your own resource endpoints, keys, API versions, and deployment names in the agent-pool configuration. The `agent_real*.yaml` files contain blank keys and example endpoints that must be replaced locally.

### Docker Sandbox

Build the CPU sandbox from its Docker build context:

```bash
docker build -t my-jupyter-image \
  -f datawiseagent/coding/jupyter/default_jupyter_server.dockerfile \
  datawiseagent/coding/jupyter
```

A CUDA sandbox definition is also provided in `datawiseagent/coding/jupyter/cuda_jupyter_server.dockerfile`.

Set `code_executor.use_docker`, `code_executor.use_gpu`, and `code_executor.image_name` in your configuration to match your environment. For the CPU image, set `use_gpu: false`.

## Running HACO

Start the backend from the repository root:

```bash
python main.py --port 8008
```

The runner and backend should use the same `server.port`. The released code provides the backend and evaluation scripts; a separate frontend is not included.

The main settings are in [`configs/default_config_haco.yaml`](configs/default_config_haco.yaml):

| Setting | Purpose |
| --- | --- |
| `agent.selection_strategy` | Select HACO or a baseline |
| `agent.agent_config_path` | Select the role-specific candidate pool |
| `agent.haco.reliability_target` | Set the target reliability |
| `agent.haco.exploration_factor` | Control uncertainty-aware exploration |
| `agent.haco.risk_aversion_factor` | Control conservative candidate estimates |
| `agent.haco.jitter_penalty` | Adjust sensitivity to network variability |
| `agent.haco.latency_discount_alpha` | Adjust latency discounting |

These parameters define an experimental setting; use the settings in the paper for the corresponding comparison.

## Evaluation

Prepare the benchmark data separately in the paths expected by each runner. In particular, the runner scripts expect data under `evaluation/InfiAgentBench/data/`, `evaluation/MatplotBench/data/`, and `evaluation/DataModeling/data/`; these directories are not bundled with the release.

With the backend running, load your environment variables in a second terminal and run:

```bash
conda activate haco
set -a
source .env
set +a
cd evaluation

# InfiAgent-Bench
python eval_infiagent_bench.py --note haco_infiagent --start_from 1

# MatplotBench
python eval_matplotbench.py --note haco_matplot --start_from 1

# DSBench
python eval_dsbench.py --user_name haco_dsbench \
  --result_path ./results/DSBench/gpt-4o-mini --agent-type mas --start_from 1
```

For MatplotBench's visual tool, add `--with_tool`; its provider credentials are read from `OPENAI_API_KEY` and `OPENAI_BASE_URL`.

Scoring scripts are provided in `evaluation/InfiAgentBench/scripts/`, `evaluation/MatplotBench/scripts/`, and `evaluation/DataModeling/scripts/`. The MatplotBench judge and InfiAgent-Bench reformatter also accept provider credentials through the environment.

### Baselines and Ablations

Switch `CUSTOM_CONFIG` to one of the following configurations and restart the backend:

| Method | Configuration |
| --- | --- |
| HACO | `configs/default_config_haco.yaml` |
| BestOne | `configs/default_config_bestone.yaml` |
| MoA | `configs/default_config_moa.yaml` |
| Random | `configs/default_config_random.yaml` |

Additional agent-pool configurations cover network degradation, candidate degradation, evaluator degradation, and changes to role-specific candidate pools. Ablation implementations are included alongside HACO in `datawiseagent/agents/os/process_management/`.

## Citation

If you use HACO in your research, please cite:

```bibtex
@inproceedings{li2026haco,
  title     = {HACO: Hedged Agent Computing for Reliable LLM Systems},
  author    = {Li, Enhan and Du, Hongyang},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2607.19215}
}
```
