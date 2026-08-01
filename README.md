# A Unified Framework for Scalable and Robust Paper Assignment

Official implementation of
[A Unified Framework for Scalable and Robust Paper Assignment](https://arxiv.org/abs/2601.14402)
by Michael Cui, Chenxin Dai, Yixuan Even Xu, and Fei Fang.

The repository contains the fractional assignment algorithms described in the
paper and a dependent-rounding sampler that converts a fractional solution into
an integral paper-reviewer matching. The implementation supports RAMP,
Perturbed Maximization, randomized assignment, and an unperturbed linear
assignment baseline.

## Reproduce the sample run

### Requirements

- Linux or macOS
- Python 3
- [Gurobi](https://www.gurobi.com/) with a valid license
- GNU Make and a C++17 compiler

The artifact was verified on Ubuntu x86-64 with Python 3.14.4, Gurobi 13.0.2,
GNU Make 4.4.1, and g++ 15.2.0. Other recent Python 3 and C++17 environments
should also work.

Create a Python environment and compile the sampler:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
make
```

Run the included synthetic instance:

```bash
python3 runner.py --config configs/sample_config.yaml
```

The sample has 20 papers, 30 reviewers, and 200 candidate edges. A successful
run writes a 60-assignment matching to
`results/sample/ramp/matching_files/matching_pairs.csv`. Sampling is randomized,
so the selected pairs may differ between runs.

The sample is a small executable example, not the dataset used for every table
in the paper. To run another instance, prepare the five CSV files described
below and update their paths in the YAML configuration.

## Sampling modes

There is one sampler implementation and one executable, `build/bin/bvn`.
`optimize_sampling` controls only how the fractional solution is rounded; it
does not change the fractional optimization problem.

```yaml
optimize_sampling: false  # standard Birkhoff-von Neumann traversal
optimize_sampling: true   # attribute-aware traversal from the paper
```

The optimized traversal prioritizes edges using reviewer regions and coauthor
relations. The option is independent of `algo_name`, so either sampling mode
can be used with any fractional assignment algorithm.

## Configuration

`runner.py` accepts a YAML file through `--config`. The included configuration
documents a complete RAMP run.

| Field | Description |
| --- | --- |
| `dataset_dir` | Directory containing the five input CSV files with their standard names |
| `output_dir` | Directory in which results are written |
| `algo_name` | `RAMP`, `Perturbed_Maximization`, `Randomized`, or `Default` |
| `2stage` | Whether to assign senior reviewers in a separate first stage |
| `optimize_sampling` | Whether to enable attribute-aware sampling; default `false` |
| `maxprob` | Optional assignment-probability cap in `(0, 1]` |
| `dynamic_maxprob` | Optional probability cap for bid edges; non-bid edges use `1.0` |
| `beta` | Perturbation strength for RAMP and Perturbed Maximization |
| `reward_region` | RAMP reward for region diversity |
| `pen_coauthor` | RAMP coauthor penalty |
| `pen_2cycle` | RAMP two-cycle penalty |

With `2stage: true`, stage 1 assigns the requested number of senior reviewers
using only the senior pool, and stage 2 fills the remaining demand using the
full reviewer pool. With `2stage: false`, all reviewers are assigned in one
optimization. A zero `reward_region`, `pen_coauthor`, or `pen_2cycle` disables
that RAMP component completely, so its auxiliary variables and constraints are
not built.

## Input format

Every CSV file must include exactly the header shown below. Paper and reviewer
IDs are arbitrary non-empty strings; the loader maps them to compact internal
indices and restores the original IDs in all outputs.

The directory configured by `dataset_dir` must contain `paper_info.csv`,
`reviewer_info.csv`, `similarity_scores.csv`, `bid_scores.csv`, and
`constraints.csv`.

`paper_info.csv`:

```text
paper_id,senior_reviewers_needed,reviewers_needed
```

`reviewer_info.csv`:

```text
reviewer_id,max_load,min_load,seniority,region,authored_paper_ids,coauthor_reviewer_ids
```

The last two columns contain JSON arrays of external IDs. `seniority` and
`region` are non-negative integers; a positive seniority marks a senior
reviewer.

`similarity_scores.csv` and `bid_scores.csv`:

```text
paper_id,reviewer_id,score
```

`constraints.csv`:

```text
paper_id,reviewer_id,constraint
```

A constraint of `-1` forbids an edge. A constraint of `1` records a previously
assigned edge, which must also occur in `similarity_scores.csv`. Only similarity
edges are solver candidates; bid and constraint rows add metadata but do not
create candidate edges. Similarity rows with a score less than or equal to zero
are discarded while loading and do not create solver variables. Memory for
candidate adjacency is therefore proportional to the number of positive
similarity edges rather than to the full paper-reviewer Cartesian product.

## Outputs

The YAML `output_dir` is the complete experiment output path; the runner does
not derive directory names from `algo_name`. For a one-stage run,
`output_dir/matching_files/` contains:

- `prob_assignment.csv`: nonzero fractional assignment probabilities
- `matching_pairs.csv`: sampled integral assignments
- `matching.log`: run metadata and timing

For a two-stage run, the CSV names are prefixed with `stage1_` and `stage2_`.
The base output directory contains `statistics.txt` and the diagnostic plots.
Use `--log-level DEBUG`, `INFO`, `WARNING`, or `ERROR` to select console
verbosity.

## Repository layout

```text
configs/       Example experiment configuration
cpp/bvn.cpp    Dependent-rounding sampler and optional sampling heuristic
datasets/      Included synthetic sample
python/        Instance loader, optimization algorithms, and analysis
runner.py      Experiment entry point
Makefile       C++ build targets
```

The sampler binary is generated under `build/bin/` and is not committed. Use
`make rebuild` for a clean compilation or `make clean` to remove build
artifacts.

## Citation

```bibtex
@misc{cui2026unified,
  title         = {A Unified Framework for Scalable and Robust Paper Assignment},
  author        = {Michael Cui and Chenxin Dai and Yixuan Even Xu and Fei Fang},
  year          = {2026},
  eprint        = {2601.14402},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SI}
}
```
