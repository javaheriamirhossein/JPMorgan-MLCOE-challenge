# Discrete Choice Models for Credit Card Offers

This repository contains the code and report for the JPMorgan MLCOE Challenge
on Discrete Choice Models for Credit Card Offers. It is built on top of the
[choice-learn](https://github.com/artefactory/choice-learn) framework and
TensorFlow. It includes two packages corresponding to Part 1 and Part 2 of the
challenge, baseline comparisons, and unit tests.

---

## Repository Structure


### Part 1 (`DeepHalo`)
The files for the **DeepHalo** package are located in the `Part1/` folder.
This package implements the context-dependent discrete choice model of Zhang
et al. (2025), using a permutation-equivariant neural encoder to learn
latent product embeddings from market-level data.

### Part 2 (`BayesianSparseDeepHalo`)
The files for the **BayesianSparseDeepHalo** package are located in the
`Part2/` folder. This package implements the Bayesian Sparse Random Logit
model of Lu (2025), extended with a Monte Carlo EM loop that jointly trains
the DeepHalo encoder with the Bayesian sparse sampler.


---
## Installing the packages (Part1 and Part2)
Run these from the repository root:

```bash
# Part 1: DeepHalo
python -m pip install -e Part1

# Part 2: BayesianSparseDeepHalo
python -m pip install -e Part2

```

---
## Dependencies

- Python ≥ 3.10
- [choice-learn](https://github.com/artefactory/choice-learn) ≥ 0.2
- TensorFlow ≥ 2.12
- NumPy ≥ 1.24
- SciPy ≥ 1.10

---

## References

- Z. Lu and K. Shimizu (2025), [*Estimating Discrete Choice Demand Models with Sparse Market-Product Shock*](https://arxiv.org/abs/2501.02381v2).
- S. Zhang, Z. Wang, R. Gao, and S. Li (2025), [*Deep context-dependent choice model*](https://openreview.net/forum?id=bXTBtUjb0c).

