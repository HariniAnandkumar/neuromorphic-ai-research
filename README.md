# Compute Accounting in Hybrid ANN-SNN Architectures on Fashion-MNIST

Code and results for the paper of the same name. We compare three matched neural
networks on Fashion-MNIST to show that spike-count (SynOps) numbers overstate the
efficiency of hybrid ANN-SNN models, because they hide the ordinary computation the
hybrid still does in its non-spiking layers.

- **ANN** — standard convolutional network
- **SNN** — same shape, every ReLU replaced by a Leaky Integrate-and-Fire neuron (snnTorch)
- **Hybrid** — ANN convolutional front-end feeding a small spiking classifier head

All three share identical layer shapes and exactly 1,199,882 parameters, and are
trained the same way.

## Accuracy (Fashion-MNIST, 5 seeds)

| Model  | Accuracy       |
| ------ | -------------- |
| ANN    | 92.82 ± 0.25%  |
| Hybrid | 92.66 ± 0.25%  |
| SNN    | 92.00 ± 0.16%  |

## Per-image compute

| Model  | MACs   | SynOps | Spike rate |
| ------ | ------ | ------ | ---------- |
| ANN    | 11.99M | —      | —          |
| SNN    | —      | 13.5M  | 4.4%       |
| Hybrid | 11.99M | 2,602  | 19.6%      |

**Key point:** The hybrid's spiking classifier fires only 2,602 SynOps per image,
against 13.5M for the pure SNN. Reported on its own, that looks like a ~5,000x
efficiency win. It is not. The hybrid still runs ~12M multiply-add operations (MACs)
per image in its ordinary layers, and a spike count never shows them. MACs and SynOps
are different kinds of work and should be reported separately, not merged into one
efficiency number without a hardware energy model. The reporting recommendation is the
contribution.

**Robustness:** Under input damage (Gaussian noise, salt-and-pepper, occlusion), the
pure SNN is the most robust to pixel-level noise by a wide margin, while the hybrid
sits between the ANN and the SNN. The hybrid does not inherit the best of both parts.

## Training config

Identical across all models: Adam (lr=1e-3) + StepLR (step=5, γ=0.5), batch size 128,
15 epochs, cross-entropy loss, 10 timesteps for the spiking models. Trained on an
NVIDIA Tesla T4 via Google Colab.

## Notebooks

- `01_ann_fmnist.ipynb`, `02_snn_fmnist.ipynb`, `03_hybrid_fmnist.ipynb` — the three models.
- `04_robustness_fmnist.ipynb` — the 5-seed accuracy and full noise-robustness run;
  produces `robustness_results.csv` (240 rows: 5 seeds × 3 models × 16 conditions).
- `Graph.ipynb` — figures.
- `neuron_simulation.ipynb` — LIF neuron demo.

## Data

`robustness_results.csv` — per-seed, per-condition test accuracy behind Table 5 and
Figure 2 of the paper.

## Stack

Python, PyTorch, snnTorch. All notebooks run on Google Colab with no local setup.

## Author

Harini Anandkumar, 2025–26
