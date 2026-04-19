# Temporal Efficiency in Hybrid ANN–SNN Architectures

This project compares three neural network architectures on MNIST to test whether spike-based computation can match standard deep learning accuracy at lower compute cost — and whether a hybrid design outperforms either pure system.

- **ANN** — standard CNN baseline
- **SNN** — same architecture with Leaky Integrate-and-Fire neurons (snnTorch)
- **Hybrid** — ANN convolutional feature extractor + spiking classifier head

## Results (MNIST)

| Model  | Accuracy | Train time | Params | Spike events |
|--------|----------|-----------|--------|--------------|
| ANN    | 99.14%   | 155s      | 1.20M  | —            |
| SNN    | 99.10%   | 842s      | 1.20M  | 178.7M       |
| Hybrid | 99.14%   | 263s      | 1.20M  | 3.44M        |

**Key finding:** The hybrid matches ANN accuracy exactly while using 52x fewer spike events than the pure SNN. This suggests that most of the SNN's computational overhead comes from spiking in the convolutional layers, not the classifier. Using ANN layers for spatial feature extraction and spiking layers only for classification retains the efficiency benefit without sacrificing accuracy.

Training config (identical across all models): Adam (lr=1e-3) + StepLR (step=5, γ=0.5), batch size 128, 15 epochs, cross-entropy loss. Trained on NVIDIA Tesla T4 via Google Colab.

## Stack

Python, PyTorch, snnTorch. All notebooks run on Google Colab with no local setup.

## Status

MNIST results complete. Next: Fashion-MNIST, unified energy proxy (FLOPs-equivalent for spikes), and noise robustness experiments.

## Author

Harini Anandkumar, 2025–26
