# FILE: clip-sae-interpret/clip-sae-interpret_clean/README.md

# CLIP + SAE Interpretability Project

Sparse Autoencoder (SAE) trained on CLIP (ViT-L/14) embeddings to extract interpretable latent features and steer Kandinsky 2.2 text-to-image model.

## Features
- Training SAE on CLIP embeddings
- Zero-shot degradation analysis
- Feature interpretability via VLM
- Steering text-to-image generation (+/- feature)
- Ready for Colab demo and HF publication

## Install

```bash
pip install -r requirements.txt
pip install -e .
