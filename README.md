# scGPT SAE Refinement

**Interpreting and Refining scGPT Cell Embeddings with Structure-Preserving Sparse Autoencoders**

Authors: Xia Haoxiang, Chang Runjia, Xiao Jiahao.

This BS6204 course project studies sparse autoencoders (SAEs) for interpreting and refining frozen scGPT cell embeddings.

The project examines:

- Cell-type associations and gene-input substitution attribution of sparse features.
- SAE reconstruction with MSE and KL alignment of cell–cell similarity distributions.
- Classification, clustering, and sensitivity to the relational regularization weight.
- Expression prediction and masked-input robustness using a separate expression-supervised SAE extension.

## Repository status

This repository has been initialized for the forthcoming code release. Training and evaluation code, reproducible figure scripts, and setup instructions will be organized here. The current commit contains project documentation and ignore rules only; it is not a runnable implementation yet.

Raw expression data, pretrained scGPT weights, SAE checkpoints, credentials, and local machine-specific files are not included.

The study reports task- and dataset-dependent results; a separate model branch is used for the expression-supervised experiments.
