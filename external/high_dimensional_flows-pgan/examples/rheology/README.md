# Example: inversion of ice shelf rheology in 1D

The goal is to estimate the spatial field of ice rheology over a synthetic, 1D ice-shelf. We will use two classes of variational approximations: 1) mean field; and 2) mean field + inverse autoregressive network.

## Scripts

For training the mean field surrogate, we use the script `mean_field_rheology.py`. After training, we generate samples from the surrogate posterior using the script `generate_samples.py` and analyze statistics of the samples using `analyze_samples.py`.
