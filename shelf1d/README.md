# Inference of ice rheology for 1D synthetic ice shelf

This example demonstrates variational inference of ice shelf rheology for a synthetic 1D ice shelf. The ice shelf is modeled using the Shallow-Shelf Approximation (SSA) and is assumed to be laterally-confined with a rectangular cross section.

### Scripts and notebooks

All scripts and Jupyter notebooks are written in Python. 

- `run.cfg`: Configuration file for setting training and modeling options. Values set to those used in the main manuscript.
- `models.py`: Definitions for neural network and variational Gaussian Process models. Also contains implementation of prior distribution.
- `utilities.py`: Definitions for loading and normalizing data, as well as for reading configuration files.
- `pretrain_solution.py`: Script for training neural networks to reconstruct velocity and thickness. No physics information is used.
- `train.py`: Script for training variational Gaussian Process. Uses the pre-trained neural networks to predict velocity and thickness.
- `predict.py`: Script for generating predictions of velocity and thickness, as well as for generating samples from the variational distribution for ice rigidity.
- `mcmc_numpyro.ipynb`: Jupyter notebook for running MCMC for equivalent posterior inference. Uses NUTS sampler as implemented in `NumPyro`.
- `compare_samples.ipynb`: Jupyter notebook for comparing MCMC and variational inference results. Generates Figure 2 in main manuscript.


### Simulation data

The simulated velocity and thickness are stored in the HDF5 file `data_noise.h5`.

```
$ h5ls data_noise.h5

B_ref                    Dataset {400}
D                        Dataset {400, 400}
h                        Dataset {400}
h_err                    Dataset {400}
h_noisy                  Dataset {400}
h_ref                    Dataset {400}
u                        Dataset {400}
u_err                    Dataset {400}
u_noisy                  Dataset {400}
u_ref                    Dataset {400}
x                        Dataset {400}
```
The x-coordinates are stored in the dataset `x`. The true rigidity values are in `B_ref` (units of yr$^{1/4}$ kPa). The noise-free velocity and thickness are in `u_ref` and `h_ref`, respectively. Velocity with added colored noise (total standard deviation of 13 m/yr) is in `u_noisy`, and thickness with added colored noise (total standard deviation of 30 m) is in `h_noisy`. We apply a smoothing filter with a window size of 20 ice thicknesses to the noisy velocity and thickness, and the smoothed profiles are stored in `u` and `h`. The smoothed profiles are used for the variational inference.

## Example workflow

In this example, we chose to adopt a workflow where we first pre-train a neural network to reconstuct the ice surface velocity and thickness without using any physics information. We do use a smoothness cost function based on first-order gradients. To run pre-training, use:

```
./pretrain_solution.py run.cfg
```
This will generate checkpoints (saved weights) in the directory `checkpoints/checkpoints_pretrain`.

The next step is to train a variational Gaussian Process to predict a variational approximation to the posterior distribution of ice rigidity. We will use the pre-trained neural network from the previous step to predict velocity and thickness at arbitrary spatial coordinates.

```
./train.py run.cfg
```
This will generate checkpoints (saved weights) in the directory `checkpoints/checkpoints`. A log of the train and test losses will be saved to `log_train`. The 1st and 3rd columns of the log contain the train and test losses, respectively, for the negative log-likelihood of the ELBO loss. The 2nd and 4th columns of the log contain the train and test losses for the KL-divergence. Use the log to monitor training performance, and if extra training is needed, modify the `run.cfg` file in the `train` section to restore previous weights. For example:

```
[train]
lr = 0.0005
restore = True
```

After training, we can generate predictions of velocity, thickness, and rigidity using the `predict.py` script. We can compare samples of the rigidity from variational inference with MCMC-derived samples (produced with the `mcmc_numpyro.ipynb` notebook) using `compare_samples.ipynb`.