# Inference of ice rheology for 2D synthetic ice shelf

This example demonstrates variational inference of ice shelf rheology for a synthetic 2D ice shelf. The ice shelf is modeled using the Shallow-Shelf Approximation (SSA) with the `icepack` modeling package (https://icepack.github.io). The ice shelf is closely related to the [ice shelf tutorial](https://icepack.github.io/notebooks/tutorials/02-synthetic-ice-shelf/) with modeling of damage. We have added a few random pinning points throughout the shelf in order to simulate localized grounding.

### Scripts and notebooks

All scripts and Jupyter notebooks are written in Python. 

- `run.cfg`: Configuration file for setting training and modeling options. Values set to those used in the main manuscript.
- `models.py`: Definitions for neural network and variational Gaussian Process models. Also contains implementation of prior distribution.
- `utilities.py`: Definitions for loading and normalizing data, as well as for reading configuration files.
- `pretrain_solution.py`: Script for training neural networks to reconstruct velocity and thickness. No physics information is used.
- `train.py`: Script for training variational Gaussian Process. Uses the pre-trained neural networks to predict velocity and thickness.
- `predict_joint.ipynb`: Jupyter notebook for generating predictions of velocity and thickness, as well as for generating mean and standard deviation for rheology.
- `generate_covariance.ipynb`: Jupyter notebook for generating shelf-wide samples of rheology.

### Simulation data

The original simulated velocity and thickness are stored as 2D datasets in the file `data_grids.h5`.

```
$ h5ls data_grids.h5

B                        Dataset {400, 800}
B_ref                    Dataset {400, 800}
H                        Dataset {400, 800}
S                        Dataset {400, 800}
U                        Dataset {400, 800}
V                        Dataset {400, 800}
X                        Dataset {400, 800}
Y                        Dataset {400, 800}
mask                     Dataset {400, 800}
```
Note the rigidity dataset `B` is the initial rigidity estimated using a conventional inversion in `icepack`. The dataset `B_ref` is the true rigidity field.

Randomly sampled data points used for training are stored in `data.h5`.

```
$ h5ls data.h5

b                        Dataset {20000, 1}
h                        Dataset {20000, 1}
h_err                    Dataset {20000, 1}
indices                  Dataset {20000}
s                        Dataset {20000, 1}
u                        Dataset {20000, 1}
u_err                    Dataset {20000, 1}
v                        Dataset {20000, 1}
v_err                    Dataset {20000, 1}
x                        Dataset {20000, 1}
xp                       Dataset {20000, 1}
y                        Dataset {20000, 1}
yp                       Dataset {20000, 1}
```
The datasets `xp` and `yp` are uniform random coordinates throughout the ice shelf used for evaluting the ELBO loss for the variational Gaussian Process.

## Example workflow

In this example, we chose to adopt a workflow where we first pre-train a neural network to reconstuct the ice surface velocity and thickness without using any physics information. To run pre-training, use:

```
./pretrain_solution.py run.cfg
```
This will generate checkpoints (saved weights) in the directory `checkpoints/checkpoints_pretrain`.

The next step is to jointly train a variational Gaussian Process and the previous neural network to predict a variational approximation to the posterior distribution of ice rigidity. We will use the pre-trained neural network weights from the previous step.

```
./train.py run.cfg
```
This will generate checkpoints (saved weights) in the directory `checkpoints/checkpoints`. A log of the train and test losses will be saved to `log_train`. The 1st and 4th columns of the log contain the train and test losses, respectively, for the negative log-likelihood of the observations. The 2nd and 5th columns of the log contain the train and test losses for the negative log-likelihood of the ELBO loss. The 3rd and 6th columns of the log contain the train and test losses for the KL-divergence. Use the log to monitor training performance, and if extra training is needed, modify the `run.cfg` file in the `train` section to restore previous weights. For example:

```
[train]
lr = 0.0005
restore = True
```

After training, we can generate predictions of velocity, thickness, and rigidity mean and standard deviation using the `predict_joint.ipynb` notebook. We can generate shelf-wide samples of the rigidity using the notebook `generate_covariance.ipynb`. That notebook utilizes a Gibbs sampler to sequentially generate a shelf-wide rigidity sample using block-by-block mean and covariance matrices.