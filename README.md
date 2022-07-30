# Variational inference of ice shelf rheology with physics-informed machine learning

This repository contains example Python scripts and notebooks that implement the methods presented in "Variational inference of ice shelf rheology with physics-informed machine learning" by Riel and Minchew, 2022. The examples included here are inference of ice rheology for 1D (`shelf1d`) and 2D (`shelf2d`) synthetic ice shelves.

## Prerequisites

All neural network and variational Gaussian Processes are implemented using TensorFlow v2.7+. A few segments of the training scripts use data objects from the `pgan` package (https://github.com/bryanvriel/pgan/tree/lite-tf2) and training loops from the `hdflows` package (https://github.com/bryanvriel/high_dimensional_flows/tree/pgan). Snapshots of both packages are included in the `external` folder.
