CosyMAML contains the code used for the development of the neural network based cosmological emulator presented in (paper), wherein we investigate the use of the MAML algorithm (link to MAML github) in cosmology.

It is recomnended to set up the environment using conda
```bash
    conda env create -f environment.yml
    conda activate cosymaml
```

The codebase is structured as follows:

- src: A module containing the code used to define neural network architechtures (models.py), train and test said neural networks (training.py), generate training data (simulate.py) and run Markov-Chain Monet-Carlo inference (mcmc.py)

- results: Contains hdf5 files which contain results of various tests of the emulators presented in (paper)

- plots: Constains png and pdf files of the plots presented in (paper)

- OLD: An unorganised collection of various notebooks and scripts that were previously used in the development of the code found in src and the scripts/notebooks present in the parent directory.

- weights: Trained weights for the emulator, including both weights that have been trained using MAML and those that have been trained in the standard fashion.

- TJPCov: A clone of the (TJPCov) code used to estimate covariance matrices for cosmoloigcal surveys. This is directly added here rather than installed as part of the requirments, as the source code of ~/TJPCov/tjpcov/covariance_gaussian_fsky.py has been modified to allow for the logarithmic binning of angular multipoles used in this work.

Aside from these sub-directories, several scripts and notebooks are present in the parent directory. These are organised by tags present at the beginning of their names:

- ANALYSIS_: A notebook used to produce and/or plot results presented in (paper)

- PLOTTING_: A notebook which does not produce any results itself, but produces plots from (paper) using a file in the results directory

- generate_: A script used to generate power spectra for training and testing the emulators.

- run_: A script which produces results presented in (paper).

