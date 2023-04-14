
## Code to run token algorithms

This repository contains the code for the AISTATS 2023 submission "A principled framework for the design and analysis of
token algorithms", which is accessible at the following link: https://proceedings.mlr.press/v206/hendrikx23a.html
Details and theory for the various token algorithms can be found in the paper.  

# Requirements

Create a new python 3 environment:

`conda create --name token_env python=3.8`

Then, switch to this environment using:

`conda activate token_env`

Install the following packages:

`conda install openmpi matplotlib numba scikit-learn mpi4py numpy scipy`


# Run the code

To run the code, and plot the results, use the command:

`mpirun -n nb_nodes python main.py --plot`

with nb_nodes the number nodes in the graph. For instance:

`mpirun -n 4 python main.py --plot`

# Configuration

By default, the `config.json` file in the current directory is used. Another file can be specified using the `--config_file` option. A sample configuration file is provided that the user can modify to test different options.

In particular, it is necessary to change the `path_to_data` and `output_path` options to specify the dataset that should be used. 
    
