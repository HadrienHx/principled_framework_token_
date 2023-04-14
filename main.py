from numba import njit
import scipy.sparse as sparse
import numpy as np
import logging

"""
To run: 
`mpirun -n 12 --oversubscribe python main.py --plot'
"""

import mpi4py
import argparse
import time 
import os 


from dataset.libsvm_loader import LIBSVM_Loader 
from models.logistic_regression import LogisticRegression
from solver.distributed.solvers import AvailableSolvers
from graph.basic_graphs import get_graph_class
from plotter import Plotter
from parser import Parser 


# Retrieve args
args_parser = Parser()
args, data_args, algo_args, model_args, solvers = args_parser.get_args()

# Identify node
comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()


# Set up loggers
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(f"Main {rank}")

# Load model and dataset
model = LogisticRegression(**model_args)

error_model = model.get_global(comm.size)
global_dataset = LIBSVM_Loader(**data_args, seed=args.seed, rank=rank).load(**data_args, comm_size=comm.size)
local_dataset = global_dataset.get_truncated(rank, comm.size)
if rank > 0:
    global_dataset = None

# Build graph
graph = get_graph_class(args.graph)(comm.size, seed=args.seed, logger=log)
log.info(graph)

# Name the run
filename = str(time.time()).split(".")[0]
timestamp = np.array([int(filename)])
comm.Bcast(timestamp, root=0)
filename = str(timestamp[0])

run_solvers = []
for solver_name, solver_args in solvers:
    # Retrieve solver (server / worker)
    solver_class = AvailableSolvers.get(solver_name)
    solver = solver_class(name=solver_name, comm=comm,
                model=model, error_model=model, graph=graph,
                dataset=local_dataset, error_dataset=global_dataset,
                seed=args.seed, timestamp=filename, **algo_args, **solver_args)

    # Run algorithm
    solver.solve(args.nb_epochs)
    run_solvers.append(solver)

if rank == 0:

    # Maybe just replace by get_attributes or sth.
    x_time = [solver.time for solver in run_solvers]
    x_comm = [solver.x_comm for solver in run_solvers]
    x_comp = [solver.x_comp for solver in run_solvers]

    for metric, m_name in [(x_time, "time"), (x_comm, "comm"), (x_comp, "comp")]:
        plotter = Plotter(filename=args.plotter_path)
        
        for solver_metric, solver, (_, solver_args) in zip(metric, run_solvers, solvers): 
        # Servers stores results for plotting
            plotter.add_curve(solver.name, solver.error, solver_metric, solver_args,
                [solver.step_type, solver.iteration_index])

        #_metric plots and saves results to disk
        plotter.plot(show=args.plot, xlabel=m_name)
        plotter.save(filename=filename, output_path=args.output_path, suffix=m_name, save_png=args.save_png)
    
    args_parser.save_config(filename=filename, output_path=args.output_path)
    log.info("Finished execution")