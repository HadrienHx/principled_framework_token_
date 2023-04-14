from mpi4py import MPI
import numpy as np
import pickle 

from solver.distributed.distributed_solver import DistributedSolver


class GD(DistributedSolver):
    def __init__(self, batch_factor=1., **kwargs):
        super(GD, self).__init__(**kwargs)

        own_smooth = np.array([batch_factor * np.sum(self.model.get_smoothnesses(self.dataset)) + self.model.c])
        max_L = np.empty((1,), dtype=np.float64)
        self.comm.Allreduce(own_smooth, max_L, op=MPI.MAX) 

        self.step_size = 1. / (max_L[0])

        self.g_global = np.empty(self.dataset.d)

    def run_step(self):
        g = self.model.get_gradient(self.x, self.dataset)
        self.comm.Allreduce(g, self.g_global, op=MPI.SUM)  
        self.x -= self.step_size * self.g_global / self.comm.size
