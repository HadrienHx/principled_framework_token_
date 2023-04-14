from mpi4py import MPI
import numpy as np

from solver.utils import sparse_dot, sparse_add, AmortizedChoice
from solver.distributed.simple_token import Token_algo 


class Token_VR(Token_algo):

    COMM_STEP_TYPE = 1
    COMP_STEP_TYPE = 2

    def __init__(self, **kwargs):
        super(Token_VR, self).__init__(**kwargs)

        self.nb_comps_per_step = 1
        self.computation_time_per_step = self.computation_time

        self.error_computation_period = 5000 * self.graph.size
        self.iterations_multiplier = int(self.graph.size * (self.p_comp * self.dataset.N + self.p_comm * self.nb_tokens)) / 50

    def initialize_params(self):
        # Initializing parameters
        self.z = np.zeros(self.dataset.N)
        self.last_g = [self.model.get_1d_stochastic_gradient(z, self.dataset, i) for i, z in enumerate(self.z)]
        self.x = np.zeros((self.dataset.d))
        for i, g in enumerate(self.last_g):
            sparse_add(self.x, self.dataset.X.T[i].indices, self.dataset.X.T[i].data, - g / self.sigma)

        self.X = [[x.data, x.indices] for x in self.dataset.X.T]

    def initialize_local_smoothnesses(self,smooth):
        self.fs_probas = np.ones((self.dataset.N,)) / self.dataset.N
        self.amortized_choice = AmortizedChoice(self.fs_probas)

        self.L_rel_array = np.divide(
             self.graph.size * self.alpha * (1 + np.max(self.local_smoothnesses) / self.sigma), self.fs_probas 
            )

        L_comp_buffer = np.array([max(self.L_rel_array)])
        max_L_comp = np.empty((1,), dtype=np.float64)
        self.comm.Allreduce(L_comp_buffer, max_L_comp, op=MPI.MAX) 
        return max_L_comp[0]

    def comp_step(self):
        j = self.amortized_choice.get() #j = np.random.choice(self.local_samples, p=self.fs_probas)
        rho_ij = self.rho / self.fs_probas[j]
        assert(rho_ij < 1.)
        data_j, indices_j = self.X[j]
        self.z[j] = (1 - rho_ij) * self.z[j] + rho_ij * sparse_dot(self.x, indices_j, data_j)
        new_g = self.model.get_1d_stochastic_gradient(self.z[j], self.dataset, j)
        sparse_add(self.x, indices_j, data_j, - (new_g - self.last_g[j]) / self.sigma) 
        
        self.last_g[j] = new_g
