from mpi4py import MPI
import numpy as np
import pickle
from solver.distributed.distributed_solver import DistributedSolver 


class Walkman(DistributedSolver):

    """
    TODO: Change for specific token step types
    """
    COMM_STEP_TYPE = 1
    COMP_STEP_TYPE = 2

    def __init__(self, batch_factor=1., **kwargs):
        super(Walkman, self).__init__(**kwargs)
        
        self.nb_comps_per_step = self.dataset.N
        self.computation_time_per_step = self.batch_computation_time

        self.error_computation_period = 50 * self.graph.size
        
        self.initialize_sigma_and_tau()
        self.initialize_smoothness_params(batch_factor)
        self.initialize_params()
        self.initialize_tokens()

        self.iterations_multiplier = int(self.graph.size)


    def initialize_sigma_and_tau(self):
        self.sigma = self.model.c

    def initialize_params(self):
        # Initializing parameters
        self.y = np.zeros(self.dataset.d)
        self.z = np.zeros(self.dataset.d)

        # Only used for error computation
        self.x = np.zeros(self.dataset.d)

    def initialize_smoothness_params(self, batch_factor):
        self.local_smoothnesses = self.model.get_smoothnesses(self.dataset)
        own_smooth = np.array([self.sigma + batch_factor * np.sum(self.local_smoothnesses)])
        smooth = np.empty((1,), dtype=np.float64)
        self.comm.Allreduce(own_smooth, smooth, op=MPI.MAX) 

        self.beta = smooth[0]

    def initialize_tokens(self):
        self.token_location = 0
        self.token_x = None

        if self.id == 0:
            self.token_x = np.copy(self.x)

    def run_step(self):
        # It is important to have a coordinated seed for this.
        source = self.token_location
        dest = self.rs.choice(self.graph.neighbours[source] + [source])
        self.exchange_token(dest)
        self.nb_comm += 1

        if self.id == dest:
            self.comp_step()
            self.nb_comp += self.nb_comps_per_step
        
    def comp_step(self):
        new_y = self.token_x + (self.z - self.model.get_gradient(self.y, self.dataset)) / self.beta
        new_z = self.z + self.beta * (self.token_x - new_y) 
        
        self.token_x += ((new_y - new_z / self.beta) - (self.y - self.z / self.beta)) / self.graph.size
        self.y = new_y
        self.z = new_z

        # Only used for error computation
        self.x = self.token_x

    def exchange_token(self, dest):
        source = self.token_location
        if not source == dest: 
            if self.id == source:
                self.send_to_neighbour(self.token_x, dest)
                self.token_x = None

            elif self.id == dest:
                self.token_x = self.receive_from_neighbour(source)

            self.token_location = dest

        if self.id == source or self.id == dest:
            self.current_time = self.synch_time(source, dest, self.current_time)
            self.current_time += self.communication_time + self.computation_time_per_step
            
            self.step_type.append(self.COMM_STEP_TYPE)
            self.step_type.append(self.COMP_STEP_TYPE)

    def update_time(self):
        pass