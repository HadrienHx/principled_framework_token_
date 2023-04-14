from lib2to3.pgen2 import token
from mpi4py import MPI
import numpy as np
import pickle
from solver.distributed.distributed_solver import DistributedSolver 


class Token_algo(DistributedSolver):

    """
    TODO: Change for specific token step types
    """
    COMM_STEP_TYPE = 1
    COMP_STEP_TYPE = 2

    def __init__(self, nb_tokens=1, batch_factor=1., **kwargs):
        super(Token_algo, self).__init__(**kwargs)
        self.nb_tokens = nb_tokens

        self.nb_comps_per_step = self.dataset.N
        self.computation_time_per_step = self.batch_computation_time

        self.error_computation_period = 50 * self.graph.size
        
        self.initialize_sigma_and_tau()
        self.initialize_smoothness_params(batch_factor)
        self.initialize_params()
        self.initialize_tokens()

        self.iterations_multiplier = 2 *  int(self.graph.size * (self.p_comp + self.p_comm * self.nb_tokens))

        self.log.info(f"Number of tokens: {self.nb_tokens}")


    def initialize_params(self):
        # Initializing parameters
        self.z = np.zeros(self.dataset.d)
        self.last_g = self.model.get_gradient_without_reg(self.z, self.dataset)
        self.x = - self.model.get_gradient_without_reg(self.z, self.dataset) / self.sigma

    def initialize_local_smoothnesses(self, smooth):
        self.L_rel = self.graph.size * self.alpha * (1 + smooth[0] / self.sigma)

        L_comp_buffer = np.array([self.L_rel])
        max_L_comp = np.empty((1,), dtype=np.float64)
        self.comm.Allreduce(L_comp_buffer, max_L_comp, op=MPI.MAX) 
        return L_comp_buffer[0]


    def initialize_smoothness_params(self, batch_factor):
        # Initializing LRs
        self.log.info(f"batch_factor: {batch_factor}")
        self.local_smoothnesses = self.model.get_smoothnesses(self.dataset)

        own_smooth = np.array([self.sigma + batch_factor * np.sum(self.local_smoothnesses)])
        smooth = np.empty((1,), dtype=np.float64)
        self.comm.Allreduce(own_smooth, smooth, op=MPI.MAX) 

        nb_nodes = self.graph.size 
        self.alpha = 2 * self.nb_tokens / smooth[0] 

        L_comp = self.initialize_local_smoothnesses(smooth)

        L_comm_buffer = np.array([2 * self.nb_tokens * nb_nodes / (self.sigma * self.get_gamma())])

        max_L_comm = np.empty((1,), dtype=np.float64)
        self.comm.Allreduce(L_comm_buffer, max_L_comm, op=MPI.MAX) 
        L_comm = max_L_comm[0]

        self.p_comp = 1. / (1 + L_comm / L_comp)
        self.p_comm = 1 - self.p_comp
        
        self.log.info(f"p_comp: {self.p_comp}")
        
        # The two should be equal
        self.step_size = min(self.p_comp / L_comp, self.p_comm / L_comm) 
        assert(np.isclose(self.p_comp / L_comp,self.p_comm / L_comm))

        self.rho = nb_nodes * self.step_size * self.alpha / self.p_comp

    def get_gamma(self):
        if self.graph.name == "complete":
            return 1.
        # We're not tuning down the step-size, since it doesn't really seem necessary
        return 1. # self.graph.gamma


    def initialize_sigma_and_tau(self):
        self.sigma = self.model.c * ( self.graph.size / (self.graph.size + self.nb_tokens))


    def initialize_tokens(self):
        assert(not self.nb_tokens > self.graph.size)
        self.token_location = list(range(self.nb_tokens))
        self.tokens = [None] * self.nb_tokens

        if self.id < self.nb_tokens:
            self.tokens[self.id] = np.zeros(self.dataset.d)

    def run_step(self):
        # It is important to have a coordinated seed for this.
        if self.rs.random() < self.p_comm:
            self.comm_step()

        else:
            active_node = self.rs.randint(self.graph.size)
            if self.id == active_node:
                self.comp_step()
                self.nb_comp += self.nb_comps_per_step
                self.current_time += self.computation_time_per_step
                self.step_type.append(self.COMP_STEP_TYPE)
 
    def comm_step(self):
        token_id = self.rs.randint(self.nb_tokens)
        dest = self.push_token(token_id)
        if self.id == dest: 
            self.update_param_with_token(token_id)

    def push_token(self, token_id):
        self.nb_comm += 1. 
        source = self.token_location[token_id]
        dest = self.rs.choice(self.graph.neighbours[source] + [source])
        self.exchange_token(token_id, source, dest)

        source_and_travel_time = self.get_time(source, dest, self.current_time)

        if self.id == dest:
            self.current_time = max(self.current_time, source_and_travel_time)
            self.step_type.append(self.COMM_STEP_TYPE)
        
        return dest 

    def comp_step(self):
        self.z = (1 - self.rho) * self.z + self.rho * self.x
        new_g = self.model.get_gradient_without_reg(self.z, self.dataset)
        self.x = self.x - (new_g - self.last_g) / self.sigma

        self.last_g = new_g


    def exchange_token(self, token_id, source, dest):
        if not source == dest: 
            if self.id == source:
                self.send_to_neighbour(self.tokens[token_id], dest)
                self.tokens[token_id] = None

            elif self.id == dest:
                self.tokens[token_id] = self.receive_from_neighbour(source)

            self.token_location[token_id] = dest


    def update_param_with_token(self, token_id):
        average_update =  self.x - self.tokens[token_id]
        coeff = self.nb_tokens * self.graph.size * self.step_size / (self.sigma * self.p_comm)

        self.x = self.x - coeff * average_update
        self.tokens[token_id] = self.tokens[token_id] + coeff * average_update
    
    def update_time(self):
        pass