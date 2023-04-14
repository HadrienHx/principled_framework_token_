import numpy as np 
from mpi4py import MPI 
import logging 

from solver.solver import Solver


class DistributedSolver(Solver):
    def __init__(self, comm=None, graph=None, communication_time=1., computation_time=1., **kwargs):
        self.comm = comm 
        self.id = comm.Get_rank()
        super(DistributedSolver, self).__init__(**kwargs)
        self.graph = graph
        self.in_buffer = np.empty([len(self.graph.neighbours[self.id])] + [self.dataset.d], dtype=np.float64)
        self.token_buffer = np.empty(self.dataset.d, dtype=np.float64)
        self.neigh_weights = np.array([self.graph.laplacian[self.id, neigh_id] for neigh_id in self.graph.neighbours[self.id]])
        
        self.log.info(f"gamma: {self.graph.gamma}")
        
        self.batch_computation_time = computation_time * self.dataset.N
        self.computation_time = computation_time
        self.communication_time = communication_time

        self.nb_comm = 0
        self.nb_comp = 0

        self.step_type = []

    def get_logger(self):
        log = logging.getLogger(self.name + " " + str(self.id))
        if self.id > 0:
            log.setLevel(logging.WARNING)
        return log 

    def compute_error(self):
        if self.id == 0:
            return self.error_model.compute_error(self.x, self.error_dataset)

    def multiply_by_w(self, local_grad):
        return self.graph.laplacian[self.id, self.id] * local_grad + np.dot(self.neigh_weights, self.exchange_with_neighbors(local_grad))

    def update_time(self):
        self.current_time += self.batch_computation_time + self.communication_time
        self.nb_comm += self.graph.get_nb_edges()
        self.nb_comp += self.dataset.N
        self.step_type.append(0)

    def exchange_with_neighbors(self, data):
        s_reqs = [
            self.comm.Isend(data, dest=neigh_id, tag=self.iteration_number)
                for neigh_id in self.graph.neighbours[self.id]
        ]

        r_reqs = [
            self.comm.Irecv(self.in_buffer[n_nb], source=neigh_id, tag=self.iteration_number)
                for n_nb, neigh_id in enumerate(self.graph.neighbours[self.id])
        ]
        MPI.Request.Waitall(r_reqs + s_reqs)
        return self.in_buffer

    def synch_time(self, source, dest, time):
        if source == dest: 
            return time 

        if self.id == source:  
            other_node = dest 
        elif self.id == dest:
            other_node = source 
        else: 
            raise Exception("One node should be the source and the other the dest to synchronize time")
        
        data = np.array([float(time)])
        in_data = np.array([0.])
        s_req = self.comm.Isend(data, dest=other_node)
        r_req = self.comm.Irecv(in_data, source=other_node)
    
        MPI.Request.Waitall([s_req, r_req])

        return int(max(time, in_data[0]))

    def get_time(self, source, dest, time):
        if source == dest: 
            return time 

        if self.id == source:
            data = np.array([float(time)])
            s_req = self.comm.Isend(data, dest=dest)
            MPI.Request.Wait(s_req)
            return time

        elif self.id == dest:
            in_data = np.array([0.])
            r_req = self.comm.Irecv(in_data, source=source)
            MPI.Request.Wait(r_req)
            return in_data[0] + self.communication_time
    

    def send_to_neighbour(self, data, dest):
        req = self.comm.Isend(data, dest=dest, tag=self.iteration_number)
        MPI.Request.Wait(req)

    def receive_from_neighbour(self, source):
        req = self.comm.Irecv(self.token_buffer, source=source, tag=self.iteration_number)
        MPI.Request.Wait(req)
        return self.token_buffer

    def init_chebyshev_constants(self):
        self.nb_comm_steps = int(np.ceil(1. / np.sqrt(self.graph.gamma)))
        assert(self.graph.gamma < 1.)

        sq_gamma = np.sqrt(self.graph.gamma)

        self.c1 = (1 - sq_gamma) / (1 + sq_gamma)
        self.c2 = (1 + self.graph.gamma) / (1 - self.graph.gamma)
        self.c3 = 2 / ((1 + self.graph.gamma) * self.graph.max_eig)

    def multiply_by_PW(self, local_grad, nb_comm_steps):
        a0 = 1. 
        a1 = self.c2 
        x0 = local_grad
        x1 = self.c2 * (local_grad - self.c3 * self.multiply_by_w(local_grad))

        for _ in range(1, nb_comm_steps):
            a1, a0 = [2 * self.c2 * a1 - a0, a1]
            x1, x0 = [2 * self.c2 * (x1 - self.c3 * self.multiply_by_w(x1)) - x0, x1]

        return local_grad - x1 / a1