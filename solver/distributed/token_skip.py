from solver.distributed.simple_token import Token_algo 


class Token_skip(Token_algo):
    COMM_STEP_TYPE = 1
    COMP_STEP_TYPE = 2

    def __init__(self, **kwargs):
        super(Token_skip, self).__init__(**kwargs)
        self.comm_steps_to_mixing = int(1. / self.graph.gamma)
        
    def comm_step(self):
        token_id = self.rs.randint(self.nb_tokens)
        for _ in range(self.comm_steps_to_mixing):
            dest = self.push_token(token_id)
    
        if self.id == dest: 
            self.update_param_with_token(token_id)