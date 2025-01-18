from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                print("grad",p.grad.data)
                # State should be stored in this dictionary
                state = self.state[p]
                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                print("hyperparameter is",group)
                ###initalize all hyperparameter
                if 'm' not in state:
                    state['m']=0
                    print('m is initialized')
                if 'n' not in state:
                    state['n']=0
                    print('n is initialized')
                if 't' not in state:
                    state['t']=torch.tensor(0)
                    print('t is initialized')
                state['t']=state['t']+1
                print("state[t]",state['t'])
                # Update first and second moments of the gradients
                state['m']=group["betas"][0]*state['m']+(1-group["betas"][0])*grad
                state['n']=group["betas"][1]*state['n']+(1-group["betas"][1])*(grad*grad)
                print('self.state[n]',state['n'])
                print('self.state[m]',state['m'])

                if group["correct_bias"]:
                # Bias correction
                    #m_hat=state['m']/(1-torch.pow(group["betas"][0],state['t']))
                    #n_hat=state['n']/(1-torch.pow(group["betas"][1],state['t']))
                    #print("m_hat is",m_hat)
                    #print("n_hat is",n_hat)
                    a_t=alpha*torch.sqrt((1-torch.pow(group["betas"][1],state['t'])))/(1-torch.pow(group["betas"][0],state['t']))
                    #p.data=p.data-(m_hat*alpha/(torch.sqrt(n_hat)+group["eps"])+group["weight_decay"]*p.data)
                    p.data=p.data-(a_t*state['m']/(torch.sqrt(state['n'])+group['eps']))
                    p.data=p.data-group["weight_decay"]*p.data*alpha
                else:
                    p.data=p.data-state['m']*alpha/(torch.sqrt(state['n'])+group["eps"])+group["weight_decay"]*p.data
                    print("we are here!")
                print("p.data is now",p.data)
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                
                # Update parameters
                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.

        return loss
