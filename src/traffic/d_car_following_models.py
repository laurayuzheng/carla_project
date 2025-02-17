"""
Contains several custom car-following control models.

These controllers can be used to modify the acceleration behavior of vehicles
in Flow to match various prominent car-following models that can be calibrated.

Each controller includes the function ``get_accel(self, env) -> acc`` which,
using the current state of the world and existing parameters, uses the control
model to return a vehicle acceleration.
"""
from code import interact
import math
import numpy as np
import torch 
from flow.controllers.d_base_controller import DiffBaseController

def simulate_idm_timestep(q_0: torch.Tensor, rl_actions: torch.Tensor, rl_indices=[], t_delta=0.1, v0=30., s0=2., T=1.5, a=0.73, b=1.67, delta=4):
    vehicle_length = 5.
    q = torch.zeros_like(q_0).type_as(q_0)
    rl_actions_i = 0
    q_clone = q_0.clone()
    max_accels = torch.zeros_like(rl_actions).type_as(q_0)
    
    vs = q_clone[1::2]
    xs = q_clone[0::2]

    last_xs = torch.roll(xs, 1)
    last_vs = torch.roll(vs, 1)

    s_star = s0 + vs*T + (vs * (vs - last_vs))/(2*math.sqrt(a*b))
    interaction_terms = (s_star/(last_xs - xs - vehicle_length))**2
    interaction_terms[0] = 0.

    dv = a * (1 - (vs / v0)**delta - interaction_terms) # calculate acceleration
    
    for i in rl_indices: # use RL vehicle's acceleration action
        if i == 0:
            max_accel = 20 
        else: 
            max_accel = (xs[i-1] - xs[i] + 2*t_delta*vs[i-1] - 2*t_delta*vs[i] + t_delta**2*dv[i-1] + s0) / (t_delta**2)
            max_accels[rl_actions_i] = max_accel 

        dv[i] = torch.sigmoid(rl_actions[rl_actions_i]) * max_accel
        rl_actions_i += 1

    q[0::2] = xs + vs*t_delta
    q[1::2] = torch.max(vs + dv*t_delta, torch.tensor([0]).type_as(q_0))

    return q

def IDMJacobian(q_0, rl_indices, max_num_rl, t_delta=0.1, v0=30., s0=2., T=1.5, a=0.73, b=1.67, delta=4):
    '''rl_indices does not necessarily contain max number of vehicles, so we cannot infer the number of vehicles from indices directly. '''
    vehicle_length = 5.
    num_vehicles = int(len(q_0) / 2)
    J = torch.zeros((2*num_vehicles, 2*num_vehicles)).type_as(q_0)
    J_actions = torch.zeros((2*num_vehicles, max_num_rl)).type_as(q_0)
    
    ind = np.diag_indices(num_vehicles)
    ind = (torch.Tensor(ind[0]).long().type_as(q_0), torch.Tensor(ind[1]).long().type_as(q_0))

    ind = ((ind[0]*2).long(), (ind[1]*2).long())
    ind2 = (ind[0], (ind[1]+1))
    ind3 = (ind[0]+1, ind[1])
    ind4 = (ind[0]+1, ind[1]+1)

    subind3 = (ind[0]+1, ind[1]-2)
    subind4 = (ind[0]+1, ind[1]-1)

    vs = q_0[1::2]
    xs = q_0[0::2]

    last_xs = torch.roll(xs.clone(), 1)
    last_vs = torch.roll(vs.clone(), 1)

    s_star = s0 + vs*T + (vs * (vs - last_vs))/(2*math.sqrt(a*b))
    s_alpha = last_xs - xs - vehicle_length

    interaction_terms = (s_star/(last_xs - xs - vehicle_length))**2
    interaction_terms[0] = 0. # leading vehicle does not need 

    dv = a * (1 - (vs / v0)**delta - interaction_terms)
    
    J[ind2[0], ind2[1]] = 1.
    J[ind3[0], ind3[1]] = (-2 * a * s_star**2)/ (s_alpha**3) # dg / dx
    J[ind4[0], ind4[1]] = (-a*delta*(vs**(delta-1))/(v0**delta)) + \
                (-2*a/(s_alpha**2))*(T + (vs + vs - last_vs)/(2*math.sqrt(a*b)))*s_star # dg /dv
    
    J[1, 0] = 0
    J[1, 1] = (-a*delta*(q_0[1]**(delta-1))/v0**delta)

    J[subind3[0], subind3[1]] = (2*a*(s_star**2)) / (s_alpha**3)
    J[subind4[0], subind4[1]] = (2*a*s_star*vs) / (2*math.sqrt(a*b)*(s_alpha**2))

    J[1, -2] = 0.
    J[1, -1] = 0. 

    for ind,i in enumerate(rl_indices):
        J_actions[2*i+1, ind] = 1 # RL action influences state in update only
        J[2*i, 2*i] = 0. 
        J[2*i+1, 2*i] = 0. 
        J[2*i, 2*i+1] = 0. 
        J[2*i+1, 2*i+1] = 0. 
        J[2*i, 2*i - 2] = 0.
        J[2*i+1, 2*i - 2] = 0.
        J[2*i, 2*i - 1] = 0.
        J[2*i+1, 2*i - 1] = 0.

    return J, J_actions * t_delta

class IDMStepLayer(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, input, rl_actions, rl_indices, max_num_rl, sim_step, v0, s0, T, a, b, delta):
        ctx.sim_step = sim_step 
        ctx.v0 = v0 
        ctx.s0 = s0 
        ctx.T = T 
        ctx.a = a 
        ctx.b = b 
        ctx.delta = delta 
        ctx.rl_indices = rl_indices
        
        J, J_actions = IDMJacobian(input, rl_indices=rl_indices, max_num_rl=max_num_rl, t_delta=sim_step, 
                                v0=v0, s0=s0, T=T, a=a, b=b, 
                                delta=delta)

        ctx.save_for_backward(input, J, J_actions)


        return simulate_idm_timestep(input, rl_actions=rl_actions, 
                                        rl_indices=rl_indices, t_delta=sim_step, 
                                        v0=v0, s0=s0, T=T, a=a, b=b, delta=delta)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q0, J, J_actions = ctx.saved_tensors  
        ones = torch.ones(q0.size()).type_as(q0)
        ones = torch.diag(ones, 0).type_as(q0)
        
        J = J*0.1 + ones 
        J = J.cpu().numpy() 
        J_actions = J_actions.cpu().numpy() 

        grad_clone = grad_output.cpu().numpy().copy()
        one_hot = (grad_clone.sum()-np.ones(grad_clone.shape[0])).sum()==0

        if one_hot: 
            grad_input = J[np.where(grad_clone==1)]
            grad_rl_actions = J_actions[np.where(grad_clone==1)]
        else:
            grad_input = J.T @ grad_clone
            grad_rl_actions = J_actions.T @ grad_clone
        
        grad_input = torch.Tensor(grad_input).float().type_as(q0)
        grad_rl_actions = torch.Tensor(grad_rl_actions).float().type_as(q0)

        return grad_input, grad_rl_actions, None, None, None, None, None, None, None, None, None 

