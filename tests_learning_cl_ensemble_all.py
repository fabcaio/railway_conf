# tests_learning_cl_ensemble only tests milp and learning+lp methods
# this script is more complete and it also tests minlp(10minutes), minlp(with warm-start) and learning+nlp

from rail_fun import cost_per_step, build_delta_vector, get_state_learning, build_stacked_state
import numpy as np
from rail_rl_env import RailNet, d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot, mdl_feasible, action_dict, gurobi_minlp, gurobi_nlp_presolve, gurobi_lp_presolve, gurobi_milp
import time
from rail_data_preprocess_reduced import get_dataset_min_max, get_preprocessed_data
from rail_training_reduced import Network, Network_CE, Network_mask1, Network_mask2, Network_mask3
import torch
import sys, os
import datetime

testing = False

x = datetime.datetime.now()
date_str = '%.2d%.2d_%.2d%.2d' %(x.month, x.day, x.hour, x.minute)
print('date_time: ' + date_str)

start_loop = time.time()

# added by Xiaoyu
match_nlp = np.zeros(3, dtype=int)
match_lp = np.zeros(3, dtype=int)
match_constant = np.zeros(3, dtype=int)
match_constant[0] = 21 # 4 x 21: time for the train to return the first station (from the original timetable)
match_constant[1] = 27
match_constant[2] = 23

N = 40

opt_label = 'classification' #(regression is not implemented yet)

if testing == True:
    import psutil
    p = psutil.Process()
    job_idx = 1
    n_threads = 8
    timeout = 5*60 #minutes
    timelimit_gurobi = 10
    timelimit_reference = 10 # sets the time for the longest milp (with warm-start)
else:
    n_threads = int(sys.argv[1])
    timelimit_job = int(sys.argv[2]) #(in seconds)    
    job_idx = int(sys.argv[3])
    timeout = timelimit_job - 40*60
    timelimit_gurobi = 240 # (in seconds, for minlp and milp)
    timelimit_reference = 600 # sets the time for the longest milp (with warm-start)
    
    # for testing
    # testing=True
    # timeout = 25*60 #minutes
    # timelimit_gurobi = 21
    # timelimit_reference = 21 # sets the time for the longest milp (with warm-start)
    
device='cpu'
torch.set_num_threads(n_threads)

num_layers=1
lr=1e-3
batch_size=1 #
threshold_counts = 50

mipgap = 1e-3
log = 0
early_term = 1

N_control = N-2
seed = 4
np.random.seed(seed)
torch.manual_seed(seed)

print('N: %d \t mipgap: %.4f \t timelimit_gurobi: %.2f \t n_threads: %d \t early_term: %d' %(N, mipgap, timelimit_gurobi, n_threads, early_term))
print('threshold counts: %d' %threshold_counts)
print('timeout: %.2f,\t' %timeout, 'mipgap %.4f' %mipgap)

def build_network(network_type, input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout, list_masks=None, device=device):
    
    if network_type == 'Network':
        network = Network(input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout)
    elif network_type == 'Network_CE':
        network = Network_CE(input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout)
    elif network_type == 'Network_mask1':
        network = Network_mask1(input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout, list_masks, device)
    elif network_type == 'Network_mask2':
        network = Network_mask2(input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout, list_masks, device)
    elif network_type == 'Network_mask3':
        network = Network_mask3(input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout, list_masks, device)
    else:
        raise KeyError('Network type has not been found.')
    
    return network

#############################################################

opt_data = 'milp_cl'
opt_preprocess='reduced'
  
dict_data = get_dataset_min_max(opt_data, opt_preprocess, threshold_counts, N, testing, opt_label='classification')
list_masks = dict_data['list_masks']
total_action_set_reduced = dict_data['total_action_set']
action_dict_reduced = {}
for i in total_action_set_reduced:
    action_dict_reduced[str(i)] = action_dict[str(i)]
num_actions_reduced = total_action_set_reduced.shape[0]

if testing==True:
    hyperpar_set = np.load('training_data_%s/milp_cl_N40_hyperparm_set.npy' %opt_preprocess, allow_pickle=True).item()
else:
    hyperpar_set = np.load('//scratch/cfoliveiradasi//railway_conf//training_data_reduced//milp_cl_N40_hyperparm_set.npy', allow_pickle=True).item()

model_list_reduced = []
h0 = []
c0 = []
opt_state = []
state_min_reduced = []
state_max_reduced = []
idx_best_networks = [51, 17, 21, 55, 49, 23, 19, 53, 61, 29, 25, 27, 57, 59, 31]
num_networks = len(idx_best_networks)
for idx in idx_best_networks:
    
    if testing==True:
        network_info_path = 'training_data_%s//' %opt_preprocess + opt_data + '_N%d_%.3d_' %(N,idx) + 'info.npy'
        network_weight_path = 'training_data_%s//' %opt_preprocess + opt_data + '_N%d_%.3d_' %(N,idx) + 'weight'
    else:
        network_info_path = '//scratch/cfoliveiradasi//railway_conf//training_data_%s//%s_N%.2d_%.3d_info.npy' %(opt_preprocess, opt_data, N, idx)
        network_weight_path = '//scratch/cfoliveiradasi//railway_conf//training_data_%s//%s_N%.2d_%.3d_weight' %(opt_preprocess, opt_data, N, idx)
        
    opt_state.append(hyperpar_set[str(idx)][-1])
    state_min_reduced.append(dict_data['state_min_%d' %opt_state[-1]])
    state_max_reduced.append(dict_data['state_max_%d' %opt_state[-1]])
        
    network_info = np.load(network_info_path, allow_pickle=True).item()    
    network_type = network_info['network_type']
    input_size_reduced = network_info['input_size']
    hidden_size = network_info['hidden_size']
    num_layers = network_info['num_layers']
    lr = network_info['lr']
    dropout = network_info['dropout']
    LR_scheduler = network_info['LR_scheduler']
    
    network = build_network(network_type, input_size_reduced, hidden_size, num_layers, lr, num_actions_reduced, batch_size, dropout, list_masks=list_masks)
    
    network.load_state_dict(torch.load(network_weight_path, weights_only=True))
    
    h0.append(torch.zeros(num_layers, 1, hidden_size))
    c0.append(torch.zeros(num_layers, 1, hidden_size))
    
    model_list_reduced.append(network.eval())
    
##########################################################

# opt_preprocess='original'
  
# model_list_original = []
# h0_original = []
# c0_original = []
# opt_state_original = []
# state_min_original = []
# state_max_original = []
# idx_best_networks_original = [14, 15, 12, 1, 0, 10, 7, 3, 11, 2, 6, 8, 13, 9, 4]
# num_networks_original = len(idx_best_networks_original)
# for idx in idx_best_networks_original:
    
#     if testing==True:
#         network_info_path = 'training_data_%s//' %opt_preprocess + opt_data + '_N%d_%.3d_' %(N,idx) + 'info.npy'
#         network_weight_path = 'training_data_%s//' %opt_preprocess + opt_data + '_N%d_%.3d_' %(N,idx) + 'weight'
#     else:
#         network_info_path = '//scratch/cfoliveiradasi//railway_conf//training_data_%s//%s_N%.2d_%.3d_info.npy' %(opt_preprocess, opt_data, N, idx)
#         network_weight_path = '//scratch/cfoliveiradasi//railway_conf//training_data_%s//%s_N%.2d_%.3d_weight' %(opt_preprocess, opt_data, N, idx)
        
#     network_info = np.load(network_info_path, allow_pickle=True).item()
#     opt_state_original.append(network_info['opt_state'])
#     state_min_original.append(network_info['state_min'])
#     state_max_original.append(network_info['state_max'])
         
#     network_type = network_info['network_type']
#     input_size = network_info['input_size']
#     hidden_size = network_info['hidden_size']
#     num_layers = network_info['num_layers']
#     lr = network_info['lr']
#     dropout = network_info['dropout']
#     LR_scheduler = network_info['LR_scheduler']
    
#     list_masks = network_info['list_masks']
#     total_action_set_original = network_info['total_action_set']
#     action_dict_original = {}
#     for i in total_action_set_original:
#         action_dict_original[str(i)] = action_dict[str(i)]
#     num_actions_original = total_action_set_original.shape[0]
    
#     network = build_network(network_type=network_type, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, lr=lr, num_actions=num_actions_original, batch_size=batch_size, dropout=dropout, list_masks=list_masks, device=device)
#     network.load_state_dict(torch.load(network_weight_path, weights_only=True))
    
#     h0_original.append(torch.zeros(num_layers, 1, hidden_size))
#     c0_original.append(torch.zeros(num_layers, 1, hidden_size))
    
#     model_list_original.append(network)

##########################################################

# original (old - journal paper)

output_get_preprocessed_data = get_preprocessed_data(opt='milp_ol', threshold_counts=25, N=N)

list_masks=output_get_preprocessed_data[6]
state_min_original=output_get_preprocessed_data[9]
state_max_original=output_get_preprocessed_data[10]
total_action_set_original=output_get_preprocessed_data[12]

action_dict_original = {}
for i in total_action_set_original:
    action_dict_original[str(i)] = action_dict[str(i)]

num_actions_original = total_action_set_original.shape[0]

idx_best_networks = [31, 7, 10, 11, 26, 3, 2, 6, 33, 27, 43, 30, 8, 9, 42]
model_list_original = []
h0_original = []
c0_original = []
for i in idx_best_networks:
    network_info = np.load('training_data_old//' + 'milp_ol' + '_N%d_%.3d_' %(N,i) + 'info.npy', allow_pickle=True).item()
    network_type = network_info['network_type']
    input_size_original = network_info['input_size']
    hidden_size = network_info['hidden_size']
    num_layers = network_info['num_layers']
    lr = network_info['lr']
    dropout = network_info['dropout']
    
    network = build_network(network_type, input_size_original, hidden_size, num_layers, lr, num_actions_original, batch_size, dropout, list_masks=list_masks)
    network.load_state_dict(torch.load('training_data_old//' + 'milp_ol' + '_N%d_%.3d_' %(N,i) + 'weight', weights_only=True))
    
    h0_original.append(torch.zeros(num_layers, 1, hidden_size))
    c0_original.append(torch.zeros(num_layers, 1, hidden_size))
    
    model_list_original.append(network.eval())
    
num_networks_original = len(idx_best_networks)
        
##########################################################

# Heuristic rules

# 1 train  -> 00
# 2 trains -> 10
# 3 trains -> 01
# 4 trains -> 11

delta_heuristic1 = np.zeros((12,))
delta_heuristic1[[2,6,10]] = 0     # 1st bit train composition
delta_heuristic1[[3,7,11]] = 0     # 2nd bit train composition
delta_heuristic1[[0,4,8]] = 1      # xi_1 indices
delta_heuristic1[[1,5,9]] = 0      # xi_2 indices

stacked_delta_heuristic1 = []
for i in range(0,N_control):
    stacked_delta_heuristic1.append(delta_heuristic1)
stacked_delta_heuristic1 = np.array(stacked_delta_heuristic1)

delta_heuristic2 = np.zeros((12,))
delta_heuristic2[[2,6,10]] = 1     # 1st bit train composition
delta_heuristic2[[3,7,11]] = 0     # 2nd bit train composition
delta_heuristic2[[0,4,8]] = 1      # xi_1 indices (1 is to keep the original order)
delta_heuristic2[[1,5,9]] = 0      # xi_2 indices

stacked_delta_heuristic2 = []
for i in range(0,N_control):
    stacked_delta_heuristic2.append(delta_heuristic2)
stacked_delta_heuristic2 = np.array(stacked_delta_heuristic2)

# add by Xiaoyu
current_delta = stacked_delta_heuristic2

###########################################################

def store_data(Env, list_data, mdl, warm_start, cost_reference, total_inf_time=0, number_model=0):    
    
    """
    store the closed-loop data in a list for the optimization-based or learning-based approaches
    
    arguments:
        Env: environment of the approach
        list_data: list of the approach
        mdl: output of the optimization (provided by Env.step())
        warm_start: (bool)
        cost_reference: step cost for the reference approach
        reference: (bool) whether the approach is the reference or not
        
    output:

        cost: one-step cost for the approach
    
    """
    
    state_n = np.expand_dims(Env.state_n, axis=1)
    state_n_after = np.expand_dims(Env.state_n_after, axis=1)
    state_d = np.expand_dims(Env.state_d[:,0,:], axis=1)
    state_l = np.expand_dims(Env.state_l[:,0,:], axis=1)
    cost = cost_per_step(state_n, state_n_after, Env.d_pre_cut_old, state_d, state_l, 1)
    state_a = np.expand_dims(Env.state_a[:, 0, :], axis=1)
    
    if warm_start==1:
        runtime = mdl._Runtime
    else:
        runtime = mdl.Runtime
        
    if Env.name == 'minlp_net_ws_10m': # this is the reference
        perc_minlp_cost = 0
    else:
        perc_minlp_cost = (cost-cost_reference)/cost_reference*100
        
    if Env.name == 'learning_nlp' or Env.name == 'learning_lp':
        list_data.append([runtime, cost, perc_minlp_cost, mdl.status, total_inf_time, number_model])
    else:
        list_data.append([runtime, cost, perc_minlp_cost, mdl.status, 0, 0])
    
    print(Env.name + '\t\t %.2f \t\t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(time.time()-start_time, runtime, cost, perc_minlp_cost, mdl.status))
        
    return cost

def get_gap(cost, cost_reference):
    
    """
    This function outputs the relative percentage difference between two values.    
    """
    
    opt_gap = (cost-cost_reference)/cost_reference*100
    return opt_gap

list_milp = []
list_learning_lp = []
list_minlp_net_ws_10m = []
list_minlp_ws = []
list_learning_nlp = []
list_learning_lp_original = []
list_learning_nlp_original = []

Env = RailNet(N)
Env_milp = RailNet(N)
Env_milp.name = 'milp'
Env_learning_lp = RailNet(N)
Env_learning_lp.name = 'learning_lp'
Env_learning_lp_original = RailNet(N)
Env_learning_lp_original.name = 'learning_lp_original'

Env_minlp_net_ws_10m = RailNet(N)
Env_minlp_net_ws_10m.name = 'minlp_net_ws_10m'
Env_minlp_ws = RailNet(N)
Env_minlp_ws.name = 'minlp_ws'
Env_learning_nlp = RailNet(N)
Env_learning_nlp.name = 'learning_nlp'
Env_learning_nlp_original = RailNet(N)
Env_learning_nlp_original.name = 'learning_nlp_original'

# init the current delta
Env_learning_lp.current_delta = current_delta
Env_learning_nlp.current_delta = current_delta
Env_learning_lp_original.current_delta = current_delta
Env_learning_nlp_original.current_delta = current_delta

dict_cntr_infeas = {
    'cntr_infeas_total_learning_lp': 0,
    'cntr_infeas_total_learning_nlp': 0,
    'cntr_infeas_episode_learning_lp': 0,
    'cntr_infeas_episode_learning_nlp': 0,
    'cntr_infeas_total_learning_lp_original': 0,
    'cntr_infeas_total_learning_nlp_original': 0,
    'cntr_infeas_episode_learning_lp_original': 0,
    'cntr_infeas_episode_learning_nlp_original': 0,
    'cntr_infeas_total_heuristic_lp': 0,
    'cntr_infeas_total_heuristic_nlp': 0,
    'cntr_infeas_total_heuristic_lp_original': 0,
    'cntr_infeas_total_heuristic_nlp_original': 0
}

def step_Env_learning(Env, optim, opt_preprocess):
    global flag_end_episode
    
    for i in range(num_networks):
        total_inf_time=0
        early_term = 1
        warm_start = 0
        opt_time = timelimit_gurobi
        
        start_inf_time = time.perf_counter()
        if opt_preprocess=='reduced':
            state_learning = get_state_learning(Env, opt_state[i], opt_preprocess, state_min_reduced[i], state_max_reduced[i], N_control)
            output_net = model_list_reduced[i](state_learning, h0[i], c0[i])
            action_idx = total_action_set_reduced[torch.max(output_net, dim=2)[1].squeeze().numpy()]
            delta_SL = build_delta_vector(action_idx, N_control, action_dict_reduced)
        elif opt_preprocess=='original':
            # state_learning = get_state_learning(Env, opt_state[i], opt_preprocess, state_min_original[i], state_max_original[i], N_control) # for original
            state_learning = build_stacked_state(Env.state_n, Env.state_rho, Env.state_depot, Env.state_l, input_size_original, N_control, state_min_original, state_max_original)
            output_net = model_list_original[i](state_learning, h0_original[i], c0_original[i])
            action_idx = total_action_set_original[torch.max(output_net, dim=2)[1].squeeze().numpy()]
            delta_SL = build_delta_vector(action_idx, N_control, action_dict_original)
        inf_time = time.perf_counter()-start_inf_time
    
        total_inf_time += inf_time
        opt_time = opt_time - inf_time
    
        if optim == 'lp':
            a_values, d_values, r_values, l_values, y_values, n_values, n_after_values,  mdl_learning = gurobi_lp_presolve(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, delta_SL, mipgap, log, opt_time, n_threads)
        elif optim == 'nlp':
            a_values, d_values, r_values, l_values, y_values, n_values, n_after_values,  mdl_learning = gurobi_nlp_presolve(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, delta_SL, mipgap, log, opt_time, early_term, warm_start, n_threads)
        
        number_model = i+1
        
        if mdl_feasible(mdl_learning)==True:
                break
                
    if mdl_feasible(mdl_learning)==True:
        info = Env.step(delta_SL, d_pre, rho_whole, mipgap, log, opt_time, early_term, warm_start, n_threads, opt=optim)[-1]
        Env.current_delta = delta_SL            

    else:
        print('learning + %s + %s: not feasible, model_number: %d' %(optim, opt_preprocess, number_model))
        if opt_preprocess == 'reduced':
            dict_cntr_infeas['cntr_infeas_total_learning_%s' %optim] +=1
            dict_cntr_infeas['cntr_infeas_episode_learning_%s' %optim] +=1
        elif opt_preprocess == 'original':
            dict_cntr_infeas['cntr_infeas_total_learning_%s_original' %optim] +=1
            dict_cntr_infeas['cntr_infeas_episode_learning_%s_original' %optim] +=1

        # add by Xiaoyu_20241120
        for line in range(3):
            if Env.current_delta[N_control - 2, 1 + 4 * line] == 0:
                match_lp[line] = match_constant[line]
            else:
                match_lp[line] = match_constant[line] + 1

        # add by Xiaoyu
        delta_recursive = np.zeros([N_control, 12], dtype=int)
        delta_recursive[:-1, :] = Env.current_delta[1:, :]
        for line in range(3):
            delta_recursive[N_control - 1, 0 + 4 * line] = Env.current_delta[N_control - 2, 0 + 4 * line]
            delta_recursive[N_control - 1, 1 + 4 * line] = Env.current_delta[N_control - 2, 1 + 4 * line]
            delta_recursive[N_control - 1, 2 + 4 * line] = Env.current_delta[N_control - 1 - match_lp[line], 2 + 4 * line]
            delta_recursive[N_control - 1, 3 + 4 * line] = Env.current_delta[N_control - 1 - match_lp[line], 3 + 4 * line]
        
        if optim=='lp':
            a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl_learning = gurobi_lp_presolve(N, Env.d_pre_cut, Env.state_rho, Env.state_a,Env.state_d, Env.state_r, Env.state_l, Env.state_y,Env.state_n, Env.state_depot, delta_recursive, mipgap, log, opt_time, n_threads)
        elif optim=='nlp':
            a_values, d_values, r_values, l_values, y_values, n_values, n_after_values,  mdl_learning = gurobi_nlp_presolve(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, delta_recursive, mipgap, log, opt_time, early_term, warm_start, n_threads)

        # add by Xiaoyu_20241120
        if mdl_feasible(mdl_learning) == True:
            info = Env.step(delta_recursive, d_pre, rho_whole, mipgap, log, opt_time, early_term, warm_start, n_threads, opt=optim)[-1]
            Env.current_delta = delta_recursive
        else:
            print('recursive solution + %s + %s is infeasible' %(optim, opt_preprocess))
            # add by Xiaoyu_20241120
            
            if optim=='lp':
                a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl_learning = gurobi_lp_presolve(N, Env.d_pre_cut, Env.state_rho, Env.state_a,Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, stacked_delta_heuristic1, mipgap, log, opt_time, n_threads)
            elif optim=='nlp':
                a_values, d_values, r_values, l_values, y_values, n_values, n_after_values,  mdl_learning = gurobi_nlp_presolve(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, stacked_delta_heuristic1, mipgap, log, opt_time, early_term, warm_start, n_threads)

            if mdl_feasible(mdl_learning) == True:
                # cost_learning_nlp = store_data(Env_learning_nlp, list_learning_nlp, mdl, warm_start, cost_reference, total_inf_time, number_model)
                info = Env.step(stacked_delta_heuristic1, d_pre, rho_whole, mipgap, log, opt_time, early_term, warm_start, n_threads, opt=optim)[-1]
                Env.current_delta = stacked_delta_heuristic1
            else:
                if opt_preprocess=='reduced':
                    dict_cntr_infeas['cntr_infeas_total_heuristic_%s' %optim] += 1
                elif opt_preprocess == 'original':
                    dict_cntr_infeas['cntr_infeas_total_heuristic_%s_original' %optim] += 1
                flag_end_episode = True
                print('heuristic solution + %s + %s is infeasible' %(optim, opt_preprocess))
                
    return mdl_learning, number_model, total_inf_time

method_number=7 # number of methods
step_cost_list_stack = []
runtime_episode = []
opt_gap_episode = []
cost_episode = []

with torch.no_grad():
    
    k = 1 # counts the number of data points

    while time.time() < start_loop + timeout :

        j = 0 # counts the time step in the episode
            
        Env.set_randState(d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot)
        idx_cntr_init = Env.idx_cntr
        idx_cntr = Env.idx_cntr
        
        Env_milp.copyEnv(Env)
        Env_learning_lp.copyEnv(Env)
        
        Env_minlp_net_ws_10m.copyEnv(Env)
        Env_minlp_ws.copyEnv(Env)
        Env_milp.copyEnv(Env)
        Env_learning_nlp.copyEnv(Env)
        Env_learning_lp.copyEnv(Env)
        Env_learning_nlp_original.copyEnv(Env)
        Env_learning_lp_original.copyEnv(Env)
        
        dict_cntr_infeas['cntr_infeas_episode_learning_nlp'] = 0
        dict_cntr_infeas['cntr_infeas_episode_learning_lp'] = 0
        dict_cntr_infeas['cntr_infeas_episode_learning_nlp_original'] = 0
        dict_cntr_infeas['cntr_infeas_episode_learning_lp_original'] = 0
        
        cost_episode_milp = 0         
        cost_episode_minlp_net_ws_10m = 0
        cost_episode_minlp_ws = 0
        cost_episode_learning_lp = 0       
        cost_episode_learning_nlp = 0
        cost_episode_learning_lp_original = 0       
        cost_episode_learning_nlp_original = 0
        
        runtime_episode_minlp_net_ws_10m = 0                
        runtime_episode_minlp_ws = 0        
        runtime_episode_milp = 0
        runtime_episode_learning_lp = 0
        runtime_episode_learning_nlp = 0
        runtime_episode_learning_lp_original = 0
        runtime_episode_learning_nlp_original = 0
                    
        step_cost_list_stack.append(np.zeros((method_number,80,1)))
        
        flag_end_episode = False
        
        while (Env_learning_lp.terminated or Env_learning_lp.truncated or Env_learning_nlp.terminated or Env_learning_nlp.truncated or Env_learning_lp_original.terminated or Env_learning_lp_original.truncated or Env_learning_nlp_original.terminated or Env_learning_nlp_original.truncated)==False and j < 80:
            
            print('\nnumber_points=%d, time_step_episode=%d' %(k,j))
            
            # minlp (without early termination) - minlp_net_ws_10m
            # this is the reference
            start_time = time.time()
            early_term = 0
            warm_start = 1
            info = Env_minlp_net_ws_10m.step(0, d_pre, rho_whole, mipgap, log, timelimit_reference, early_term, warm_start, n_threads, opt='minlp')[-1]
            mdl_reference = info['mdl']
            
            # minlp (with early termination) - minlp_ws
            start_time = time.time()
            early_term = 1
            warm_start = 1
            info = Env_minlp_ws.step(0, d_pre, rho_whole, mipgap, log, timelimit_gurobi, early_term, warm_start, n_threads, opt='minlp')[-1]
            mdl_minlp_ws = info['mdl']
            
            # milp
            start_time = time.time()
            early_term = 0
            warm_start = 0
            info = Env_milp.step(0, d_pre, rho_whole, mipgap, log, timelimit_gurobi, early_term, warm_start, n_threads, opt='milp')[-1]
            mdl_milp = info['mdl']
                    
            # learning + lp 
            mdl_learning_lp, number_model_lp, total_inf_time_lp = step_Env_learning(Env_learning_lp, 'lp', opt_preprocess='reduced')
            
            # learning + nlp
            mdl_learning_nlp, number_model_nlp, total_inf_time_nlp = step_Env_learning(Env_learning_nlp, 'nlp', opt_preprocess='reduced')
            
            # learning + lp (original)
            mdl_learning_lp_original, number_model_lp_original, total_inf_time_lp_original = step_Env_learning(Env_learning_lp_original, 'lp', opt_preprocess='original')
            
            # learning + nlp (original)
            mdl_learning_nlp_original, number_model_nlp_original, total_inf_time_nlp_original = step_Env_learning(Env_learning_nlp_original, 'nlp', opt_preprocess='original')
                        
            if (Env_learning_nlp.terminated or Env_learning_nlp.truncated)==True or (Env_learning_lp.terminated or Env_learning_lp.truncated)==True or (Env_learning_nlp_original.terminated or Env_learning_nlp_original.truncated)==True or (Env_learning_lp_original.terminated or Env_learning_lp_original.truncated)==True or j >= 80:
                flag_end_episode = True
                    
            if flag_end_episode==False:
                
                # minlp_net_ws_10m (reference)
                cost_reference = store_data(Env_minlp_net_ws_10m, list_minlp_net_ws_10m, mdl_reference, 1, 0)
                cost_episode_minlp_net_ws_10m += cost_reference
                runtime_episode_minlp_net_ws_10m += mdl_reference._Runtime
                
                # minlp_ws
                cost_minlp_ws = store_data(Env_minlp_ws, list_minlp_ws, mdl_minlp_ws, 1, cost_reference)
                cost_episode_minlp_ws += cost_minlp_ws
                runtime_episode_minlp_ws += mdl_minlp_ws._Runtime
                
                # milp
                cost_milp = store_data(Env_milp, list_milp, mdl_milp, 0, cost_reference)
                cost_episode_milp += cost_milp
                runtime_episode_milp += mdl_milp.Runtime
                
                # learning approaches
                cost_learning_lp = store_data(Env_learning_lp, list_learning_lp, mdl_learning_lp, warm_start, cost_reference, total_inf_time_lp, number_model_lp)
                cost_episode_learning_lp += cost_learning_lp
                runtime_episode_learning_lp += mdl_learning_lp.Runtime + total_inf_time_lp
                print('infeasibility (lp) (episode): %d out of %d, \t infeasibility (total): %d out of %d, number_model %d'%(dict_cntr_infeas['cntr_infeas_episode_learning_lp'], j+1, dict_cntr_infeas['cntr_infeas_total_learning_lp'], k, number_model_lp))
                
                cost_learning_nlp = store_data(Env_learning_nlp, list_learning_nlp, mdl_learning_nlp, warm_start, cost_reference, total_inf_time_nlp, number_model_nlp)  
                cost_episode_learning_nlp += cost_learning_nlp
                runtime_episode_learning_nlp += mdl_learning_nlp.Runtime + total_inf_time_nlp
                print('infeasibility (nlp) (episode): %d out of %d, \t infeasibility (total): %d out of %d number_model %d'%(dict_cntr_infeas['cntr_infeas_episode_learning_nlp'], j+1, dict_cntr_infeas['cntr_infeas_total_learning_nlp'], k, number_model_nlp))
                
                cost_learning_lp_original = store_data(Env_learning_lp_original, list_learning_lp_original, mdl_learning_lp_original, warm_start, cost_reference, total_inf_time_lp_original, number_model_lp_original)
                cost_episode_learning_lp_original += cost_learning_lp_original
                runtime_episode_learning_lp_original += mdl_learning_lp_original.Runtime + total_inf_time_lp_original
                print('infeasibility (lp) (episode): %d out of %d, \t infeasibility (total): %d out of %d, number_model %d'%(dict_cntr_infeas['cntr_infeas_episode_learning_lp_original'], j+1, dict_cntr_infeas['cntr_infeas_total_learning_lp_original'], k, number_model_lp_original))
                
                cost_learning_nlp_original = store_data(Env_learning_nlp_original, list_learning_nlp_original, mdl_learning_nlp_original, warm_start, cost_reference, total_inf_time_nlp_original, number_model_nlp_original)  
                cost_episode_learning_nlp_original += cost_learning_nlp_original
                runtime_episode_learning_nlp_original += mdl_learning_nlp_original.Runtime + total_inf_time_nlp_original
                print('infeasibility (nlp) (episode): %d out of %d, \t infeasibility (total): %d out of %d number_model %d'%(dict_cntr_infeas['cntr_infeas_episode_learning_nlp_original'], j+1, dict_cntr_infeas['cntr_infeas_total_learning_nlp_original'], k, number_model_nlp_original))   
                
                step_cost_list_stack[-1][0,j,0] = cost_reference
                step_cost_list_stack[-1][1,j,0] = cost_minlp_ws
                step_cost_list_stack[-1][2,j,0] = cost_milp
                step_cost_list_stack[-1][3,j,0] = cost_learning_lp
                step_cost_list_stack[-1][4,j,0] = cost_learning_nlp
                step_cost_list_stack[-1][5,j,0] = cost_learning_lp_original
                step_cost_list_stack[-1][6,j,0] = cost_learning_nlp_original       
                                
                opt_gap_ref = 0
                opt_gap_minlp_ws = get_gap(cost_episode_minlp_ws, cost_episode_minlp_net_ws_10m)
                opt_gap_milp = get_gap(cost_episode_milp, cost_episode_minlp_net_ws_10m)
                opt_gap_learning_lp = get_gap(cost_episode_learning_lp, cost_episode_minlp_net_ws_10m)
                opt_gap_learning_nlp = get_gap(cost_episode_learning_nlp, cost_episode_minlp_net_ws_10m)
                opt_gap_learning_lp_original = get_gap(cost_episode_learning_lp_original, cost_episode_minlp_net_ws_10m)
                opt_gap_learning_nlp_original = get_gap(cost_episode_learning_nlp_original, cost_episode_minlp_net_ws_10m)                
                
                print('episode costs: \t ref %.2f minlp_ws %.2f milp %.2f learn_lp %.2f learn_nlp %.2f learn_lp_ori %.2f learn_nlp_ori %.2f' %(cost_episode_minlp_net_ws_10m, cost_episode_minlp_ws, cost_episode_milp, cost_episode_learning_lp, cost_episode_learning_nlp, cost_episode_learning_lp_original, cost_episode_learning_nlp_original))                
                print('episode opt_gaps: \t minlp_ws %.2f milp %.2f learn_lp %.2f learn_nlp %.2f learn_lp_ori %.2f learn_nlp_ori %.2f' %(opt_gap_minlp_ws, opt_gap_milp, opt_gap_learning_lp, opt_gap_learning_nlp, opt_gap_learning_lp_original, opt_gap_learning_nlp_original))                
                print('episode runtimes: \t minlp_ws %.2f milp %.2f learn_lp %.2f learn_nlp %.2f learn_lp_ori %.2f learn_nlp_ori %.2f' %(runtime_episode_minlp_ws, runtime_episode_milp, runtime_episode_learning_lp, runtime_episode_learning_nlp, runtime_episode_learning_lp_original, runtime_episode_learning_nlp_original))
                
                print('elapsed_time = %.2f' %(time.time()-start_loop))
                
                j += 1
                k +=1
                idx_cntr +=1              
            
            #store data           
            if flag_end_episode == True:
                
                cost_episode.append([cost_episode_minlp_net_ws_10m, cost_episode_minlp_ws, cost_episode_milp, cost_episode_learning_lp, cost_episode_learning_nlp, cost_episode_learning_lp_original, cost_episode_learning_nlp_original])
                opt_gap_episode.append([opt_gap_ref, opt_gap_minlp_ws, opt_gap_milp, opt_gap_learning_lp, opt_gap_learning_nlp, opt_gap_learning_lp_original, opt_gap_learning_nlp_original])
                runtime_episode.append([runtime_episode_minlp_net_ws_10m, runtime_episode_minlp_ws, runtime_episode_milp, runtime_episode_learning_lp, runtime_episode_learning_nlp, runtime_episode_learning_lp_original, runtime_episode_learning_nlp_original])
                
                print('end of episode')
                
                print(idx_cntr_init, Env_milp.idx_cntr, Env_milp.idx_cntr-idx_cntr_init)           
                
                break
            
            if time.time() > start_loop + timeout:
                break      
                        
        print('\n')   

#----------------------------------------------------------------------------------------------
dict_arrays={}

array_minlp_net_ws_10m = np.array(list_minlp_net_ws_10m)
array_minlp_ws = np.array(list_minlp_ws)
array_milp = np.array(list_milp)
array_learning_lp = np.array(list_learning_lp)
array_learning_nlp = np.array(list_learning_nlp)
array_learning_lp_original = np.array(list_learning_lp_original)
array_learning_nlp_original = np.array(list_learning_nlp_original)

array_step_cost = np.array(step_cost_list_stack)
array_runtime_episode = np.array(runtime_episode)
array_opt_gap_episode = np.array(opt_gap_episode)
array_cost_episode = np.array(cost_episode)

dict_arrays = {
    'array_minlp_net_ws_10m': array_minlp_net_ws_10m,
    'array_minlp_ws': array_minlp_ws,
    'array_milp': array_milp,
    'array_learning_lp': array_learning_lp,
    'array_learning_nlp': array_learning_nlp,
    'array_learning_lp_original': array_learning_lp_original,
    'array_learning_nlp_original': array_learning_nlp_original,     
    'cntr_infeas_total_learning_nlp': dict_cntr_infeas['cntr_infeas_total_learning_nlp'],
    'cntr_infeas_total_learning_lp': dict_cntr_infeas['cntr_infeas_total_learning_lp'],
    'cntr_infeas_total_heuristic_nlp': dict_cntr_infeas['cntr_infeas_total_heuristic_nlp'],
    'cntr_infeas_total_heuristic_lp': dict_cntr_infeas['cntr_infeas_total_heuristic_lp'],
    'cntr_infeas_total_learning_nlp_original': dict_cntr_infeas['cntr_infeas_total_learning_nlp_original'],
    'cntr_infeas_total_learning_lp_original': dict_cntr_infeas['cntr_infeas_total_learning_lp_original'],
    'cntr_infeas_total_heuristic_nlp_original': dict_cntr_infeas['cntr_infeas_total_heuristic_nlp_original'],
    'cntr_infeas_total_heuristic_lp_original': dict_cntr_infeas['cntr_infeas_total_heuristic_lp_original'],
    'array_step_cost': array_step_cost,
    'array_runtime_episode': array_runtime_episode,
    'array_opt_gap_episode': array_opt_gap_episode,
    'array_cost_episode': array_cost_episode,
    'info': 'Each array has a list containing [mdl.Runtime, cost_minlp, perc_minlp_cost, mdl.status]. The arrays learning_nlp, learning_nlp have extra elements [inference_time, number_model]. \n Arrays for the episodes have the total [cost, opt_gap, runtime] for each episode.',
    'info_step_cost': '[episode, method, step, 0]',
    # 'info_episode': 'Each array has a list containing [d_real, a_real, l_real, n_real, n_after_real, n_before_real, idx_cntr, step cost, step optimality gap]. \nd_real = [episode, method, data, data, data]. \nidx_cntr = [episode, 0:2]; 0: idx_cntr_init, 1: idx_cntr_end, 2: episode length \n step_cost/opt_gap/runtime = [episode, method, episode number] ',
    'info (runtime and opt_gap episodes)': '[episode number, method] 0: reference, 1: minlp_ws_et, 2: milp, 3: learning_lp, 4: learning_nlp, 5: learning_lp_original, 6: learning_nlp_original'
}

if testing == True:
    os.makedirs('tests_ensemble_all', exist_ok=True) 
    np.save('tests_ensemble_all//test_ensemble_all_%.3d.npy' %(job_idx), dict_arrays)
else:
    os.makedirs('//scratch//cfoliveiradasi//railway_conf//tests_ensemble_all', exist_ok=True)
    np.save('//scratch/cfoliveiradasi//railway_conf//tests_ensemble_all//test_ensemble_all_%.3d.npy' %(job_idx), dict_arrays)
    
if testing==True:
    print('peak RAM usage: %.2f' %p.memory_info().peak_wset)
    print('current RAM usage: %.2f' %p.memory_info().rss)
else:   
    pass

x = datetime.datetime.now()
date_str = '%.2d%.2d_%.2d%.2d' %(x.month, x.day, x.hour, x.minute)
print('date_time: ' + date_str)

print('total time: %f' %(time.time()-start_loop))
print('test completed!')
#-----------------------------------------
