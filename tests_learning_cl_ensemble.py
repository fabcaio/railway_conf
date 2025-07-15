from rail_fun import cost_per_step, build_delta_vector, get_state_learning
import numpy as np
from rail_rl_env import RailNet, d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot, mdl_feasible, action_dict, gurobi_minlp, gurobi_nlp_presolve, gurobi_lp_presolve, gurobi_milp
import time
from rail_data_preprocess_reduced import get_dataset_min_max
from rail_training_reduced import Network, Network_CE, Network_mask1, Network_mask2, Network_mask3
import torch
import sys
import datetime

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
testing = False

opt_label = 'classification' #(regression is not implemented yet)

if testing == True:
    import psutil
    p = psutil.Process()
    opt_data = 'milp_cl'
    idx = 1
    n_threads = 8
    timeout = 2*60 #minutes
    timelimit_gurobi = 30
    timelimit_reference = 30 # sets the time for the longest milp (with warm-start)
else:
    opt_data = sys.argv[1]
    n_threads = int(sys.argv[2])
    timelimit_job = int(sys.argv[3]) #(in seconds)
    # idx = int(sys.argv[4])
    timelimit_gurobi = 60 # (in seconds, for minlp and milp)
    timeout = timelimit_job - 30*60
    
device='cpu'
torch.set_num_threads(n_threads)

num_layers=1
lr=1e-3
batch_size=1 #
threshold_counts = 50

mipgap = 1e-3
log = 0
early_term = 1

print('N: %d \t mipgap: %.4f \t timelimit_gurobi: %.2f \t n_threads: %d \t early_term: %d' %(N, mipgap, timelimit_gurobi, n_threads, early_term))
print('opt_data: ' + opt_data + '\t threshold counts: %d' %threshold_counts)
print('timeout: %.2f,\t' %timeout, 'mipgap %.4f' %mipgap)
    
dict_data = get_dataset_min_max(opt_data, threshold_counts, N, testing)

N_control = dict_data['N_control']
list_masks = dict_data['list_masks']
total_action_set = dict_data['total_action_set']

action_dict_reduced = {}
for i in total_action_set:
    action_dict_reduced[str(i)] = action_dict[str(i)]

num_actions = total_action_set.shape[0]
seq_len=N_control

seed = 4
np.random.seed(seed)
torch.manual_seed(seed)

idx_best_networks = [51, 17, 21, 55, 49, 23, 19, 53, 61, 29, 25, 27, 57, 59, 31]
num_networks = len(idx_best_networks)

def build_network(network_type, input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout, list_masks=list_masks, device=device):
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

if testing==True:
    hyperpar_set = np.load('training_data_reduced/milp_cl_N40_hyperparm_set.npy', allow_pickle=True).item()
else:
    hyperpar_set = np.load('//scratch/cfoliveiradasi//railway_largescale//training_data_reduced//milp_cl_N40_hyperparm_set.npy', allow_pickle=True).item()

model_list = []
h0 = []
c0 = []
opt_state = []
state_min = []
state_max = []
for idx in idx_best_networks:
    
    if testing==True:
        network_info_path = 'training_data_reduced//' + opt_data + '_N%d_%.3d_' %(N,idx) + 'info.npy'
        network_weight_path = 'training_data_reduced//' + opt_data + '_N%d_%.3d_' %(N,idx) + 'weight'
    else:
        network_info_path = '//scratch/cfoliveiradasi//railway_largescale//training_data_reduced//%s_N%.2d_%.3d_info.npy' %(opt_data, N, idx)
        network_weight_path = '//scratch/cfoliveiradasi//railway_largescale//training_data_reduced//%s_N%.2d_%.3d_weight' %(opt_data, N, idx)
        
    opt_state.append(hyperpar_set[str(idx)][-1]) # opt_state was not saved before, but is solved! (necessary to access 'reduced' existing results)
    state_min.append(dict_data['state_min_%d' %opt_state[-1]])
    state_max.append(dict_data['state_max_%d' %opt_state[-1]])
        
    network_info = np.load(network_info_path, allow_pickle=True).item()    
    network_type = network_info['network_type']
    input_size = network_info['input_size']
    hidden_size = network_info['hidden_size']
    num_layers = network_info['num_layers']
    lr = network_info['lr']
    dropout = network_info['dropout']
    LR_scheduler = network_info['LR_scheduler']
    
    network = build_network(network_type, input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout)
    network.load_state_dict(torch.load(network_weight_path, weights_only=True))
    
    h0.append(torch.zeros(num_layers, 1, hidden_size))
    c0.append(torch.zeros(num_layers, 1, hidden_size))
    
    model_list.append(network)


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
        
    if Env.name == 'milp': # this is the reference
        perc_minlp_cost = 0
    else:
        perc_minlp_cost = (cost-cost_reference)/cost_reference*100
        
    # if Env.name == 'learning_nlp' or Env.name == 'learning_lp':
    #     list_data.append([runtime, cost, perc_minlp_cost, mdl.status])
    # else:
    #     list_data.append([runtime, cost, perc_minlp_cost, mdl.status])
        
    list_data.append([runtime, cost, perc_minlp_cost, mdl.status])       
    
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

Env = RailNet(N)
Env_milp = RailNet(N)
Env_milp.name = 'milp'
Env_learning_lp = RailNet(N)
Env_learning_lp.name = 'learning_lp'

cntr_infeas_total_learning_lp = 0
cntr_infeas_episode_learning_lp = 0
cntr_infeas_total_heuristic_lp = 0

method_number=1 # number of methods
opt_gap_stack = np.zeros((method_number,30,1))
opt_gap_list_stack = []

with torch.no_grad():
    
    k = 1 # counts the number of data points

    while time.time() < start_loop + timeout :

        j = 0 # counts the time step in the episode
            
        Env.set_randState(d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot)
        idx_cntr_init = Env.idx_cntr
        idx_cntr = Env.idx_cntr
        
        Env_milp.copyEnv(Env)
        Env_learning_lp.copyEnv(Env)
        
        cntr_infeas_episode_learning_nlp = 0
        cntr_infeas_episode_learning_lp = 0
        
        cost_episode_milp = 0
        cost_episode_learning_lp = 0
        
        runtime_episode_milp = 0
        runtime_episode_learning_lp = 0
        
        opt_gap_list_stack.append(np.zeros((method_number,30,1)))
        
        flag_end_episode = False
        
        while (Env_learning_lp.terminated or Env_learning_lp.truncated)==False and j < 30:
            
            print('\nnumber_points=%d, time_step_episode=%d' %(k,j))
            
            # milp
            start_time = time.time()
            early_term = 0
            warm_start = 0
            info = Env_milp.step(0, d_pre, rho_whole, mipgap, log, timelimit_gurobi, early_term, warm_start, n_threads, opt='milp')[-1]
            mdl_milp = info['mdl']
                    
            # learning + lp       
            
            for i in range(num_networks):
                start_time=time.time()
                total_inf_time=0
                early_term = 1
                warm_start = 0
                opt_time = timelimit_gurobi
                
                state_learning = get_state_learning(Env_learning_lp, opt_state[i], state_min[i], state_max[i], N_control)
                
                start_inf_time = time.time()
                output_net = model_list[i](state_learning, h0[i], c0[i])
                action_idx = total_action_set[torch.max(output_net, dim=2)[1].squeeze().numpy()]
                delta_SL = build_delta_vector(action_idx, N_control, action_dict_reduced)
                inf_time = time.time()-start_inf_time
            
                total_inf_time += inf_time
                opt_time = opt_time - inf_time
            
                a_values, d_values, r_values, l_values, y_values, n_values, n_after_values,  mdl_learning_lp = gurobi_lp_presolve(N, Env_learning_lp.d_pre_cut, Env_learning_lp.state_rho, Env_learning_lp.state_a, Env_learning_lp.state_d, Env_learning_lp.state_r, Env_learning_lp.state_l, Env_learning_lp.state_y, Env_learning_lp.state_n, Env_learning_lp.state_depot, delta_SL, mipgap, log, opt_time, n_threads)
                
                number_model = i+1
                
                if mdl_feasible(mdl_learning_lp)==True:
                        break
            
            if mdl_feasible(mdl_learning_lp)==True:
                info = Env_learning_lp.step(delta_SL, d_pre, rho_whole, mipgap, log, opt_time, early_term, warm_start, n_threads, opt='lp')[-1]
                current_delta = delta_SL            
            
            else:
                print('learning + lp: not feasible, model_number: %d' %(number_model))
                cntr_infeas_total_learning_lp +=1
                cntr_infeas_episode_learning_lp +=1

                # add by Xiaoyu_20241120
                for line in range(3):
                    if current_delta[N_control - 2, 1 + 4 * line] == 0:
                        match_lp[line] = match_constant[line]
                    else:
                        match_lp[line] = match_constant[line] + 1
            
                # add by Xiaoyu
                delta_recursive = np.zeros([N_control, 12], dtype=int)
                delta_recursive[:-1, :] = current_delta[1:, :]
                for line in range(3):
                    delta_recursive[N_control - 1, 0 + 4 * line] = current_delta[N_control - 2, 0 + 4 * line]
                    delta_recursive[N_control - 1, 1 + 4 * line] = current_delta[N_control - 2, 1 + 4 * line]
                    delta_recursive[N_control - 1, 2 + 4 * line] = current_delta[N_control - 1 - match_lp[line], 2 + 4 * line]
                    delta_recursive[N_control - 1, 3 + 4 * line] = current_delta[N_control - 1 - match_lp[line], 3 + 4 * line]
                    
                a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl_learning_lp = gurobi_lp_presolve(N, Env_learning_lp.d_pre_cut, Env_learning_lp.state_rho, Env_learning_lp.state_a,Env_learning_lp.state_d, Env_learning_lp.state_r, Env_learning_lp.state_l, Env_learning_lp.state_y,Env_learning_lp.state_n, Env_learning_lp.state_depot, delta_recursive, mipgap, log, opt_time, n_threads)

                # add by Xiaoyu_20241120
                if mdl_feasible(mdl_learning_lp) == True:
                    info = Env_learning_lp.step(delta_recursive, d_pre, rho_whole, mipgap, log, opt_time, early_term, warm_start, n_threads, opt='lp')[-1]
                    current_delta = delta_recursive
                else:
                    print('recursive solution (lp) is infeasible')
                    # add by Xiaoyu_20241120
                    a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl_learning_lp = gurobi_lp_presolve(N, Env_learning_lp.d_pre_cut, Env_learning_lp.state_rho, Env_learning_lp.state_a,Env_learning_lp.state_d, Env_learning_lp.state_r, Env_learning_lp.state_l, Env_learning_lp.state_y, Env_learning_lp.state_n, Env_learning_lp.state_depot, stacked_delta_heuristic1, mipgap, log, opt_time, n_threads)

                    if mdl_feasible(mdl_learning_lp) == True:
                        # cost_learning_nlp = store_data(Env_learning_nlp, list_learning_nlp, mdl, warm_start, cost_reference, total_inf_time, number_model)
                        info = Env_learning_lp.step(stacked_delta_heuristic1, d_pre, rho_whole, mipgap, log, opt_time, early_term, warm_start, n_threads, opt='lp')[-1]
                        current_delta = stacked_delta_heuristic1
                    else:
                        cntr_infeas_total_heuristic_lp += 1
                        flag_end_episode = True
                        print('heuristic solution (lp) is infeasible')
                    
            if (Env_learning_lp.terminated or Env_learning_lp.truncated)==True or j >= 30:
                flag_end_episode = True
                    
            if flag_end_episode==False:
                
                # milp
                cost_reference = store_data(Env_milp, list_milp, mdl_milp, 0, 0)
                cost_episode_milp += cost_reference
                runtime_episode_milp += mdl_milp.Runtime
                
                # learning approaches
                cost_learning_lp = store_data(Env_learning_lp, list_learning_lp, mdl_learning_lp, warm_start, cost_reference, total_inf_time)
                print('infeasibility (lp) (episode): %d out of %d, \t infeasibility (total): %d out of %d, number_model %d'%(cntr_infeas_episode_learning_lp, j+1, cntr_infeas_total_learning_lp, k, number_model))               
                
                cost_episode_learning_lp += cost_learning_lp
                runtime_episode_learning_lp += mdl_learning_lp.Runtime + total_inf_time                
                
                opt_gap_list_stack[-1][0,j,0] = get_gap(cost_episode_learning_lp, cost_episode_milp)
                
                print('episode costs: \t milp %.2f learn_lp %.2f' %(cost_episode_milp, cost_episode_learning_lp))
                
                print('episode opt_gaps: \t learn_lp %.2f' %(get_gap(cost_episode_learning_lp, cost_episode_milp)))
                
                print('episode runtimes: \t milp %.2f learn_lp %.2f' %(runtime_episode_milp, runtime_episode_learning_lp))
                
                print('elapsed_time = %.2f' %(time.time()-start_loop))
                
                j += 1
                k +=1
                idx_cntr +=1              
            
            #store data           
            if flag_end_episode == True:
                
                print('end of episode')
                
                print(idx_cntr_init, Env_milp.idx_cntr, Env_milp.idx_cntr-idx_cntr_init)           
                
                break
            
            if time.time() > start_loop + timeout:
                break      
                        
        print('\n')   

#----------------------------------------------------------------------------------------------
dict_arrays={}

array_milp = np.array(list_milp)
array_learning_lp = np.array(list_learning_lp)
array_opt_gap = np.array(opt_gap_list_stack)

dict_arrays = {
    'array_milp': array_milp,
    'array_learning_lp': array_learning_lp,
    'cntr_infeas_total_learning_lp': cntr_infeas_total_learning_lp,
    'cntr_infeas_total_heuristic_lp': cntr_infeas_total_heuristic_lp,
    'array_opt_gap': array_opt_gap,
    'info': 'Each array has a list containing [mdl.Runtime, cost_minlp, perc_minlp_cost, mdl.status]. The arrays learning_nlp, learning_nlp have extra elements [inference_time, number_model]. \n Arrays for the episodes have the total [cost, opt_gap, runtime] for each episode.',
    'info_episode': 'Each array has a list containing [d_real, a_real, l_real, n_real, n_after_real, n_before_real, idx_cntr, step cost, step optimality gap]. \nd_real = [episode, method, data, data, data]. \nidx_cntr = [episode, 0:2]; 0: idx_cntr_init, 1: idx_cntr_end, 2: episode length \n step_cost/opt_gap/runtime = [episode, method, episode number] ',
    'info_env_status': '0: nlp.terminated, 1: nlp.truncated, 2: lp.terminated, 3: lp.truncated'
}

if testing == True:
    np.save('tests_ensemble//tests_ensemble_%.3d.npy', dict_arrays)
else:
    np.save('//scratch/cfoliveiradasi//railway_largescale//tests_ensemble//test_ensemble.npy', dict_arrays)
    
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
