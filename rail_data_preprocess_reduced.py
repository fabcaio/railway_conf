import numpy as np
import torch

from rail_fun import norm_state, preprocess_state, inv_action_dict

#tests conversion from original list_actions to reduced list of actions
def reduce_list_actions(list_action, total_action_set, N_control):
    
    list_actions_reduced = np.zeros((N_control,), dtype=np.int32)
    for i in range(N_control):
        list_actions_reduced[i] = np.where(list_action[i] == total_action_set)[0][0]
        
    return list_actions_reduced

def split_train_val_test(stacked_array, val_split=0.8):
    
    # 80/10/10 split
    
    N_datapoints = stacked_array.shape[0]
    test_split = (1-(1-val_split)/2)
    
    stacked_array_train = stacked_array[:int(np.ceil(N_datapoints*val_split))]
    
    stacked_array_val = stacked_array[int(np.ceil(N_datapoints*val_split)):int(np.ceil(N_datapoints*test_split))]
    
    stacked_array_test = stacked_array[int(np.ceil(N_datapoints*test_split)):]
    
    return stacked_array_train, stacked_array_val, stacked_array_test

def reduce_Dataset(stacked_states, stacked_actions, total_action_set, N_control):
    
        """
        It removes from the dataset the state-actions pairs that are removed by thresholding, which is used to remove unlikely state-action pairs; thus reducing the output space and the complexity of the neural network.
        """
        
        stacked_states_reduced = []
        stacked_actions_reduced = []
        cntr_outlier = 0
        for i in range(stacked_actions.shape[0]):
            try:
                stacked_actions_reduced.append(reduce_list_actions(stacked_actions[i], total_action_set, N_control))
                stacked_states_reduced.append(stacked_states[i])
            except:
                cntr_outlier +=1
                
        return stacked_states_reduced, stacked_actions_reduced, cntr_outlier

######################################################################################################################

def get_preprocessed_data_reduced(opt_data, opt_preprocess, threshold_counts, N, opt_state, opt_label, testing):
    
    """
    opt_data:
        'milp_ol', 'milp_cl'
        
    opt_preprocess:
        'reduced', 'original'
        
    threshold counts:
        eliminates unsual data (that appears below this count)
        
    N:
        prediction horizon
           
    opt_state: determines state pre-processing
        1: l_0 + mean
        2: l_{0,1,2} + mean
        3: l_0 + downsampling
        4: l_{0,1,2} + downsampling
        
    opt_label: defines training loss
        'classification' or 'regression' (not implemented yet)
        
    testing:
        True: code executed in local machine
        False: code executed in server
        
    output: dict_data
        dictionary with training, validation, and training losses
        note that additional pre-processing is still done during training (to save RAM memory)
    
    """
    
    N_control = N - 2
            
    if testing==True:
        str_data = 'data_optimal_%s//data_%s_%s_N%d_%.3d.npy' %(opt_preprocess, opt_preprocess, opt_data, N, 0)
    elif testing==False:
        str_data = '//scratch//cfoliveiradasi//railway_conf//data_optimal_%s//data_%s_%s_N%d_%.3d.npy' %(opt_preprocess, opt_preprocess, opt_data, N, 0)
    array_data = np.load(str_data, allow_pickle=True)
    dict_state_list = array_data[0]
    dict_output_list = array_data[1]
    
    state_n = dict_state_list['state_n']
    state_depot = dict_state_list['state_depot']
    idx_cntr = dict_state_list['idx_cntr']
    state_l_0= dict_state_list['state_l_0']
    state_l_1= dict_state_list['state_l_1']
    state_l_2= dict_state_list['state_l_2']
    
    if opt_preprocess == 'reduced':
        state_rho_down= dict_state_list['state_rho_down']
        state_rho_mean= dict_state_list['state_rho_mean']
        n_datasets = 35
    elif opt_preprocess == 'original':
        idx_group = dict_state_list['idx_group']
        state_rho = dict_state_list['state_rho']
        n_datasets = 13
    
    if opt_label=='classification':
        stacked_list_actions = dict_output_list['list_actions']
    elif opt_label=='regression':
        stacked_list_actions = dict_output_list['delta']
         
    for i in range(1,n_datasets):
        if testing==True:
            str_data = 'data_optimal_%s//data_%s_%s_N%d_%.3d.npy' %(opt_preprocess, opt_preprocess, opt_data, N, i)
        elif testing==False:
            str_data = '//scratch//cfoliveiradasi//railway_conf//data_optimal_%s//data_%s_%s_N%d_%.3d.npy' %(opt_preprocess, opt_preprocess, opt_data, N, i)
        array_data = np.load(str_data, allow_pickle=True)
        dict_state_list = array_data[0]
        dict_output_list = array_data[1]
        
        state_n = np.concatenate((state_n, dict_state_list['state_n']))
        state_depot = np.concatenate((state_depot, dict_state_list['state_depot']))
        idx_cntr = np.concatenate((idx_cntr, dict_state_list['idx_cntr']))
        state_l_0= np.concatenate((state_l_0, dict_state_list['state_l_0']))
        state_l_1= np.concatenate((state_l_1, dict_state_list['state_l_1']))
        state_l_2= np.concatenate((state_l_2, dict_state_list['state_l_2']))
        
        if opt_preprocess == 'reduced':
            state_rho_down= np.concatenate((state_rho_down, dict_state_list['state_rho_down']))
            state_rho_mean= np.concatenate((state_rho_mean, dict_state_list['state_rho_mean']))
        elif opt_preprocess == 'original':
            idx_group = np.concatenate((idx_group, dict_state_list['idx_group']))
            state_rho = np.concatenate((state_rho, dict_state_list['state_rho']))
        
        if opt_label=='classification':
            tmp = dict_output_list['list_actions']
        elif opt_label=='regression':
            tmp = dict_output_list['delta']    
        stacked_list_actions = np.concatenate((stacked_list_actions, tmp))
    
    if opt_preprocess == 'reduced':
        if opt_state==1:
            stacked_states = np.concatenate((state_n, state_depot, idx_cntr, state_rho_mean, state_l_0), axis=1)
        elif opt_state==2:
            stacked_states = np.concatenate((state_n, state_depot, idx_cntr, state_rho_mean, state_l_0, state_l_1, state_l_2), axis=1)
        elif opt_state==3:
            stacked_states = np.concatenate((state_n, state_depot, idx_cntr, state_rho_down, state_l_0), axis=1)
        elif opt_state==4:
            stacked_states = np.concatenate((state_n, state_depot, idx_cntr, state_rho_down, state_l_0, state_l_1, state_l_2), axis=1)
    if opt_preprocess == 'original':
        stacked_states = np.concatenate((state_n, state_depot, idx_cntr, state_rho, state_l_0, state_l_1, state_l_2), axis=1)
    
    # cost_list = dict_output_list['mdl_Obj']
    # mipgap_list = dict_output_list['mdl_mipgap']
    # runtime_list = dict_output_list['mdl_runtime']
    # status_list = dict_output_list['mdl_status']
    
    state_min = np.min(stacked_states, axis=0)
    state_max = np.max(stacked_states, axis=0)
    stacked_states = norm_state(stacked_states, state_min, state_max)
    
    input_size = stacked_states.shape[1]

    # train/val/test split
    
    stacked_states_train, stacked_states_val, stacked_states_test = split_train_val_test(stacked_states, val_split=0.8)
    stacked_actions_train, stacked_actions_val, stacked_actions_test = split_train_val_test(stacked_list_actions, val_split=0.8)
    
    # cost_list_train, cost_list_val, cost_list_test = split_train_val_test(cost_list, val_split=0.8)
    # mipgap_list_train, mipgap_list_val, mipgap_list_test = split_train_val_test(mipgap_list, val_split=0.8)
    # runtime_list_train, runtime_list_val, runtime_list_test = split_train_val_test(runtime_list, val_split=0.8)
    # status_list_train, status_list_val, status_list_test = split_train_val_test(status_list, val_split=0.8)
    
    if opt_label == 'classification':
        #reduces the set of actions (subset of all optimal actions)
        action_set, counts = np.unique(stacked_list_actions[:,0], return_counts=True)
        action_set = action_set[np.where(counts>threshold_counts)]
        list_action_set = [action_set.astype(np.int32)]
        total_action_set = action_set.astype(np.int32)

        for i in range(1,N_control):
            action_set, counts = np.unique(stacked_list_actions[:,i], return_counts=True)
            action_set = action_set[np.where(counts>threshold_counts)]
            total_action_set = np.union1d(total_action_set, action_set.astype(np.int32))
            list_action_set.append(action_set.astype(np.int32))

        #creates the masking for each step of the control horizon
        list_masks = []

        for j in range(N_control):
            mask = np.zeros(total_action_set.shape, dtype=np.int32)
            for i in range(list_action_set[j].shape[0]):
                mask[np.where(total_action_set==list_action_set[j][i])] = 1
            list_masks.append(mask)
            
        num_actions = total_action_set.shape[0]
        print('num_actions =', num_actions)
        
        stacked_states_reduced_train, stacked_actions_reduced_train, cntr_outlier_train = reduce_Dataset(stacked_states_train, stacked_actions_train, total_action_set, N_control)
        stacked_states_reduced_val, stacked_actions_reduced_val, cntr_outlier_val = reduce_Dataset(stacked_states_val, stacked_actions_val, total_action_set, N_control)
        stacked_states_reduced_test, stacked_actions_reduced_test, cntr_outlier_test = reduce_Dataset(stacked_states_test, stacked_actions_test, total_action_set, N_control)
        
        stacked_actions_reduced_train = np.array(stacked_actions_reduced_train)
        stacked_actions_reduced_val = np.array(stacked_actions_reduced_val)
        stacked_actions_reduced_test = np.array(stacked_actions_reduced_test)
    
        stacked_states_train_tensor = torch.tensor(np.array(stacked_states_reduced_train), dtype=torch.float32)
        stacked_states_val_tensor = torch.tensor(np.array(stacked_states_reduced_val), dtype=torch.float32)
        stacked_states_test_tensor = torch.tensor(np.array(stacked_states_reduced_test), dtype=torch.float32) 
        
        print('number of training points (before reduction): %d' %stacked_actions_train.shape[0])
        print('number of validation points (before reduction): %d' %stacked_actions_val.shape[0])
        print('number of test points (before reduction): %d' %stacked_actions_test.shape[0])
        print('number of training points (after reduction): %d' %stacked_states_train_tensor.shape[0])
        print('number of validation points (after reduction): %d' %stacked_states_val_tensor.shape[0])
        print('number of test points (after reduction): %d' %stacked_states_test_tensor.shape[0])
        print('cntr_outlier_train: %d\t' %cntr_outlier_train, 'cntr_outlier_val: %d' %cntr_outlier_val, 'cntr_outlier_test: %d' %cntr_outlier_test)

    #TODO: finish implementation for regression
    elif opt_label == 'regression':
        list_masks = []
        total_action_set = []
        stacked_states_train_tensor = torch.tensor(np.array(stacked_states_train), dtype=torch.float32)
        stacked_states_val_tensor = torch.tensor(np.array(stacked_states_val), dtype=torch.float32)
        stacked_states_test_tensor = torch.tensor(np.array(stacked_states_test), dtype=torch.float32) 

    # del stacked_actions_train, stacked_actions_val

    print('data-processing finished')
    
    dict_data = {
        'N': N,
        'N_control': N_control,
        'stacked_states_train_tensor': stacked_states_train_tensor,
        'stacked_states_val_tensor': stacked_states_val_tensor,
        'stacked_states_test_tensor': stacked_states_test_tensor,
        'stacked_actions_reduced_train': stacked_actions_reduced_train,
        'stacked_actions_reduced_val': stacked_actions_reduced_val,
        'stacked_actions_reduced_test': stacked_actions_reduced_test,
        'list_masks': list_masks,
        'state_min': state_min,
        'state_max': state_max,
        'input_size': input_size,
        'total_action_set': total_action_set
    }
    
    return dict_data


def get_dataset_min_max(opt_data, opt_preprocess, threshold_counts, N, testing, opt_label='classification'):
    
    """
    opt_data:
        'milp_ol', 'milp_cl'
        
    opt_preprocess:
        'reduced', 'original'
        
    threshold counts:
        eliminates unsual data (that appears below this count)
        
    N:
        prediction horizon
        
    opt_label: defines training loss
        'classification' or 'regression'
        
    testing:
        True: code executed in local machine
        False: code executed in server
        
    output: dict_data
        state_min/max for each opt_state, list_masks, input_size
    
    """
    
    N_control = N - 2
            
    if testing==True:
        str_data = 'data_optimal_%s//data_%s_%s_N%d_%.3d.npy' %(opt_preprocess, opt_preprocess, opt_data, N, 0)
    elif testing==False:
        str_data = '//scratch//cfoliveiradasi//railway_conf//data_optimal_%s//data_%s_%s_N%d_%.3d.npy' %(opt_preprocess, opt_preprocess, opt_data, N, 0)
    array_data = np.load(str_data, allow_pickle=True)
    dict_state_list = array_data[0]
    dict_output_list = array_data[1]
    
    state_n = dict_state_list['state_n']
    state_depot = dict_state_list['state_depot']
    idx_cntr = dict_state_list['idx_cntr']
    state_l_0= dict_state_list['state_l_0']
    state_l_1= dict_state_list['state_l_1']
    state_l_2= dict_state_list['state_l_2']
    
    if opt_preprocess == 'reduced':
        state_rho_down= dict_state_list['state_rho_down']
        state_rho_mean= dict_state_list['state_rho_mean']
        n_datasets = 35
    elif opt_preprocess == 'original':
        idx_group = dict_state_list['idx_group']
        state_rho = dict_state_list['state_rho']
        n_datasets = 13
    
    if opt_label=='classification':
        stacked_list_actions = dict_output_list['list_actions']
    elif opt_label=='regression':
        stacked_list_actions = dict_output_list['delta']     
         
    for i in range(1,n_datasets):
        if testing==True:
            str_data = 'data_optimal_%s//data_%s_%s_N%d_%.3d.npy' %(opt_preprocess, opt_preprocess, opt_data, N, i)
        elif testing==False:
            str_data = '//scratch//cfoliveiradasi//railway_conf//data_optimal_%s//data_%s_%s_N%d_%.3d.npy' %(opt_preprocess, opt_preprocess, opt_data, N, i)
        array_data = np.load(str_data, allow_pickle=True)
        dict_state_list = array_data[0]
        dict_output_list = array_data[1]
        
        state_n = np.concatenate((state_n, dict_state_list['state_n']))
        state_depot = np.concatenate((state_depot, dict_state_list['state_depot']))
        idx_cntr = np.concatenate((idx_cntr, dict_state_list['idx_cntr']))
        state_l_0= np.concatenate((state_l_0, dict_state_list['state_l_0']))
        state_l_1= np.concatenate((state_l_1, dict_state_list['state_l_1']))
        state_l_2= np.concatenate((state_l_2, dict_state_list['state_l_2']))
        
        if opt_preprocess == 'reduced':
            state_rho_down= np.concatenate((state_rho_down, dict_state_list['state_rho_down']))
            state_rho_mean= np.concatenate((state_rho_mean, dict_state_list['state_rho_mean']))
        elif opt_preprocess == 'original':
            idx_group = np.concatenate((idx_group, dict_state_list['idx_group']))
            state_rho = np.concatenate((state_rho, dict_state_list['state_rho']))
        
        if opt_label=='classification':
            tmp = dict_output_list['list_actions']
        elif opt_label=='regression':
            tmp = dict_output_list['delta']    
        stacked_list_actions = np.concatenate((stacked_list_actions, tmp))
    
    if opt_preprocess == 'reduced':
        stacked_states_1 = np.concatenate((state_n, state_depot, idx_cntr, state_rho_mean, state_l_0), axis=1)
        stacked_states_2 = np.concatenate((state_n, state_depot, idx_cntr, state_rho_mean, state_l_0, state_l_1, state_l_2), axis=1)
        stacked_states_3 = np.concatenate((state_n, state_depot, idx_cntr, state_rho_down, state_l_0), axis=1)
        stacked_states_4 = np.concatenate((state_n, state_depot, idx_cntr, state_rho_down, state_l_0, state_l_1, state_l_2), axis=1)
        state_min_1 = np.min(stacked_states_1, axis=0)
        state_max_1 = np.max(stacked_states_1, axis=0)
        state_min_2 = np.min(stacked_states_2, axis=0)
        state_max_2 = np.max(stacked_states_2, axis=0)
        state_min_3 = np.min(stacked_states_3, axis=0)
        state_max_3 = np.max(stacked_states_3, axis=0)
        state_min_4 = np.min(stacked_states_4, axis=0)
        state_max_4 = np.max(stacked_states_4, axis=0)
        input_size_1 = stacked_states_1.shape[1]
        input_size_2 = stacked_states_2.shape[1]
        input_size_3 = stacked_states_3.shape[1]
        input_size_4 = stacked_states_4.shape[1]
    elif opt_preprocess == 'original':
        stacked_states = np.concatenate((state_n, state_depot, idx_cntr, state_rho, state_l_0, state_l_1, state_l_2), axis=1)
        input_size = stacked_states.shape[1]    
        state_min = np.min(stacked_states, axis=0)
        state_max = np.max(stacked_states, axis=0)

    #reduces the set of actions (subset of all optimal actions)
    action_set, counts = np.unique(stacked_list_actions[:,0], return_counts=True)
    action_set = action_set[np.where(counts>threshold_counts)]
    list_action_set = [action_set.astype(np.int32)]
    total_action_set = action_set.astype(np.int32)

    for i in range(1,N_control):
        action_set, counts = np.unique(stacked_list_actions[:,i], return_counts=True)
        action_set = action_set[np.where(counts>threshold_counts)]
        total_action_set = np.union1d(total_action_set, action_set.astype(np.int32))
        list_action_set.append(action_set.astype(np.int32))

    #creates the masking for each step of the control horizon
    list_masks = []

    for j in range(N_control):
        mask = np.zeros(total_action_set.shape, dtype=np.int32)
        for i in range(list_action_set[j].shape[0]):
            mask[np.where(total_action_set==list_action_set[j][i])] = 1
        list_masks.append(mask)
        
    num_actions = total_action_set.shape[0]
    print('num_actions =', num_actions)
    
    if opt_preprocess == 'reduced':
        dict_data = {
            'N': N,
            'N_control': N_control,
            'list_masks': list_masks,
            'state_min_1': state_min_1,
            'state_min_2': state_min_2,
            'state_min_3': state_min_3,
            'state_min_4': state_min_4,
            'state_max_1': state_max_1,
            'state_max_2': state_max_2,
            'state_max_3': state_max_3,
            'state_max_4': state_max_4,
            'input_size_1': input_size_1,
            'input_size_2': input_size_2,
            'input_size_3': input_size_3,
            'input_size_4': input_size_4,
            'total_action_set': total_action_set
        }
    elif opt_preprocess == 'original':
            dict_data = {
            'N': N,
            'N_control': N_control,
            'list_masks': list_masks,
            'state_min': state_min,
            'state_max': state_max,
            'input_size': input_size,
            'total_action_set': total_action_set
        }
    
    return dict_data


################################################################
# imported from old code (journal paper)

def build_list_action(delta: np.array, N_control: np.int32) -> list:
    delta = delta.astype(np.int32)
    list_action = [inv_action_dict[tuple(delta[0])]]
    for i in range(1, N_control):
        list_action.append(inv_action_dict[tuple(delta[i])])
        
    return list_action

def split_train_validation(stacked_states, stacked_list_actions, split_position=0.8):
       
    N_datapoints = stacked_states.shape[0]
    
    stacked_states_train = stacked_states[:int(np.ceil(N_datapoints*split_position))]
    # stacked_labels_train = stacked_labels[:int(np.ceil(N_datapoints*split_position))]
    stacked_actions_train = stacked_list_actions[:int(np.ceil(N_datapoints*split_position))]
    
    stacked_states_val = stacked_states[int(np.ceil(N_datapoints*split_position)):]
    # stacked_labels_val = stacked_labels[int(np.ceil(N_datapoints*split_position)):]
    stacked_actions_val = stacked_list_actions[int(np.ceil(N_datapoints*split_position)):]
    
    return stacked_states_train, stacked_actions_train, stacked_states_val, stacked_actions_val

def decompress_minlp_info(minlp_info_compressed, N_control):
    
    # input minlp_info_compressed[i]
    
    if N_control==18: #(N=20)    
        state_n = minlp_info_compressed[0:3*38].reshape(3,38)    
        state_rho = minlp_info_compressed[3*38 : 3*38 + 3*21*38].reshape(3,21,38)    
        state_depot = minlp_info_compressed[3*38 + 3*21*38 : 3*38 + 3*21*38 + 3].reshape(3,)    
        state_l = minlp_info_compressed[3*38 + 3*21*38 + 3 : 3*38 + 3*21*38 + 3 + 3*3*38].reshape(3,3,38)   
        delta_minlp = minlp_info_compressed[3*38 + 3*21*38 + 3 + 3*3*38 : 3*38 + 3*21*38 + 3 + 3*3*38 + 12*18].reshape(N_control,12)    
        obj_val = minlp_info_compressed[-4]
        mipgap = minlp_info_compressed[-3]
        runtime = minlp_info_compressed[-2]
        status = minlp_info_compressed[-1]
    elif N_control==38: #(N=40)
        state_n = minlp_info_compressed[0:3*38].reshape(3,38)    
        state_rho = minlp_info_compressed[3*38 : 3*38 + 3*41*38].reshape(3,41,38)    
        state_depot = minlp_info_compressed[3*38 + 3*41*38 : 3*38 + 3*41*38 + 3].reshape(3,)    
        state_l = minlp_info_compressed[3*38 + 3*41*38 + 3 : 3*38 + 3*41*38 + 3 + 3*3*38].reshape(3,3,38)   
        delta_minlp = minlp_info_compressed[3*38 + 3*41*38 + 3 + 3*3*38 : 3*38 + 3*41*38 + 3 + 3*3*38 + 12*38].reshape(N_control,12)    
        obj_val = minlp_info_compressed[-4]
        mipgap = minlp_info_compressed[-3]
        runtime = minlp_info_compressed[-2]
        status = minlp_info_compressed[-1]
    else:
        print('Invalid prediction horizon (N must be 20 or 40).')
    
    return state_n, state_rho, state_depot, state_l, delta_minlp, obj_val, mipgap, runtime, status

def get_preprocessed_data(opt, threshold_counts, N):
    # import numpy as np
    # N=40
    # minlp_info_compressed = np.load('data_milp//data_milp_N%.2d_%.2d.npy' %(N, 0), allow_pickle=True)
    # for job_idx in range(1,28):
    #     tmp_vector = np.load('data_milp//data_milp_N%.2d_%.2d.npy' %(N, job_idx), allow_pickle=True)
    #     minlp_info_compressed = np.concatenate((minlp_info_compressed, tmp_vector))
    # minlp_info_compressed = minlp_info_compressed[:120000, :]
    # np.save('data_milp//data_milp_ol_N%.2d_condensed.npy' %N, minlp_info_compressed, allow_pickle=True)
    
    N_control = N - 2
    
    str_data = 'data_optimal_old//data_%s_N%d_condensed.npy' %(opt, N)
    minlp_info_compressed = np.load(str_data, allow_pickle=True)
        
    N_datapoints = minlp_info_compressed.shape[0]

    state_min = np.min(minlp_info_compressed, axis=0)
    state_max = np.max(minlp_info_compressed, axis=0)

    state_n_min, state_rho_min, state_depot_min, state_l_min, _, _, _, _, _ = decompress_minlp_info(state_min,N_control)
    state_min_reduced = preprocess_state(state_n_min, state_rho_min, state_depot_min, state_l_min)

    state_n_max, state_rho_max, state_depot_max, state_l_max, _, _, _, _, _ = decompress_minlp_info(state_max,N_control)
    state_max_reduced = preprocess_state(state_n_max, state_rho_max, state_depot_max, state_l_max)

    # does one iteration of the state preprocessing to find out the input_size of the vector
    normalized_state = norm_state(minlp_info_compressed[0], state_min, state_max)
    state_n_norm, state_rho_norm, state_depot_norm, state_l_norm, _, _, _, _, _ = decompress_minlp_info(normalized_state,N_control)
    state_learning = preprocess_state(state_n_norm, state_rho_norm, state_depot_norm, state_l_norm)
    input_size = state_learning.shape[0]

    stacked_states = np.zeros((N_datapoints,1,input_size))
    # stacked_states = np.zeros((N_datapoints,N_control,input_size))

    stacked_list_actions = np.zeros((N_datapoints, N_control))

    # num_actions = int(list(action_dict)[-1])+1 #index starts at 0
    # stacked_labels = np.zeros((N_datapoints,N_control,num_actions)) # occupies to much RAM

    stacked_obj_val = np.zeros((N_datapoints,))
    stacked_mipgap = np.zeros((N_datapoints,))
    stacked_runtime = np.zeros((N_datapoints,))
    stacked_status = np.zeros((N_datapoints,))

    for j in range(N_datapoints):
        
        #input preprocessing
        
        _, _, _, _, delta_minlp, obj_val, mipgap, runtime, status = decompress_minlp_info(minlp_info_compressed[j],N_control)
        normalized_state = norm_state(minlp_info_compressed[j], state_min, state_max)
        state_n_norm, state_rho_norm, state_depot_norm, state_l_norm, _, _, _, _, _ = decompress_minlp_info(normalized_state,N_control)
        
        stacked_states[j,0,:] = preprocess_state(state_n_norm, state_rho_norm, state_depot_norm, state_l_norm)
        # stacked_states[j,1:,:] = np.zeros((N_control-1, input_size)) # padding is done during training to save RAM
        
        #label preprocessing (label is preprocessed later to save RAM)

        # idx_0 = j*np.ones(N_control, dtype=np.int32)
        # idx_1 = np.arange(N_control)
        # idx_2 = build_list_action(np.round(delta_minlp,2), N_control)
        # idx_list = list(zip(idx_0, idx_1, idx_2))

        # for k in range(N_control):
        #     stacked_labels[idx_list[k]] = 1
            
        stacked_obj_val[j] = obj_val
        stacked_mipgap[j] = mipgap
        stacked_runtime[j] = runtime
        stacked_status[j] = status
        
        stacked_list_actions[j] = build_list_action(np.round(delta_minlp,2), N_control)
        
    del minlp_info_compressed

    #reduces the set of actions (subset of all optimal actions)
    # threshold_counts = 50 # to eliminate actions that appear rarely
    action_set, counts = np.unique(stacked_list_actions[:,0], return_counts=True)
    action_set = action_set[np.where(counts>threshold_counts)]
    list_action_set = [action_set.astype(np.int32)]
    total_action_set = action_set.astype(np.int32)

    for i in range(1,N_control):
        action_set, counts = np.unique(stacked_list_actions[:,i], return_counts=True)
        action_set = action_set[np.where(counts>threshold_counts)]
        total_action_set = np.union1d(total_action_set, action_set.astype(np.int32))
        list_action_set.append(action_set.astype(np.int32))

    #creates the masking for each step of the control horizon
    list_masks = []

    for j in range(N_control):
        mask = np.zeros(total_action_set.shape, dtype=np.int32)
        for i in range(list_action_set[j].shape[0]):
            mask[np.where(total_action_set==list_action_set[j][i])] = 1
        list_masks.append(mask)
        
    num_actions = total_action_set.shape[0]
    print('num_actions =', num_actions)

    stacked_states_train, stacked_actions_train, stacked_states_val, stacked_actions_val = split_train_validation(stacked_states, stacked_list_actions, split_position=0.8)

    # stacked_actions_reduced_train = np.zeros(stacked_actions_train.shape)
    # for i in range(stacked_actions_train.shape[0]):
    #     stacked_actions_reduced_train[i] = reduce_list_actions(stacked_actions_train[i],total_action_set)
        
    # stacked_actions_reduced_val = np.zeros(stacked_actions_val.shape)
    # for i in range(stacked_actions_val.shape[0]):
    #     stacked_actions_reduced_val[i] = reduce_list_actions(stacked_actions_val[i],total_action_set)
    
    stacked_actions_reduced_train = []
    stacked_states_reduced_train = []
    cntr_outlier_train = 0
    for i in range(stacked_actions_train.shape[0]):
        try:
            stacked_actions_reduced_train.append(reduce_list_actions(stacked_actions_train[i],total_action_set, N_control))
            stacked_states_reduced_train.append(stacked_states_train[i])
        except:
            cntr_outlier_train +=1
       
    stacked_actions_reduced_val = []
    stacked_states_reduced_val = []
    cntr_outlier_val = 0        
    for i in range(stacked_actions_val.shape[0]):
        try:
            stacked_actions_reduced_val.append(reduce_list_actions(stacked_actions_val[i],total_action_set, N_control))
            stacked_states_reduced_val.append(stacked_states_val[i])
        except:
            cntr_outlier_val +=1
            
    print('number of training points (before reduction): %d' %stacked_actions_train.shape[0])
    print('number of validation points (before reduction): %d' %stacked_actions_val.shape[0])
        
    del stacked_actions_train, stacked_actions_val

    # stacked_states_train_tensor = torch.tensor(stacked_states_train, dtype=torch.float32)
    # stacked_states_val_tensor = torch.tensor(stacked_states_val, dtype=torch.float32)
    
    stacked_actions_reduced_train = np.array(stacked_actions_reduced_train)
    stacked_actions_reduced_val = np.array(stacked_actions_reduced_val)
    
    stacked_states_reduced_train = np.array(stacked_states_reduced_train)
    stacked_states_reduced_val = np.array(stacked_states_reduced_val)    
    
    stacked_states_train_tensor = torch.tensor(stacked_states_reduced_train, dtype=torch.float32)
    stacked_states_val_tensor = torch.tensor(stacked_states_reduced_val, dtype=torch.float32)
    
    # stacked_states_reduced_train_tensor = torch.tensor(stacked_states_reduced_train, dtype=torch.float32)
    # stacked_states_reduced_val_tensor = torch.tensor(stacked_states_reduced_val, dtype=torch.float32)
    
    print('number of training points (after reduction): %d' %stacked_states_train_tensor.shape[0])
    print('number of validation points (after reduction): %d' %stacked_states_val_tensor.shape[0])
    print('cntr_outlier_train: %d\t' %cntr_outlier_train, 'cntr_outlier_val: %d' %cntr_outlier_val)
    print('data-processing finished')
    
    return N, N_control, stacked_states_train, stacked_states_val, stacked_actions_reduced_train, stacked_actions_reduced_val, list_masks, stacked_states_train_tensor, stacked_states_val_tensor, state_min_reduced, state_max_reduced, input_size, total_action_set#, stacked_states_reduced_train_tensor, stacked_states_reduced_val_tensor