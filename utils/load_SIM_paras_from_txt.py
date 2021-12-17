import pandas as pd
from utils import load_configuration_parameters
import numpy as np
import torch
import os

def load_params():
    """
    load the real value of structured illumination parameters from  txt file
    :return: SIM_params
    """
    train_net_parameters = load_configuration_parameters.load_train_net_config_paras()
    train_directory_file = train_net_parameters['data_directory_file']
    txt_directory = train_directory_file+"SIMdata_SR_train/illumination_parameters.txt"
    if os.path.exists(txt_directory):
        params_array =  np.loadtxt(txt_directory)
        SIM_params = torch.tensor(params_array)
    else:
        SIM_params = torch.ones(6,4)

    return SIM_params

if __name__ == '__main__':
    train_net_parameters = load_configuration_parameters.load_train_net_config_paras()
    train_directory_file = train_net_parameters['data_directory_file']
    params_array =  np.loadtxt(train_directory_file+"SIMdata_SR_train/illumination_parameters.txt")
    SIM_params = torch.tensor(params_array)
    print(SIM_params)