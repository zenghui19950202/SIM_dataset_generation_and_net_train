from utils import load_configuration_parameters
from simulation_data_generation import generate_sinusoidal_SIMdata
from configparser import ConfigParser
from simulation_data_generation import Pipeline_speckle
from Augmentor.Operations import Crop
from simulation_data_generation.fuctions_for_generate_pattern import SinusoidalPattern

if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')
    config.sections()
    SourceFileDirectory = config.get('image_file', 'SourceFileDirectory')
    sample_num_train = config.getint('SIM_data_generation', 'sample_num_train')
    sample_num_valid = config.getint('SIM_data_generation', 'sample_num_valid')
    image_size = config.getint('SIM_data_generation', 'image_size')
    data_num = config.getint('SIM_data_generation', 'data_num')
    # SourceFileDirectory = "D:\DataSet\DIV2K\DIV2K_valid_LR_unknown/test/test2"

    train_directory = SourceFileDirectory + '/train_2x'
    valid_directory = SourceFileDirectory + '/valid'

    p = Pipeline_speckle.Pipeline_speckle(source_directory=train_directory, output_directory="../SIMdata_SR_train")
    p.add_operation(Crop(probability=1, width=image_size, height=image_size, centre=False))
    p.add_operation(SinusoidalPattern(probability=1,image_size = None,config_params_direct = 'config.ini'))
    p.sample(1, multi_threaded=False, data_type='train', data_num=data_num)