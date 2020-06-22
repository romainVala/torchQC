import commentjson  as json
import argparse
import matplotlib.pyplot as plt

#from segmentation.data import load_data, generate_dataset, generate_dataloader
#from segmentation.visualization import parse_visualization_config_file

from segmentation.utils import instantiate_logger
from segmentation.config import Config
import logging

from plot_dataset import PlotDataset
from torch_summary import summary, print_FOV
import torch

file='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/run_segment/pve_07res_experiment2/main.json'

file='/home/romain.valabregue/datal/QCcnn/run_segment/main.json'
file='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion_New/main.json'
file='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_random_noise_New/main.json'
file='/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion_New/mainCATI.json'

result_dir='/home/romain.valabregue/datal/QCcnn/run_segment/test_mot'

logger = instantiate_logger('info', logging.INFO, result_dir + '/info.txt')

torch.manual_seed(12)
#config = Config(file, result_dir, logger, None, 'train')
config = Config(file, result_dir, logger, None, 'visualization')

config.run()


result_dir = '/tmp/'
logger = instantiate_logger('info', logging.INFO, result_dir + '/info.txt')
file = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion_New/main.json'
efile = '/network/lustre/iss01/cenir/analyse/irm/users/romain.valabregue/QCcnn/NN_regres_motion_New/result_HCP_T1_AffEla_mvt_rescal_mask/eval_test.json'

config = Config(file, result_dir, logger, None, 'eval', extra_file=efile)
config.run()


model_structure = config.parse_model_file(config.main_structure['model'])
model, device = config.load_model(model_structure)

input_size = (1, 84, 84, 84)
input_size = (1, 182, 218, 182)
print(summary(model, input_size, batch_size=1))
#print(summary(model, (1, 84, 84, 84), batch_size=1))
print_FOV(model, input_size, dtypes=None, upsample_list=None)

info=json.load(open(file))

run_model = RunModel(model, train_loader, val_loader, val_set, info['folder'], info['train'])
run_model.train()

import torch
x = torch.rand([1,182,218,182])

s = train_set[0]
plot_volume_interactive(s['t1']['data'][0].numpy())