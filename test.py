import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device, read_json
import os
import wandb
from tqdm import tqdm
import glob


def average(arr):
    # receives Nx5 and outputs 1x5
    return np.average(arr, axis=0)

def median(arr):
    # receives Nx5 and outputs 1x5
    return np.median(arr, axis=0)

aggregate = {
    "median": median,
    "average": average,
    }

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# fix random seeds for reproducibility
SEED = 6
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def print_results(gt, pred):
    mse = ((gt - pred)**2).mean(axis=0)
    print(f"MSE for OCEAN: {mse}")

def test_single(config, leftout_idx, method):
    if config.resume is None:
        print("[ERROR] Please, specify the checkpoint to load by using the --resume/-r argument.")
        return None, None

    # load the data loader for this leftout_idx
    config["data_loader"]["args"]["log"] = False
    config["data_loader"]["args"]["drop_last"] = False
    config["data_loader"]["args"]["leftout_idx"] = leftout_idx
    #config["data_loader"]["args"]["batch_size"] = 16
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader, test_data_loader = data_loader.split_validation_leaveoneout()

    # build model architecture
    config["arch"]["args"]["seq_length"] = config["data_loader"]["args"]["w_size"] # they are equal
    model = config.init_obj('arch', module_arch)

    #print('Loading checkpoint: {} ...'.format(config.resume)) 
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    av_loss = 0
    av_mets = {}
    # we iterate through session and segment
    all_results = []
    with torch.no_grad():
        # prepare batch data
        for batch_idx, batch in enumerate(test_data_loader):
                data, target = batch[0].to(device), batch[1].to(device)

                output = model(data)

                all_results.append(output.cpu().detach().numpy())

    output = np.concatenate(all_results, axis=0)

    #print(f"Aggregating with method '{method}'.")
    y_pred = aggregate[method](output)
    
    y_GT = data_loader.dataset.get_GT_personality(leftout_idx)
    return y_GT, y_pred


###################################################################333
def test_leave_one_out(config_path, checkpoints_path, method):
    config = ConfigParser(read_json(config_path))

    config.config["seed"] = SEED
    config["data_loader"]["args"]["log"] = False
    logger = config.get_logger('train', config['trainer']['verbosity'])

    for i in range(torch.cuda.device_count()):
        logger.info(f"> GPU {i} ready: {torch.cuda.get_device_name(i)}")

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)

    participants_ids = data_loader.dataset.participants_ids
    num_participants = len(participants_ids)
    check_models(checkpoints_path, participants_ids)

    all_preds, all_gts = [], []
    for p_id in tqdm(participants_ids):
        config = ConfigParser(read_json(config_path), resume=os.path.join(checkpoints_path, f"lo{p_id}_model_best.pth"))
        y_gt, y_pred = test_single(config, p_id, method=method)
        all_preds.append(y_pred)
        all_gts.append(y_gt)
        
    all_preds = np.stack(all_preds, axis=0)
    all_gts = np.stack(all_gts, axis=0)

    print_results(all_gts, all_preds)
    
    
def check_models(checkpoints_path, participants_ids):
    pth_paths = glob.glob(os.path.join(checkpoints_path, f"*model_best*.pth"))
    for p_id in participants_ids:
        assert os.path.join(checkpoints_path, f"lo{p_id}_model_best.pth") in pth_paths, f"Checkpoint for participant {p_id} not found!"
    return True


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file in the checkpoints folder to be evaluated')
    args.add_argument('-m', '--method', default=list(aggregate.keys())[0], type=str,
                      help=f'options={tuple(aggregate.keys())}')
    args = args.parse_args()
    
    assert args.method in tuple(aggregate.keys()), f"Method '{args.method}' is not implemented."
    
    checkpoints_path = os.path.dirname(args.config)
    test_leave_one_out(args.config, checkpoints_path, args.method)
