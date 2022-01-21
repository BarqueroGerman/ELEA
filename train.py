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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# fix random seeds for reproducibility
SEED = 10
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config, resume, wandb_id=None):
    config.config["seed"] = SEED
    store_to_wandb = "wandb" in config.config and config.config["wandb"]["store"]
    if store_to_wandb and resume:
        if wandb_id == None:
            raise Exception(f"[ERROR] You need to specify the wandb id for the run you want to attach.")

        wandb.init(project=config.config["wandb"]["project"],
                    name=config.config["name"],
                    id=wandb_id,
                    entity="barquerogerman",
                    notes=config.config["wandb"]["description"],
                    tags=config.config["wandb"]["description"],
                    config=config.config,
                    resume="must" if resume else "never")
        print(f"Wandb was resumed successfully.")
    elif store_to_wandb:
        wandb.init(project=config.config["wandb"]["project"],
                    name=config.config["name"],
                    entity="barquerogerman",
                    notes=config.config["wandb"]["description"],
                    tags=config.config["wandb"]["description"],
                    config=config.config)

    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader_training', module_data)
    valid_data_loader = config.init_obj('data_loader_validation', module_data)


    # ----------- TEST -------------
    # Code to test the code to access a specific segment from the data loader.
    """
    from utils import parse_lhand_at, parse_face_at
    session, num_frame, offset, task, part = "136175", 8000, 99, "talk", "FC1_T"
    #session, num_frame, offset, task, part = "136175", 22000, 99, "lego", "FC1_L"
    data, target = data_loader.dataset.get_custom_segment(session, task, num_frame, transform=True)
    loaded = np.split(data[offset].cpu().numpy().astype(int), 2)[0]
    readed = parse_face_at(os.path.join("/annotations/training/all", session, part, "annotations_raw.hdf5"), num_frame+offset)[1].ravel()
    print(loaded)
    print(readed)
    assert (loaded == readed).all(), "The loaded and readed data are not the same"
    """
    # ---------------------------------

    
    logger.info(f"Number of training samples: {data_loader.n_samples}")
    logger.info(f"Number of validation samples: {valid_data_loader.n_samples}")

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    for i in range(torch.cuda.device_count()):
        logger.info(f"> GPU {i} ready: {torch.cuda.get_device_name(i)}")
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    valid_frequency = config['trainer']['validation_frequency']
    logger.info(f"Running validation {valid_frequency} times per epoch.")

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler, 
                      validation_frequency=valid_frequency)

    logger.info("Starting training...")
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-w', '--wandb', default=None, type=str,
                      help='id of wandb run to re-attach')
    args = args.parse_args()
                      
    if args.resume:
        config_path = os.path.join(os.path.dirname(args.resume), "config.json")
    else:
        config_path = args.config

    config = read_json(config_path)
    configparser = ConfigParser(config, resume=args.resume)
    main(configparser, resume=args.resume is not None, wandb_id=args.wandb)
