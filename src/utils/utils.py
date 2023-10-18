import os
import random
import numpy as np
import json
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, word {}): {}'.format(
        args.rank, args.world_size, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def params_count(model):
    return np.sum([p.numel() for p in   model.parameters()]).item()


def save_loss(loss, epochs, out_path="loss.jpg"):
    plt.plot(epochs, loss, linewidth=1, color="orange", marker="o",label="Mean value")
    plt.legend(["Loss"],loc="upper right")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(out_path)


def getI2TR1Accuary(prob):
    temp = prob.detach().cpu().numpy()
    temp = np.argsort(temp, axis=1)
    count = 0
    for i in range(prob.shape[0]):
        if temp[i][prob.shape[1]-1] == i:
            count+=1
    acc = count/prob.shape[0]
    return acc

def getI2IR1Accuary(prob, oric, othc):
    temp = prob.detach().cpu().numpy()
    ind = np.argsort(temp, axis=1)
    count = 0
    for i in range(prob.shape[0]):
        if oric[ind[i][prob.shape[1]-1]] == othc[ind[i][prob.shape[1]-1]]:
            count+=1
        elif oric[ind[i][prob.shape[1]-1]] != othc[ind[i][prob.shape[1]-1]] and temp[i][ind[i][prob.shape[1]-1]]<=0.1:
            count+=1
    acc = count/prob.shape[0]
    return acc
