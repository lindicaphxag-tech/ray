"""Two-process native PyTorch DDP checkpoint-continuation control.

No Ray, optimizer hooks, custom gradient hooks or monkeypatches. Three
independent gloo process groups: uninterrupted, full resume, weights-only.
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD=2
SEEDS=tuple(int(x) for x in os.environ.get("DDP_ISOLATION_SEEDS", "7,19,31,43").split(","))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(5,9),nn.Tanh(),nn.Linear(9,2))
    def forward(self,x):
        return self.net(x)

def reserve_port():
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]

def worker(rank,port,stage,seed,directory):
    os.environ['MASTER_ADDR']='127.0.0.1'
    os.environ['MASTER_PORT']=str(port)
    torch.set_num_threads(1)
    dist.init_process_group('gloo',rank=rank,world_size=WORLD)
    try:
        torch.manual_seed(seed)
        net=Net()
        ddp=DDP(net)
        optimizer=torch.optim.Adam(ddp.parameters(),lr=0.001)
        torch.manual_seed(10000+seed*17+rank)
        x=torch.randn(32,5)
        target=torch.randn(32,2)
        def train_step():
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(ddp(x),target)
            loss.backward()
            optimizer.step()
            return float(loss.item())
        dirpath=Path(directory)
        checkpoint=dirpath / f'seed{seed}_warm_rank{rank}.pt'
        if stage=='uninterrupted':
            train_step()
            torch.save({'model':copy.deepcopy(net.state_dict()),
                        'optimizer':copy.deepcopy(optimizer.state_dict())},checkpoint)
        else:
            stored=torch.load(checkpoint,weights_only=True,map_location='cpu')
            net.load_state_dict(stored['model'])
            if stage=='full_resume':
                optimizer.load_state_dict(stored['optimizer'])
        dist.barrier()
        loss=train_step()
        torch.save({'model':copy.deepcopy(net.state_dict()),
                    'optimizer':copy.deepcopy(optimizer.state_dict()),
                    'loss':loss},dirpath/f'seed{seed}_{stage}_rank{rank}.pt')
        dist.barrier()
    finally:
        dist.destroy_process_group()

def flatten(v, prefix=''):
    out={}
    if isinstance(v,dict):
        for k,z in v.items():out.update(flatten(z,prefix+'/'+str(k)))
    elif isinstance(v,(list,tuple)):
        for i,z in enumerate(v):out.update(flatten(z,prefix+'/'+str(i)))
    elif torch.is_tensor(v):out[prefix]=v.detach().to(dtype=torch.float64).clone()
    elif isinstance(v,(int,float)):out[prefix]=torch.tensor(float(v),dtype=torch.float64)
    return out

def compare(a,b):
    x,y=flatten(a),flatten(b)
    assert x.keys()==y.keys(),(x.keys()-y.keys(),y.keys()-x.keys())
    return max((x[k]-y[k]).abs().max().item() for k in x) if x else 0.

def run_group(stage,seed,outdir):
    mp.spawn(worker,args=(reserve_port(),stage,seed,str(outdir)),nprocs=WORLD,join=True)

def main():
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)
        for seed in SEEDS:
            for phase in ('uninterrupted','full_resume','weights_only'):
                run_group(phase,seed,p)
            read=lambda phase,rank:torch.load(p/f'seed{seed}_{phase}_rank{rank}.pt',weights_only=True)
            ref0,ref1=read('uninterrupted',0),read('uninterrupted',1)
            full0,full1=read('full_resume',0),read('full_resume',1)
            weight0,weight1=read('weights_only',0),read('weights_only',1)
            report={
                'seed':seed,'torch':torch.__version__,
                'full_model_error':compare(ref0['model'],full0['model']),
                'full_adam_error':compare(ref0['optimizer'],full0['optimizer']),
                'weights_only_model_error':compare(ref0['model'],weight0['model']),
                'reference_ddp_peer_model_error':compare(ref0['model'],ref1['model']),
                'full_resume_ddp_peer_model_error':compare(full0['model'],full1['model']),
                'weights_only_ddp_peer_model_error':compare(weight0['model'],weight1['model'])
            }
            assert report['full_model_error']==0,report
            assert report['full_adam_error']==0,report
            assert report['reference_ddp_peer_model_error']==0,report
            assert report['full_resume_ddp_peer_model_error']==0,report
            assert report['weights_only_model_error']>0,report
            print('DDP_ISOLATION',json.dumps(report,sort_keys=True),flush=True)
        print(f'{len(SEEDS)}_SEED_DDP_ADAM_FULL_RESUME_EQUIVALENCE_PASSED',flush=True)

if __name__=='__main__':main()
