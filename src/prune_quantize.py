import argparse
import torch
import torch.nn.utils.prune as prune
from copy import deepcopy

from dataloader import make_dataloaders
from model import build_resnet50_head
from evaluate import evaluate  

def one_shot_prune(model, amount=0.5):
    params_to_prune = [(m, 'weight') for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
  
    for m, name in params_to_prune:
        try:
            prune.remove(m, name)
        except Exception:
            pass
    return model

def iterative_prune(model, steps=3, amount_per_step=0.2):
    m = deepcopy(model)
    for _ in range(steps):
        params_to_prune = [(mod, 'weight') for mod in m.modules() if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear))]
        prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=amount_per_step)
    for mod, name in params_to_prune:
        try:
            prune.remove(mod, name)
        except Exception:
            pass
    return m

def quantize_dynamic(model):
    model_cpu = deepcopy(model).to('cpu').eval()
    if torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'fbgemm' if 'fbgemm' in torch.backends.quantized.supported_engines else torch.backends.quantized.supported_engines[0]
    q_model = torch.quantization.quantize_dynamic(model_cpu, {torch.nn.Linear}, dtype=torch.qint8)
    return q_model