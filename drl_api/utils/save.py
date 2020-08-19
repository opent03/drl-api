import torch
import time
import os

def save_model(model, savedir, env_name):
    fmt = '{}-{}'
    # time.strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(), os.path.join(savedir,
                                                fmt.format(env_name, model.__class__.__name__)))
