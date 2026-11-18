import os
import json
import random
from datetime import datetime
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds MPS and CUDA


def make_run_dir(args):
    # folder name: timestamp + all training args, e.g.
    # sessions/2026-11-18_14-30-05_backbone-resnet18_epochs-10_bs-32_lr-0.0001_wd-1e-06_seed-42
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    params = "_".join(f"{key}-{value}" for key, value in vars(args).items()
                      if key not in ("output_dir", "csv_dir"))

    run_dir = os.path.join(args.output_dir, f"{timestamp}_{params}")
    os.makedirs(run_dir)

    # all args (incl. paths) for later reference
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    return run_dir
