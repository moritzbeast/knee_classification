#!/usr/bin/env bash
# Setup on the SAMK GPU server. Run once from the project folder:
#   sudo apt update && sudo apt install -y python3-venv tmux git
#   git clone the repo to ~/proj, upload data/Labeled and data/CSVs from the Mac, then cd ~/proj
#   bash setup_server.sh
# Afterwards: source ~/venv/bin/activate
set -e

python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip
# torch 2.8 instead of 2.7: the RTX PRO 4500 (Blackwell) needs a CUDA 12.8 build,
# which is the default package on PyPI from torch 2.8 on. Other versions as on the Mac.
pip install torch==2.8.0 torchvision==0.23.0 numpy==1.26.4 pandas==2.2.1 \
    opencv-python-headless==4.9.0.80 scikit-learn==1.4.1.post1 matplotlib==3.8.3 tqdm==4.66.2

# real GPU test, not only is_available (catches missing kernels for this card)
python -c "
import torch
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(256, 256, device='cuda')
print('matmul ok:', (x @ x).sum().item() == (x @ x).sum().item())
"

# data is uploaded from the Mac (not part of the repo)
if [ ! -d "data/Labeled" ]; then
    echo "data/Labeled is missing: upload it from the Mac first"
    exit 1
fi
# splits: same as on the Mac (sorted file order + seed 42), only created if missing
if [ ! -f "data/CSVs/test.csv" ]; then
    python create_folds.py
fi

echo "Done. Example training: tmux new -s train, then source ~/venv/bin/activate && python main.py"
