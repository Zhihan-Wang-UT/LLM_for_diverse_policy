# LLM_for_diverse_policy
use LLM to encourage diversity in policies.



git submodule init
git submodule update


preparing the environment
```
pip install -e ./envs/melting_pot_for_diversity
pip install -e ./envs/cleanup-5a-game 

pip install imageio 
pip install "imageio[ffmpeg]"
pip install hydra-core
pip install wandb
pip install matplotlib
```
### Test environment
```
python envs/cleanup-5a-game/test_cleanup_fullycooperative.py
```


### Run train
```
cd experiments
mkdir exp_name
cd exp_name
```

Baseline MAPPO Sample comman
```
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=/home/hg22723/projects/LLM_for_diverse_policy/experiments/nollmmappo python ../../run.py --config-path /home/hg22723/projects/LLM_for_diverse_policy/configs --config-name MAPPO_cleanup_1

```

MAPPO with LLM2Vec based exploration (required 40 GB VRAM)
```
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=/home/hg22723/projects/LLM_for_diverse_policy/experiments/llmmappo python ../../run.py --config-path /home/hg22723/projects/LLM_for_diverse_policy/configs --config-name MAPPOLLM_cleanup_1

```
