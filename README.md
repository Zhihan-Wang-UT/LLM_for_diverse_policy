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
python ../../run.py --config-path /scratch/cluster/zhihan/LLM_for_diverse_policy/configs --config-name TrajeDi_cleanup_jsd_1
```