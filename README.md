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
```
### Test environment
```
python envs/cleanup-5a-game/test_cleanup_fullycooperative.py
```