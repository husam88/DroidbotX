# DroidbotX
Fork of droidbot (https://github.com/honeynet/droidbot) that adds gym environment to Droidbot

## Prerequisite

1. `Python` (both 2 and 3 are supported)
2. `Java`
3. `Android SDK`
4. Add `platform_tools` directory in Android SDK to `PATH`
5. (Optional) `OpenCV-Python` if you want to run DroidBot in cv mode.

## How to install

Clone this repo and intall with `pip`:

```shell
git clone https://github.com/AurelianTactics/droidbot-gym.git
cd droidbot/
pip install -e .
```

## Other packages needed
1. Gym (https://github.com/openai/gym for installation instructions)
2. Stable-Baselines (https://github.com/hill-a/stable-baselines for installation instructions)
3. TensorFlow
4. pip install opencv-python
5. Humanoid (Optional) (https://github.com/yzygitzh/Humanoid for installation instructions, pyflann dependency will need to have Python code changed from 2 to 3 per instructions here: https://github.com/primetang/pyflann/issues/1#issuecomment-354001204 )

## How to run

From terminal and droidbot directory run the "start_gym_env_dqn.py" or "start_gym_env_actor_critic.py" scripts. Example usage:
* python3 start_gym_env_dqn.py -a my_apk_to_run.apk -o /my_output_dir/dqn_run_4 -is_emulator -policy gym -humanoid localhost:50405
* python3 start_gym_env_actor_critic.py -a my_apk_to_run.apk -o /my_output_dir/dqn_run_4 -is_emulator -policy gym 

