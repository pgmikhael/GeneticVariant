## Install  utils
- 'sudo yum install mosh tmux htop python-pip'
- 'pip install --upgrade pip'
- 'source activate pytorch_p36'

## On MacOS
- 'brew install mosh tmux htop zsh python-pip'
- 'pip install --upgrade pip'
- 'alias venv_name='source ~/venv_name/bin/activate'
- 'source /.zshrc'
- sudo vim /etc/passwd -> user /zsh
- 'sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"'

## Dispatcher
python -m pdb scripts/dispatcher.py --experiment_config_path configs/configuration.json --alert_config_path configs/pgm_alert_config.json --log_dir /Mounts/rbg-storage1/logs/geneticvars/ --result_path /Mounts/rbg-storage1/users/pgmikhael/result_summaries/geneticvars_baselines_06042020.csv

## Main
'CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pdb main.py'