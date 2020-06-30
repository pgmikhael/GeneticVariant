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

## Make dataset
python -m pdb commands/create_variants_dataset.py --excel_files /Mounts/rbg-storage1/datasets/GeneticVariants/ML_DNA_Variants.xlsx /Mounts/rbg-storage1/datasets/GeneticVariants/ML_Protein.xlsx /Mounts/rbg-storage1/datasets/GeneticVariants/ML_TranscriptOfGene.xlsx --output_path /Mounts/rbg-storage1/datasets/GeneticVariants/variant_classification_dataset_6040.json --split_probs [0.6,0.2,0.2]/[0.5,0.25,0.25]/[0.4,0.3,0.3]/[0.3,0.35,0.35]/[0.2,0.4,0.4]

## Dispatcher
python -m pdb commands/dispatcher.py --config_path configs/configuration.json --alert_config_path configs/pgm_alert_config.json --log_dir /Mounts/rbg-storage1/logs/geneticvars/ --result_path /Mounts/rbg-storage1/users/pgmikhael/result_summaries/geneticvars_baselines_06042020.csv

## Main
'CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pdb main.py'