import pandas as pd
import argparse
import json
import numpy as np 

parser = argparse.ArgumentParser(description='Variant Dataset Creator.')
parser.add_argument('--excel_files', type = str, nargs = '+', required = True, help = 'paths excel sheets')
parser.add_argument('--output_path', type = str, default = 'variant_classification_dataset.json')

COL2Label = {0:'transcript', 1: 'dna', 2: 'protein'}
SPLIT_PROBS = [0.7, 0.15, 0.15]

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = []
    i = 0

    for excel_file in args.excel_files:
        mini_dataset = {}
        
        metadata = pd.read_excel(excel_file)
        first_col_name = list(metadata.columns)[0]

        metadata[first_col_name].replace('', np.nan, inplace=True)
        metadata.dropna(subset=[first_col_name], inplace=True)

        mini_dataset['split'] = np.random.choice(['train', 'dev', 'test'], p = SPLIT_PROBS, size = metadata.shape[0]).tolist()
        mini_dataset['id'] = np.arange(0,metadata.shape[0])
        mini_dataset['x'] = list(metadata[first_col_name])
        y = np.argmax(np.sum(metadata.iloc[:,1:], axis = 0).tolist())
        mini_dataset['y'] = [y]* metadata.shape[0]
        mini_dataset['label'] = [COL2Label[y]]* metadata.shape[0]

        clean_y = np.zeros(metadata.shape[1] - 1).tolist()
        clean_y[y] =  metadata.shape[0]
        assert np.sum(metadata.iloc[:,1:], axis = 0).tolist() == clean_y

        dataset.extend(pd.DataFrame(mini_dataset).to_dict('records'))
        
        i += metadata.shape[0]
    
    json.dump(dataset, open(args.output_path, 'w'))