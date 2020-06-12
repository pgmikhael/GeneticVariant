import pandas as pd
import argparse
import pickle 
from collections import defaultdict

COL2Label = {0:'transcript', 1: 'dna', 2: 'protein'}

parser = argparse.ArgumentParser(description='Variant Results.')
parser.add_argument('--results_file', type = str, required = True, help = 'paths results')
parser.add_argument('--output_path', type = str, default = 'variant_classification_dataset.csv')

if __name__ == "__main__":
    args = parser.parse_args()
    results = pickle.load(open(args.results_file, 'rb'))
    summary = defaultdict(list)
    for mode in ['train', 'dev', 'test']:
        split_size = len(results['{}_stats'.format(mode)]['{}_strings'.format(mode)])
        golds = [COL2Label[i] for i in results['{}_stats'.format(mode)]['{}_golds'.format(mode)] ]
        preds = [COL2Label[i] for i in results['{}_stats'.format(mode)]['{}_preds'.format(mode)] ]
        summary['VariantName'].extend(results['{}_stats'.format(mode)]['{}_strings'.format(mode)])
        summary['DatasetSplit'].extend([mode]*split_size)
        summary['TrueLabel'].extend(golds)
        summary['PredictedLabel'].extend(preds)
        predicted_correctly = results['{}_stats'.format(mode)]['{}_preds'.format(mode)] == results['{}_stats'.format(mode)]['{}_golds'.format(mode)]
        summary['PredictedCorrectly'].extend(predicted_correctly)

    summary = pd.DataFrame(summary)
    summary = summary[summary['PredictedCorrectly'] == 0]
    summary.to_csv(args.output_path, index=False)
