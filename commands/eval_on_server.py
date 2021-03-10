import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import string 
import argparse 

'''
To test, run: python eval_on_server.py --input_string 'c.(2389+1_2390-1)_(2547+1_2548-1)del'
'''
ALL_LETTERS = string.punctuation + string.ascii_letters + string.digits
NUM_ALL_LETTERS = len(ALL_LETTERS)
MODEL_PATH = 'geneticvars/b8dbad27fb4da4206a2e07ed730dd951_model_nodevice.pt'
MAX_STR_LEN = 16
IDX2Label = {0:'transcript', 1: 'dna', 2: 'protein'}
RESULT = 'Saved predictions file to: {}'
    
parser = argparse.ArgumentParser(description='Run Variant Name Classification')
parser.add_argument('--input_textfile_path', type = str, help = 'Path to .txt file containing one variant name per line.')
parser.add_argument('--output_textfile_path', type = str, help = 'Path to .txt file where predictions are saved.')
parser.add_argument('--batch_size', type = int, default = 100, help = 'Size of batch to run through model in one step.')

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.input_size = 94
        self.hidden_dim = 4
        self.num_layers = 1
        self.dropout = 0
        self.seq_len = 16
        self.num_classes = 3
        self.device = 'cpu'

        self.gru = nn.GRU(
            input_size = self.input_size ,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
            bias = True,
            batch_first = True,
            dropout = self.dropout,
            bidirectional = False)
        self.fc = nn.Linear(self.hidden_dim*(1 + self.seq_len), self.num_classes)

    def forward(self, x, batch=None):
        h0 = self.initHidden(x)
        x = pack_padded_sequence(x, batch['string_lens'], enforce_sorted=True, batch_first = True)
        self.gru.flatten_parameters()
        output, h_n  = self.gru(x, h0)
        output, str_lens = pad_packed_sequence(output, padding_value = 0, total_length=self.seq_len, batch_first = True)
        B, _, _= output.shape
        output = output.reshape(B, -1)
        h_n = h_n[-1]
        linear_input = torch.cat([output, h_n], dim = 1)
        return self.fc(linear_input)
    

    def initHidden(self, x):
        B, _, _ = x.shape
        h0 = torch.zeros(self.num_layers, B,  self.hidden_dim)
        return h0

def strToTensor(line):
    tensor = torch.zeros(len(line), NUM_ALL_LETTERS)
    for letter_index, letter in enumerate(line):
        one_hot_index = ALL_LETTERS.find(letter)
        tensor[letter_index, one_hot_index] = 1
    return tensor

def pad_tensor(tensor):
    pad_tensor = torch.zeros(MAX_STR_LEN - tensor.shape[0], NUM_ALL_LETTERS)
    return torch.cat([tensor,pad_tensor], dim = 0 )

def prepare_input(line):
    line = line[:MAX_STR_LEN]
    x = pad_tensor(strToTensor(line))
    return x, len(line)

def run_model(x, batch, model):
    probs = model(x, batch = batch)
    preds = torch.softmax(probs, dim = -1)
    probs, preds = torch.topk(preds, k = 1)
    probs, preds = probs.view(-1), preds.view(-1)
    return [IDX2Label[p.item()] for p in preds]

if __name__ == "__main__":
    args = parser.parse_args()
    
    # LOAD MODEL
    model = GRU()
    model.load_state_dict( torch.load(MODEL_PATH, map_location = torch.device('cpu')) )
    model.eval()

    # READ INPUT
    with open(args.input_textfile_path, 'r') as f:
        raw_strings = f.readlines()

    raw_strings = [input_string.split('\n')[0] for input_string in raw_strings]
    
    batches = [ raw_strings[ i: i+ args.batch_size] for i in range(0, len(raw_strings),  args.batch_size)]
    for input_strings in batches:
        inputs_list = [prepare_input(input_string) for input_string in input_strings]
        inputs_list = sorted(inputs_list, key = lambda row: row[1], reverse=True)

        batch = { 
                'x': torch.stack([ i[0] for i in inputs_list]),
                'string_lens': torch.tensor( [i[1] for i in inputs_list], dtype = torch.int64)
        }
        
        with torch.no_grad():
            name_classes = run_model(batch['x'], batch, model)
    
        with open(args.output_textfile_path, 'a') as f:
            for input_str, pred in zip(input_strings, name_classes):
                f.write('{}, {}\n'.format(input_str, pred))

    print(RESULT.format(args.output_textfile_path))
