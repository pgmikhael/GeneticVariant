import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import string 
import argparse 


ALL_LETTERS = string.punctuation + string.ascii_letters + string.digits
NUM_ALL_LETTERS = len(ALL_LETTERS)
MODEL_PATH = '/Mounts/rbg-storage1/results/geneticvars/b8dbad27fb4da4206a2e07ed730dd951_model.pt'
MAX_STR_LEN = 16
IDX2Label = {0:'transcript', 1: 'dna', 2: 'protein'}

    
parser = argparse.ArgumentParser(description='Run Variant Name Classification')
parser.add_argument('--input_string', type = str, default = 'Input, currently a single string')

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
        if self.args.cuda:
            h0 = h0.to(self.device)
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
    line = input_string[:MAX_STR_LEN]
    x = pad_tensor(strToTensor(line))
    return x, len(line)

def run_model(x, batch, model):
    probs = model(x, batch = batch)
    preds = torch.softmax(probs, dim = -1)
    probs, preds = torch.topk(preds, k = 1)
    #probs, preds = probs.view(B), preds.view(B)
    return IDX2Label[preds.item()]

if __name__ == "__main__":
    args = parser.parse_args()
    input_string = args.input_string
    x, len_x = prepare_input(input_string)
    batch = {'string_lens': len(input_string)}
    x = x.unsqueeze(0)
    model = torch.load(MODEL_PATH)
    model.eval()
    with torch.no_grad():
        name_class = run_model(x, batch, model)
    
    print(name_class)
    