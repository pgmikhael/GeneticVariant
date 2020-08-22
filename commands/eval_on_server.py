import torch 
import string 
import argparse 


ALL_LETTERS = string.punctuation + string.ascii_letters + string.digits
NUM_ALL_LETTERS = len(ALL_LETTERS)
MODEL_PATH = '/Mounts/rbg-storage1/results/geneticvars/b8dbad27fb4da4206a2e07ed730dd951_model.pt'
MAX_STR_LEN = 16
IDX2Label = {0:'transcript', 1: 'dna', 2: 'protein'}

    
parser = argparse.ArgumentParser(description='Run Variant Name Classification')
parser.add_argument('--input_string', type = str, default = 'Input, currently a single string')

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
    