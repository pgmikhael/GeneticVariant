from models.model_factory import RegisterModel
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_packed_sequence

@RegisterModel('gru')
class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()
        self.args = args

        gru_dropout = args.dropout
        if args.num_layers==1:
            gru_dropout = 0

        self.gru = nn.GRU(
            input_size = args.input_dim,
            hidden_size = args.hidden_dim,
            num_layers = args.num_layers,
            bias = True,
            batch_first = False,
            dropout = gru_dropout,
            bidirectional = False)
        self.fc = nn.Linear(args.hidden_dim*(1 + args.seq_len), args.num_classes)

    def forward(self, x, h0, batch=None):
        output, h_n = self.gru(x, h0)
        output, str_lens = pad_packed_sequence(output, padding_value = 0, total_length=self.args.seq_len)
        _, B, _= output.shape
        output = output.transpose(1,0).reshape(B, -1)
        h_n = h_n[-1]
        linear_input = torch.cat([output, h_n], dim = 1)
        return self.fc(linear_input)
    
    def initHidden(self, x):
        _, B, _ = x.shape
        return torch.zeros(self.args.num_layers, B, self.args.hidden_dim)



