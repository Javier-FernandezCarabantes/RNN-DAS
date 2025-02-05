import torch
import torch.nn as nn
from model_builder_RNN_DAS import DRNN  

class Model(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_layers, n_classes, cell_type="LSTM"):
        super(Model, self).__init__()

        self.drnn = DRNN(n_inputs, n_hidden, n_layers, dropout=0, cell_type=cell_type)
        self.softmax = nn.Softmax(dim=2)
        self.linear = nn.Linear(n_hidden, n_classes)

    def forward(self, inputs):
        layer_outputs, _, linear_output, all_time_step = self.drnn(inputs)
        pred = linear_output
        return pred, all_time_step

def load_model(model_path, n_inputs, n_hidden, n_layers, n_classes, cell_type="LSTM"):
    model = Model(n_inputs, n_hidden, n_layers, n_classes, cell_type=cell_type)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  #
    return model