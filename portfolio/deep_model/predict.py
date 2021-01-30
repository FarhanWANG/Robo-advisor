import torch
from .wavenet import WaveNet
import os
import datetime

# hyperparams
N_LAGS = 90
Y_DAYS = 3
layer_size = 3
stack_size = 12
in_channels = 6 # 6 features
res_channels = 64
MAX_ = 260.138916015625
MIN_ = 5.868809223175049
#intensor = torch.rand(3, 90, 6)
# paths
torch.manual_seed(999)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
	'deep_model/djstock')
net = WaveNet(layer_size, stack_size, in_channels, res_channels, Y_DAYS, N_LAGS)
net.load_state_dict(torch.load(MODEL_PATH))

def stock_pred(input_tensor):
    input_tensor = torch.tensor(input_tensor).float()
    days_90 = input_tensor[:,85:,0].tolist()
    input_tensor = (input_tensor - MIN_) / (MAX_ - MIN_)

    net.eval()
    y_pred = net(input_tensor)
    y_pred = y_pred * (MAX_ - MIN_) + MIN_
    print(y_pred)

    y_pred.tolist()
    total_list = []
    for series, pred in zip(days_90, y_pred):
        total_list.append(series + pred.tolist())

    return_total = []
    for series in total_list:
        day = datetime.datetime.now()-datetime.timedelta(days=5)
        return_dict = []
        for price in series:
            day += datetime.timedelta(days=1)
            return_dict.append({'x': str(day), 'y':price})
        return_total.append(return_dict)

    return return_total




