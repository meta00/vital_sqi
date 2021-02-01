import torch
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, series_IDs, labels):
        'Initialization'
        self.labels = labels
        self.series_IDs = series_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.series_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.series_IDs[index]

        # Load data and get label
        X = get_np_arrays(ID)
        y = int(self.labels[int(ID)])

        return X, y



# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_start_method('spawn')

path = "/media/tuan/NGUI GA/ECG_split/" 

def get_np_arrays(file_name):
    arr = pd.read_csv(path + file_name)['# ECG Channel 1'].values
    return torch.from_numpy(arr).to(device)

df = pd.read_csv('mapped_df.csv')

test_df = df[df['data_type'] == 'test']
train_df = df[df['data_type'] == 'train']
val_df = df[df['data_type'] == 'val']

X_train_filenames = train_df['file'].values.tolist()
y_train = train_df['class'].values.tolist()

X_val_filenames = val_df['file'].values.tolist()
y_val = val_df['class'].values.tolist()

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 0}
max_epochs = 5

# Generators
training_set = Dataset(X_train_filenames, y_train)
training_generator = torch.utils.data.DataLoader(training_set, **params)
validation_set = Dataset(X_val_filenames, y_val)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)


class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
    
    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]

classes = (0, 1)
input_dim = 10    
hidden_dim = 256
layer_dim = 3
output_dim = 9
seq_dim = 128

model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
model = model.cuda()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

max_epochs = 1

# Loop over epochs
for epoch in range(max_epochs):
    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        
        # Model computations
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(local_batch)
        loss = criterion(outputs, local_labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    # Validation
#     with torch.set_grad_enabled(False):
#         for local_batch, local_labels in validation_generator:
#             # Transfer to GPU
#             local_batch, local_labels = local_batch.to(device), local_labels.to(device)

#             # Model computations
#             [...]