import numpy as np
import torch
from networks import U_Net, Ada_U_Net
from training import training_loop
import torch.optim as optim
import torch.nn as nn
from data_prep import tr_val_test
from perf_eval import model_perf
import pandas as pd
import scipy.io as io
import datetime

DA_flag = True  # the Data Augmentation flag

trX, trY, valX, valY, testX, testY, input_trT, input_valT, input_testT = tr_val_test(inputs='inputs.npz',
                                                                                     targets='targets.npz',
                                                                                     shuffle=True,
                                                                                     data_augmentation=DA_flag,
                                                                                     sim_flag=True)

# normalize the inputs
# x = np.concatenate((trX, valX), axis=0)
# y = np.concatenate((trY, valY), axis=0)
#
# x_avg = np.mean(x, axis=(0, 2, 3))[np.newaxis, :, np.newaxis, np.newaxis]
# x_std = np.std(x, axis=(0, 2, 3))[np.newaxis, :, np.newaxis, np.newaxis]
#
# y_avg = np.mean(y, axis=(0, 2, 3))[np.newaxis, :, np.newaxis, np.newaxis]
# y_std = np.std(y, axis=(0, 2, 3))[np.newaxis, :, np.newaxis, np.newaxis]
#
# trX = (trX - 20) / 30
# trY = (trY - 20) / 30
# #
# valX = (valX - 20) / 30
# valY = (valY - 20) / 30
# #
# testX = (testX - 20) / 30
# testY = (testY - 20) / 30

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda'
dtype = torch.float32

tr_x = torch.from_numpy(trX).to(device=device, dtype=dtype)
tr_y = torch.from_numpy(trY).to(device=device, dtype=dtype)
val_x = torch.from_numpy(valX).to(device=device, dtype=dtype)
val_y = torch.from_numpy(valY).to(device=device, dtype=dtype)
test_x = torch.from_numpy(testX).to(device=device, dtype=dtype)
test_y = torch.from_numpy(testY).to(device=device, dtype=dtype)

# since Conv2d expects a B × C × H × W shaped tensor as input

n_epochs = 100
batch_size = 48  # cannot be too small since the batch normalization
patience = 7  # How long to wait after last time validation loss improved.
loss_fn = nn.MSELoss()
model = U_Net().to(device, dtype=torch.float32)
optimizer = optim.SGD(model.parameters(), lr=1e-2)

model, avg_train_losses, avg_valid_losses = training_loop(n_epochs, batch_size, patience, optimizer, model, loss_fn,
                                                          tr_x, tr_y, val_x, val_y)

from plots import plot_sim_spatial

plot_sim_spatial()
"""
# If we trained the model with DA, the DA should be depressed as we evaluate the model performance
# So we have to obtain the data without the DA again
if DA_flag:
    device = 'cpu'
    trX, trY, valX, valY, testX, testY, input_trT, input_valT, input_testT = tr_val_test(inputs='inputs.npz',
                                                                                         targets='targets.npz',
                                                                                         shuffle=True,
                                                                                         data_augmentation=not DA_flag)
    tr_x = torch.from_numpy(trX).to(device=device, dtype=dtype)
    tr_y = torch.from_numpy(trY).to(device=device, dtype=dtype)
    val_x = torch.from_numpy(valX).to(device=device, dtype=dtype)
    val_y = torch.from_numpy(valY).to(device=device, dtype=dtype)
    test_x = torch.from_numpy(testX).to(device=device, dtype=dtype)
    test_y = torch.from_numpy(testY).to(device=device, dtype=dtype)

corr_tr, nse_tr, nrmse_tr, model_basin_tr, targets_basin_tr = model_perf(model, tr_x, trY,
                                                                         boundary_file='yangtze.txt',
                                                                         resolution=0.5,
                                                                         device=device)

corr_val, nse_val, nrmse_val, model_basin_val, targets_basin_val = model_perf(model, val_x, valY,
                                                                              boundary_file='yangtze.txt',
                                                                              resolution=0.5,
                                                                              device=device)

corr_test, nse_test, nrmse_test, model_basin_test, targets_basin_test = model_perf(model, test_x, testY,
                                                                                   boundary_file='yangtze.txt',
                                                                                   resolution=0.5,
                                                                                   device=device)

model_basin = np.vstack((model_basin_tr, model_basin_val, model_basin_test))
targets_basin = np.vstack((targets_basin_tr, targets_basin_val, targets_basin_test))
time = np.vstack((input_trT, input_valT, input_testT))
for i in range(time.shape[0]):
    temp_date = time[i][0]
    time[i][0] = datetime.datetime(temp_date.year, temp_date.month, 1)

flag_tr = np.ones((len(model_basin_tr), 1))
flag_val = np.ones((len(model_basin_val), 1)) + 1
flag_test = np.ones((len(model_basin_test), 1)) + 2
period_flag = np.vstack((flag_tr, flag_val, flag_test))
data = np.hstack((time, model_basin, targets_basin, period_flag))

basin_result = pd.DataFrame(data, columns=['time', 'model_basin', 'targets_basin',
                                           'period_flag'])  # period_flag=1, 2,3 for training, validating and testing
# period, respectively

basin_result = basin_result.sort_values(by='time').values  # convert the dataframe to the ndarray

mdic = {"corr_tr": corr_tr, "nse_tr": nse_tr, "nrmse_tr": nrmse_tr, "corr_val": corr_val, "nse_val": nse_val,
        "nrmse_val": nrmse_val, "corr_test": corr_test, "nse_test": nse_test, "nrmse_test": nrmse_test,
        "basin_result": basin_result, "avg_train_losses": avg_train_losses, "avg_valid_losses": avg_valid_losses}

print('over')
"""
# if DA_flag:
#     io.savemat("perf_withDA_noise2.mat", mdic)
#     torch.save(model.state_dict(), 'model_withDA_noise2.pt')
# else:
#     io.savemat("perf_withoutDA_ada.mat", mdic)
#     torch.save(model.state_dict(), 'model_withoutDA_ada.pt')
