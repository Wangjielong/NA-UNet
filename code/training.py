import datetime
from pytorchtools import EarlyStopping
import numpy as np
import torch


# Train the Model using Early Stopping
def training_loop(epochs, batch_size, patience, optimizer, model, loss_fn, tr_x, tr_y, val_x, val_y):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    batches = int(len(tr_x) / batch_size)
    for epoch in range(1, epochs + 1):
        # train the model
        model.train()  # prep model for training if we have BN and dropout operation
        for batch in range(1, batches + 1):
            optimizer.zero_grad()  # getting rid of the gradients from the last round
            start = batch * batch_size
            end = start + batch_size
            batch_tr_x = tr_x[start:end]
            batch_tr_y = tr_y[start:end]
            outputs = model(batch_tr_x)
            loss = loss_fn(outputs, batch_tr_y)
            loss.backward()
            optimizer.step()
            # record training loss
            train_losses.append(
                loss.item())  # transform the loss to a  Python number with .item(), to escape the gradients.

        # validate the model without the batch
        model.eval()  # prep model for evaluation if we have BN and dropout operation
        for val_idx in range(len(val_x)):
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(val_x[val_idx:val_idx + 1])
            # calculate the loss
            loss = loss_fn(output, val_y[val_idx:val_idx + 1])
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        print('{} Epoch {}, Training loss {}, Validating loss {}'.format(datetime.datetime.now(), epoch, train_loss,
                                                                         valid_loss))

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    return model, avg_train_losses, avg_valid_losses
