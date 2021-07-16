import torch
import models
import torch.nn as nn

def get_CNN_prob(
    model,
    dataloader):
    probs = np.ndarray(shape = [dataloader.dataset.len, 2])
    names = []
    with torch.no_grad():
        for idx, data in dataloader:
            inputs, names = data[0].cuda(), data[1]
            if isinstance(model, models.KeNet):
                output_batch, _,_ = model(*inputs)
            else:
                output_batch = model(inputs)
            probs[idx*dataloader.batch_size:(idx+1)*dataloader.batch_size,:]\
                = nn.functional.softmax(output_batch)
            names.extend(list(data[1]))
    return probs, names
