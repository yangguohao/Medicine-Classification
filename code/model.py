import os

import timm
import torch
import torch.optim as optim


def get_model(model_name, model_path="", ema=False, pretrained=True):
    model = timm.create_model(
        model_name,
        pretrained=pretrained, num_classes=54)

    if ema:
        ema_model = optim.swa_utils.AveragedModel(model,
                                                  avg_fn=optim.swa_utils.get_ema_avg_fn(),
                                                  use_buffers=True)
        print(os.path.exists(model_path))
        if os.path.exists(model_path):
            ema_model.load_state_dict(
                torch.load(model_path, weights_only=True, map_location='cpu'))
            print('Successfully load weights')
        model = ema_model.module
    else:
        ema_model = None
        if os.path.exists(model_path):
            model.load_state_dict(
                torch.load(model_path, weights_only=True, map_location='cpu'))
            print('Successfully load weights')
    return model, ema_model
