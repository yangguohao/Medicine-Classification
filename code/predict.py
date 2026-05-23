import collections
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from data_utils import load_dataset
from model import get_model


def ensemble_predict(ema=True, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    model_name = "caformer01234"
    output_path = f'../results_B/{model_name}'
    tmp_results = collections.Counter()
    ensemble_models = [
        [ (f'../model/caformer_ls_gc_re_muon_mix_ema_5folds_ft/best_model_ema_0.pth',
         'caformer_b36.sail_in22k_ft_in1k_384', True),
        (f'../model/caformer_ls_gc_re_muon_mix_ema_5folds_ft/best_model_ema_1.pth',
          'caformer_b36.sail_in22k_ft_in1k_384', True),
        (f'../model/caformer_ls_gc_re_muon_mix_ema_5folds_ft/best_model_ema_2.pth',
          'caformer_b36.sail_in22k_ft_in1k_384', True),
        (f'../model/caformer_ls_gc_re_muon_mix_ema_5folds_ft/best_model_ema_3.pth',
         'caformer_b36.sail_in22k_ft_in1k_384', True),
        (f'../model/caformer_ls_gc_re_muon_mix_ema_5folds_ft/best_model_ema_4.pth',
         'caformer_b36.sail_in22k_ft_in1k_384', True),],
        # [(f'../model/convnextv2_ls_gc_re_muon_mix_ema_5folds_ft/best_model_ema_0.pth',
        #  'convnextv2_huge.fcmae_ft_in22k_in1k_384', True),
        # (f'../model/convnextv2_ls_gc_re_muon_mix_ema_5folds_ft/best_model_ema_1.pth',
        #  'convnextv2_huge.fcmae_ft_in22k_in1k_384', True),
        # (f'../model/convnextv2_ls_gc_re_muon_mix_ema_5folds_ft/best_model_ema_2.pth',
        #  'convnextv2_huge.fcmae_ft_in22k_in1k_384', True),
        # (f'../model/convnextv2_ls_gc_re_muon_mix_ema_5folds_ft/best_model_ema_3.pth',
        #  'convnextv2_huge.fcmae_ft_in22k_in1k_384', True),
        # (f'../model/convnextv2_ls_gc_re_muon_mix_ema_5folds_ft/best_model_ema_4.pth',
        #  'convnextv2_huge.fcmae_ft_in22k_in1k_384', True),],
        # [(f'../model/eva02_ls_gc_re_muon_mix_ra_ema_5folds/best_model_ema_0.pth',
        #   "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", True),
        # (f'../model/eva02_ls_gc_re_muon_mix_ra_ema_5folds/best_model_ema_1.pth',
        #  "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", True),
        # (f'../model/eva02_ls_gc_re_muon_mix_ra_ema_5folds/best_model_ema_2.pth',
        #  "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", True),
        # (f'../model/eva02_ls_gc_re_muon_mix_ra_ema_5folds/best_model_ema_3.pth',
        #  "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", True),
        # (f'../model/eva02_ls_gc_re_muon_mix_ra_ema_5folds/best_model_ema_4.pth',
        #  "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", True),]
    ]
    weights = [1]

    for i, ensemble_model in enumerate(ensemble_models):
        # 初始化模型
        for (model_name, model, tta) in ensemble_model:
            model, ema_model = get_model(model, model_name, ema=ema)
            test_dataset, mapping = load_dataset(model, train=False)
            if ema:
                model = ema_model

            print("device = ", device)
            model.to(device)
            model.eval()

            test_loader = DataLoader(test_dataset, batch_size=32, pin_memory=True, num_workers=4)
            # 测试阶段（仅推理不计算准确率）
            start_idx = 0
            with torch.no_grad():
                for inputs, _ in test_loader:
                    inputs = inputs.to(device)
                    outputs = torch.softmax(model(inputs).data, dim=1)
                    if tta:
                        tta_transform = [T.RandomHorizontalFlip()]
                        for aug in tta_transform:
                            inputs = aug(inputs)
                            outputs += torch.softmax(model(inputs).data, dim=1)
                        outputs / (len(tta_transform) + 1)

                    for offset, pred in enumerate(outputs):
                        tmp_results[test_dataset.df.iloc[start_idx + offset][0]] += (pred / len(ensemble_model)) * weights[i]
                    start_idx += len(outputs)

    test_results = []
    for k, v in tmp_results.items():
        test_results.append((k, mapping[torch.argmax(v, dim=0).item()]))
    # 保存测试结果
    os.makedirs(output_path, exist_ok=True)
    results_df = pd.DataFrame(test_results, columns=['ImageID', 'label'])
    results_df.to_csv(f'{output_path}/example_B.csv', index=False)


if __name__ == "__main__":
    ensemble_predict()
