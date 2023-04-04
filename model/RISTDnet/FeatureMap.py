import torch


def GenLikeMap(feature, batch_size, W, H):
    # 生成似然图
    likelihood = torch.full((batch_size, W, H), -float('inf')).cuda()
    for i in range(feature.shape[2]):
        for j in range(feature.shape[3]):
            bl = feature[:, :, i, j]
            likelihood[:, i * 8:i * 8 + 8, j * 8:j * 8 + 8] = bl.reshape(-1, 8, 8)
    return likelihood
