import torch.nn as nn

from function import calc_mean_std


def adjust_learning_rate(args, optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def calc_content_loss(input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)

    return nn.MSELoss()(input, target)

def calc_style_loss(input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)

    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)

    mean_loss = nn.MSELoss()(input_mean, target_mean)
    std_loss = nn.MSELoss()(input_std, target_std)
    style_loss = mean_loss + std_loss

    return style_loss

def calc_total_loss(args, t, g_t_feats, style_feats):
    loss_c = calc_content_loss(g_t_feats[-1], t)

    loss_s = calc_style_loss(g_t_feats[0], style_feats[0])
    for i in range(1, 4):
        loss_s += calc_style_loss(g_t_feats[i], style_feats[i])
    
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    total_loss = loss_c + loss_s

    return total_loss
    