from torch.nn.functional import mse_loss


def monitoring_loss(model, batch, observations):
    return mse_loss(model.estimate(batch, observations), batch['target'][observations])
