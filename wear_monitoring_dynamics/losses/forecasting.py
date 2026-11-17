from torch.nn.functional import mse_loss


def forecasting_loss(model, batch, pairs):
    if not len(pairs):
        raise ValueError('Forecasting supervision requires training pairs')
    return mse_loss(model.forecast(batch, pairs), batch['target'][pairs[:, 1]])
