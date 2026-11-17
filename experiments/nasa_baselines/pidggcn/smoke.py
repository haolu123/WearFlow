"""Synthetic verification and three-epoch inner-only timing, no formal runs."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from model import PIDGGCN,dynamic_adjacency,diagonal_time_derivative,fit_condition_priors,prior_rate_u,physics_loss,HalfDecay

ROOT=Path(__file__).resolve().parent
torch.set_num_threads(1)


def main():
    start=time.perf_counter();torch.manual_seed(7)
    model=PIDGGCN().double().eval()
    x=torch.randn(6,17,dtype=torch.float64)
    y,graphs=model(x,return_graphs=True)
    assert graphs[0].shape==(6,17,17) and graphs[1].shape==(6,64,64)
    for graph in graphs:
        assert (graph>0).all();torch.testing.assert_close(graph.sum(-1),torch.ones_like(graph.sum(-1)))
    perturbed=x.clone();perturbed[3:]+=100
    causal_error=float((model(perturbed)[:3]-y[:3]).abs().max())
    prefix_error=float((model(x[:3])-y[:3]).abs().max())
    assert causal_error<1e-12 and prefix_error<1e-12
    # This diagnostic shows why a full-sequence graph needs an online adaptation.
    offline_effect=float((model(perturbed,causal=False)[:3]-model(x,causal=False)[:3]).abs().max())
    assert offline_effect>1e-9
    u=torch.linspace(.1,1.,6,dtype=torch.float64,requires_grad=True)
    pred=model(torch.cat([x[:,:-1],u[:,None]],1))
    grad=diagonal_time_derivative(pred,u)
    fd=[];eps=1e-5
    for i in range(6):
        plus=u.detach().clone();minus=plus.clone();plus[i]+=eps;minus[i]-=eps
        fd.append((model(torch.cat([x[:,:-1],plus[:,None]],1))[i]-model(torch.cat([x[:,:-1],minus[:,None]],1))[i])/(2*eps))
    derivative_error=float((grad-torch.stack(fd)).abs().max())
    assert derivative_error<1e-7
    abc=[-.9,.2,.3]
    physical=u*torch.exp(abc[0]+abc[1]*u+abc[2]*u*u)
    actual=torch.autograd.grad(physical.sum(),u)[0]
    corrected_error=float((actual-prior_rate_u(u,abc,'calculus_corrected')).abs().max())
    assert corrected_error<1e-12
    assert not torch.allclose(actual,prior_rate_u(u,abc,'paper_printed'))
    schedule=HalfDecay(100,True)
    assert schedule.step(50,100,1)==(1.,1.) and schedule.step(52,100,1)==(.5,1.)
    path=ROOT.parent/'data/fold_a/inner_1'
    a=dict(np.load(path/'arrays.npz'));meta=json.loads((path/'metadata.json').read_text())
    train=a['train_mask'];valid=a['sensor_valid'];traincases=meta['split']['train']
    prior,group_priors=fit_condition_priors(a['time'][train],a['y_mm'][train],a['case'][train],a['condition_raw'][train],meta['scalers']['time_scale'])
    torch.manual_seed(0);model=PIDGGCN().double();optim=torch.optim.RMSprop(model.parameters(),lr=.001)
    # Keep complete available same-case sensor histories including unlabeled
    # records. Only train_fit_mask labels enter supervised losses.
    sequences=[]
    for case in traincases:
        ids=np.flatnonzero((a['case']==case)&valid);ids=ids[np.argsort(a['time'][ids])]
        label=np.flatnonzero(a['train_fit_mask'][ids])
        assert len(label)
        sequences.append((ids,label))
    history=[];timing=time.perf_counter()
    for epoch in range(1,4):
        model.train();optim.zero_grad();losses=[];pieces=[]
        for ids,label in sequences:
            group_prior=group_priors[tuple(a['condition_raw'][ids[0]])]
            u=torch.tensor(a['time'][ids]/prior['time_scale'],dtype=torch.float64,requires_grad=True)
            sensor_condition=torch.tensor(a['X'][ids,:-1],dtype=torch.float64)
            time_feature=u*prior['time_scale']/meta['scalers']['time_scale']
            pred=model(torch.cat([sensor_condition,time_feature[:,None]],1))
            derivative=diagonal_time_derivative(pred,u,label)
            target=torch.tensor(a['y_z'][ids[label]],dtype=torch.float64)
            components=physics_loss(pred[label],target,derivative,u[label],group_prior['abc'],meta['scalers']['wear'][1])
            losses.append(sum(components));pieces.append([float(c.detach()) for c in components])
        loss=torch.stack(losses).mean();loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        norm=float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)))
        assert norm>0;optim.step()
        history.append(dict(epoch=epoch,total=float(loss.detach()),data_physics_monotonic=np.mean(pieces,axis=0).tolist(),gradient_norm=norm))
    training_seconds=time.perf_counter()-timing
    model.eval();before=model(x).detach();ck=ROOT/'smoke_checkpoint.pt';torch.save(model.state_dict(),ck)
    re=PIDGGCN().double().eval();re.load_state_dict(torch.load(ck,weights_only=True))
    torch.testing.assert_close(before,re(x),rtol=0,atol=0)
    result=dict(status='passed',formal_training_started=False,training_epochs=3,training_split='fold_a/inner_1',
      graph_shapes=[list(g.shape) for g in graphs],graph_nodes='features',
      prefix_invariance_error=prefix_error,future_sensor_perturbation_error=causal_error,
      full_sequence_graph_future_perturbation_effect=offline_effect,
      own_time_partial_finite_difference_error=derivative_error,corrected_prior_derivative_error=corrected_error,
      paper_printed_derivative_disagrees_with_calculus=True,checkpoint_reload_exact=True,
      loss_history=history,pooled_prior=prior,condition_priors=list(group_priors.values()),training_seconds=training_seconds,seconds_per_epoch=training_seconds/3,
      estimated_24_runs_100epochs_seconds=training_seconds/3*2400,
      total_seconds=time.perf_counter()-start,
      input_sha256=hashlib.sha256((path/'arrays.npz').read_bytes()).hexdigest())
    (ROOT/'smoke_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
