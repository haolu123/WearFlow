from .monitoring import MLPMonitor

from .dynamics import WearDynamics

from .coupled import CoupledMonitor

from .zhang import ZhangMonitor, ZhangPhysics

from .time_conditioned import EncodedMLPMonitor, FiLMMonitor, SensorAttentionMonitor



MONITORS = {'mlp': MLPMonitor, 'zhang': ZhangMonitor,

            'time_encoded': EncodedMLPMonitor, 'film': FiLMMonitor, 'sensor_attention': SensorAttentionMonitor}





def build_dynamics(cfg):

    kind = cfg.get('transition', 'ode')

    if kind in ('direct', 'direct_positive'):

        from .direct_transition import DirectTransition

        return DirectTransition(cfg['hidden_dim'], nonnegative=kind == 'direct_positive')

    if kind != 'ode':

        raise ValueError(f'Unknown transition {kind}')

    return WearDynamics(**{k: cfg[k] for k in ('hidden_dim', 'rk4_steps', 'nonnegative_rate')})





def build_model(cfg):

    m = cfg['monitor']

    if m['architecture']=='gated_feature_mlp':

        if cfg['dynamics']['mode']!='none':raise ValueError('Gate experiment is monitoring only')

        from .gated_features import GatedFeatureMLP

        return GatedFeatureMLP(m)

    if m['architecture']=='feature_mlp':

        if cfg['dynamics']['mode']!='none':raise ValueError('Feature sweep is monitoring only')

        from .feature_mlp import FeatureMLP

        return FeatureMLP(m)

    if m['architecture'].startswith('history_'):

        if cfg['dynamics']['mode'] != 'none':

            raise ValueError('Phase 1 history models must be pure supervised monitors')

        from .history import HistoryCoupled

        return HistoryCoupled(m)

    monitor = MONITORS[m['architecture']](m['inputs'], m['hidden_dims'])

    if m.get('selected_features') is not None:

        from .feature_subset import FeatureSubsetMonitor

        monitor = FeatureSubsetMonitor(monitor, m['architecture'], m['selected_features'])

    dynamics = build_dynamics(cfg['dynamics']) if cfg['dynamics']['mode'] != 'none' else None

    model = CoupledMonitor(monitor, dynamics)

    if cfg.get('zhang', {}).get('physics', False):

        if m['architecture'] != 'zhang' or dynamics is not None:

            raise ValueError('Author PINN requires Zhang monitor and no ODE branch')

        model.zhang_physics = ZhangPhysics(monitor.input_dim)

    return model

