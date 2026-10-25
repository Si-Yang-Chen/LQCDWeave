"""Resolve one public Gram strategy and its older boolean spelling."""


def resolve_gram_strategy(strategy, tiling):
    if strategy is not None and strategy not in ('reduction','tiled','gemm'):
        raise ValueError('factorized_gram_strategy must be reduction, tiled, gemm or None')
    if tiling is not None:
        if type(tiling) is not bool:
            raise TypeError('factorized_gram_tiling must be bool or None')
        legacy='tiled' if tiling else 'reduction'
        if strategy is not None and strategy!=legacy:
            raise ValueError('factorized_gram_strategy conflicts with factorized_gram_tiling')
        strategy=legacy
    return strategy


def new_gram_strategy(requested, runtime):
    """New CUDA P uses the admitted measured-domain candidate, never old archives."""
    return requested if requested is not None else ('gemm' if runtime == 'cuda' else 'reduction')


def archived_gram_strategy(calculation, requested):
    """X cannot introduce a new unbudgeted numerical implementation on load."""
    from lqcdweave.contracts.correlator.api import CfgPlan,FactorizedWideReduction
    plan=calculation.plan
    plans=(tuple(t.plan for t in plan.templates) if type(plan) is CfgPlan else
           (plan,) if plan is not None else ())
    has_gemm=any(type(a.core) is FactorizedWideReduction and a.core.gram_strategy=='gemm'
                 for item in plans for a in item.state.actions)
    if requested is not None and (requested=='gemm')!=has_gemm:
        raise ValueError('archive fixes the numerical Gram strategy; prepare a new plan offline to change it')
    return requested or ('gemm' if has_gemm else 'reduction')
