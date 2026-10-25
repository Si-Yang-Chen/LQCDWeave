"""Complete cfg payload within the existing versioned plan archive envelope."""
from .api import (
    _read_document, _write_document, _make_envelope, _decode_envelope,
    PlanArchiveError, _decode_plan_value,
)
from lqcdweave.contracts.correlator.api import CfgPlan
from .cfg_stream import needs_stream, write_stream, read_stream_if_present
from .compact import encode


def save_cfg_plan(path,plan,*,debug=False):
    """Save P's finalized descriptor; an independent audit is diagnostic only."""
    if not isinstance(plan,CfgPlan): raise TypeError('expected a complete CfgPlan')
    if debug:
        from lqcdweave.contraction.planning.api import audit_cfg_plan
        audit_cfg_plan(plan)
    if needs_stream(plan):
        return write_stream(path, plan)
    # The same positional schemas/defaults as the large stream keep small cfg
    # JSON compact too. Preserve the exact original immutable reference graph,
    # including Actions and equal-but-distinct description objects.
    payload={'encoding':'unified-device-v2','plan':encode(plan,preserve_identity=True)}
    return _write_document(path,_make_envelope('complete-cfg-plan',payload))


def load_cfg_plan(source):
    """Decode immutable arrangements; no expansion, path or resource planning."""
    from lqcdweave.contraction.planning.api import validate_cfg_allocation_envelope
    streamed = read_stream_if_present(source)
    if streamed is not None:
        return validate_cfg_allocation_envelope(streamed)
    payload=_decode_envelope(_read_document(source),expected_type='complete-cfg-plan')
    if set(payload)=={'plan'} or payload.get('encoding')=='shared-contracts-v1':
        raise PlanArchiveError('old matrix/complete cfg plan format is unsupported; generate a new plan (no migration)')
    elif set(payload)=={'plan','encoding'} and payload['encoding']=='unified-device-v2':
        plan=_decode_plan_value(payload['plan'],shared={},_consume=True)
    else: raise PlanArchiveError('invalid complete cfg payload fields or encoding')
    if not isinstance(plan,CfgPlan): raise PlanArchiveError('archive contains no complete cfg plan')
    return validate_cfg_allocation_envelope(plan)


__all__=['save_cfg_plan','load_cfg_plan']
