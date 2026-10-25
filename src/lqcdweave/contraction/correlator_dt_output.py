"""Select packed output by composing the public planning/storage services."""
from dataclasses import replace
from lqcdweave.contraction.planning.api import configure_cfg_output_layout


def configure_packed_dt(calculation,layout,coordinates):
    """Unpack Calculation metadata for the pure offline output planner."""
    from lqcdweave.contraction.correlator_cfg import _host_cap
    current=calculation.plan
    cap=(_host_cap(current) if layout=='dt_packed' and current.output_mode=='dataset'
         and current.physical_lt and current.dt_words else None)
    plan=configure_cfg_output_layout(current,layout,coordinates,host_cap=cap)
    return calculation if plan is calculation.plan else replace(calculation,plan=plan)
