"""Fixed task replay through W's existing bounded cfg output service."""
import logging
from lqcdweave.contracts.correlator.api import describe_coordinates
from lqcdweave.contraction.expansion.api import describe_input_reads, input_readers
from lqcdweave.contraction.correlator_cfg import CfgSources, readers_for_requests
from lqcdweave.contraction.correlator_window_run import run_cfg_window_calculation


def run_selected_cfg(calculation, *, output, configuration, propagator_map,
                     vertex_map, time_axes, time_pairs, physical_times,
                     physical_lt, time_roles, output_definition, backend, debug):
    """Change runtime bindings and coordinates, never operators or search options."""
    if propagator_map is None and vertex_map is None:
        readers = input_readers(calculation.inputs)
    else:
        readers = readers_for_requests(describe_input_reads(calculation.inputs),
                                      propagator_map, vertex_map)
    coordinates = calculation.expanded.coordinates
    if time_axes is not None or time_pairs is not None:
        coordinates, _ = describe_coordinates(time_axes=time_axes, time_pairs=time_pairs)
    logger = logging.getLogger('lqcdweave')
    cfg = calculation.expanded.bindings.configuration if configuration is None else configuration

    def progress(phase, details):
        logger.info('cfg=%s fixed-replay-%s %s', cfg, phase, details)
    return run_cfg_window_calculation(calculation, coordinates=coordinates,
        output=output, configuration=configuration,
        readers=CfgSources(readers, tuple(physical_times), physical_lt),
        physical_lt=physical_lt, time_roles=time_roles, output_definition=output_definition,
        # Ordinary replay preserves all pairs; dt averaging remains explicit.
        dt_fields=(), backend=backend, debug=debug, progress=progress)
