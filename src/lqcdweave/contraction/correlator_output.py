"""Coordinate complete-cfg execution and public result storage services."""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sys
from lqcdweave.contraction.execution.api import execute_cfg_plan
from lqcdweave.storage.correlator_results import (
    CfgOutputWriter, PackedDtOutputWriter, load_correlator_output,
    validate_cfg_output_binding,
)
from lqcdweave.contraction.correlator_cfg import bind_point_assemblers,assemble_instance,source_loader_factory,is_cfg_iterator,CfgSources


def run_dataset_calculation(calculation):
    plan=calculation.plan
    if not plan.output_fields or not plan.output_words:
        raise ValueError('dataset plan has no saved numeric packing layout; complete it offline')
    configuration=calculation.options.get('configuration')
    configurations=configuration if is_cfg_iterator(configuration) else (configuration,)
    original=source_loader_factory(calculation);bindings={};active={};published={}
    gpu=plan.statistics.get('postprocess_backend','cpu')=='gpu'
    from lqcdweave.contraction.correlator_postprocessing import GpuDatasetPostprocessing
    postprocessor=GpuDatasetPostprocessing(plan,bindings,debug=calculation.options.get('debug',False)) if gpu else None
    assemblers=() if gpu else bind_point_assemblers(calculation)
    @contextmanager
    def factory(cfg):
        with original(cfg) as loader:
            binding=getattr(loader,'output_binding',CfgSources({}))
            # Keep only tiny binding metadata for prefetched future cfgs.
            if plan.physical_lt is not None:
                validate_cfg_output_binding(plan,binding)
            bindings[cfg]=binding
            try: yield loader
            finally: bindings.pop(cfg,None)
    def consume(cfg,index,values):
        if cfg not in active:
            writer_type=CfgOutputWriter
            if plan.output_layout=='dt_packed':
                writer_type=PackedDtOutputWriter
            active[cfg]=writer_type(plan,cfg,Path(calculation.options['output']),bindings[cfg],
                debug=calculation.options.get('debug',False),prepacked=gpu)
        writer=active[cfg]
        if index is None:
            published[cfg]=dict(directory=str(writer.directory),configuration=str(cfg),data_id=(postprocessor.publish(cfg,writer) if gpu else writer.publish()))
            del active[cfg]
        elif gpu:
            if plan.output_layout!='dt_packed':
                writer.consume_packed(index,values[0])
        elif plan.output_coordinates:
            for position,value in assemble_instance(plan,assemblers,index,values):
                writer.consume(position,value)
        else:
            writer.consume(index,assemblers[plan.instances[index].template](values))
    try:
        from lqcdweave.contraction.correlator_cfg import execution_options
        report=execute_cfg_plan(plan,configurations,factory,consume,device_transform=postprocessor,
            **execution_options(calculation))
    except BaseException as error:
        for writer in active.values(): writer.abort(error)
        raise
    finally:
        if postprocessor is not None: postprocessor.close(primary=sys.exc_info()[1])
    if gpu:
        report.update({key:plan.statistics[key] for key in
            ('postprocess_backend','postprocess_device_bytes','postprocess_host_bytes')})
    return replace(calculation,value=published,report=report)


__all__=['load_correlator_output']
