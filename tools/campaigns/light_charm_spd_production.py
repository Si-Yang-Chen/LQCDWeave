"""Source-averaged light I0/I1, connected charmonium and D production.

No scheduler submission. Run one MPI rank per Dispatch consumer. All compatible
operators share one selected contraction path; bounded windows overlap CPU
binding with CPU/GPU execution instead of expanding all coordinate graphs.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import sys

for _thread_variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_variable,"1")

import numpy as np
import light_charm_spd as campaign


BASE = Path("/dg_hpc/LQCD/DATA/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05766_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104")
ELEMENTAL = "04.meson.basis55.mom33.nev120.t40.stout20rho0.12.v2"
LIGHT = "03.perambulator.light.single.prec1e-9.nev70.t40.select_t.v1"


def compact_relative_vsv(raw, selection, destination):
    """Convert through the native reader to the exact absolute selected slots.

    This is an input conversion, with no mode truncation or precision cast.
    Consumers subsequently use the same declared selection as the elemental.
    """
    shape = raw.sectors["vsv"].shape
    if raw.time_convention != "relative" or shape[:2] != (selection.Lt, selection.Lt):
        raise ValueError("expected original full relative-time VSV")
    target = np.lib.format.open_memmap(destination, mode="w+", dtype=raw.sectors["vsv"].dtype,
                                      shape=(selection.Nt, selection.Nt, *shape[2:]))
    try:
        for source, source_time in enumerate(selection.times):
            for sink, sink_time in enumerate(selection.times):
                target[source, sink] = raw.block(sink_time, source_time, "vsv")
        target.flush()
    finally:
        target._mmap.close()


def propagators(light, charm):
    result={}
    for flavor,reader in (("u",light),("d",light),("q",light),("c",charm)):
        result[f"S^{flavor}"]=reader
        result[rf"S^{flavor}_\mathrm{{local}}"]=reader
    return result


@contextmanager
def inputs(base, cfg, scratch):
    from lqcdweave.adapters.propagator import ArrayPropagator
    from lqcdweave.contracts.time_selection import TimeSelectionRecord
    from lqcdweave.storage.dataset import DatasetReader
    # Close mmap readers before their job-local conversion directory is removed.
    with TemporaryDirectory(prefix=f"light-charm-{cfg}-",dir=scratch) as temporary:
        with ExitStack() as scope:
            dataset=scope.enter_context(DatasetReader(base/ELEMENTAL,cfg=cfg))
            elemental=campaign.elemental_reader(dataset)
            selection=elemental.time_selection
            sidecar=json.loads((base/LIGHT/f"{cfg}.json").read_text())
            if tuple(sidecar["physical_times"])!=selection.times:
                raise ValueError("light and elemental ordered physical time selections differ")
            light=scope.enter_context(ArrayPropagator.from_npy(
                {"vsv":base/LIGHT/f"{cfg}.npy"},cfg=cfg,time_convention="absolute",
                source_selection=selection,sink_selection=selection))
            canonical=TimeSelectionRecord(128,tuple(range(128)))
            raw_path=base/"03.perambulator.charm.single.prec1e-15.nev120"/(
                f"clqcd_nf2_clov_L16_T128_b2.0_xi5_ml-0.05766_cfg_{cfg}.peram.charm.nev120.npy")
            raw=scope.enter_context(ArrayPropagator.from_npy({"vsv":raw_path},cfg=cfg,
                time_convention="relative",source_selection=canonical,sink_selection=canonical))
            compact=Path(temporary)/"charm-absolute.npy"
            compact_relative_vsv(raw,selection,compact)
            charm=scope.enter_context(ArrayPropagator.from_npy({"vsv":compact},cfg=cfg,
                time_convention="absolute",source_selection=selection,sink_selection=selection))
            if light.sectors["vsv"].shape!=(40,40,4,4,70,70):
                raise ValueError("light must contain all forty slots and seventy modes")
            if charm.sectors["vsv"].shape!=(40,40,4,4,120,120):
                raise ValueError("charm must contain all forty slots and 120 modes")
            yield elemental,light,charm,selection,dict(elemental_data_id=dataset.data_id,
                light_file=str(base/LIGHT/f"{cfg}.npy"),charm_file=str(raw_path),
                actual_perambulator_basis_identity="not certified by legacy metadata")


def operator_metadata(fields):
    return {"/".join(path):[dict(flavor=r["flavor"],operator=r["operator"],
        radial=r["radial"],c_parity=r["c_parity"],radial_exchange=r["radial_exchange"],
        terms=[dict(tag=t["tag"],coefficient=str(t["coefficient"]),
            mode_indices=t["mode_indices"]) for t in r["terms"]])
        for r in field["operators"]] for path,field in fields.items()}


def read_pair_results(directory, cfg, requests):
    """Read native pair fields with one validated Dataset registration."""
    from lqcdweave.storage.dataset import DatasetReader
    values={}
    with DatasetReader(directory,cfg=cfg) as reader:
        definition=reader.spec.document['definition']
        if (definition.get('encoding')!='native-hdf5-fields-v1'
                or definition.get('output_layout')=='dt_packed'):
            raise ValueError('production requires native pair output')
        for group,channels in requests.items():
            for channel,rows in channels.items():
                for row,fields in rows.items():
                    target=values.setdefault(group,{}).setdefault(channel,{}).setdefault(row,{})
                    for name in fields:
                        target[name]=reader.read_field(('pairs',group,channel,row,name))
    return values


def replay_statistics(report):
    names=("execution_mode","coordinate_count","planned_windows","peak_bound_windows",
        "first_window_wait_seconds","planning_wait_seconds","planning_receive_seconds",
        "planning_restore_seconds","peak_planned_host_bytes","peak_planned_device_bytes",
        "total_seconds","output_coverage","allocator_headroom")
    return {name:report[name] for name in names if name in report}


def save_source_average(output, cfg, values, selection, *, metadata, provenance):
    """Only reshape declared pair order; physical averaging is storage-owned.

    Loops retain forty absolute times to permit ensemble vacuum subtraction.
    The scalar disconnected *two-point* product is averaged like two_point.
    """
    from lqcdweave.contraction.correlator import save_correlator
    pairs=campaign.periodic_pairs(selection.times)
    averaged={};loops={}
    for group,channels in values.items():
        for channel,rows in channels.items():
            for row,fields in rows.items():
                target=averaged.setdefault(group,{}).setdefault(channel,{}).setdefault(row,{})
                for name,value in fields.items():
                    if value.shape[-1]!=len(pairs):
                        raise ValueError("result does not cover all 400 declared pairs")
                    if name in ("two_point","disconnected"):
                        target[name]=value.reshape((*value.shape[:-1],40,10))
                    else:
                        role={"vacuum_sink":"sink","vacuum_source":"source"}[name]
                        first={}
                        for i,pair in enumerate(pairs):
                            first.setdefault(pair[role],i)
                        array=np.take(value,[first[s] for s in range(40)],axis=-1)
                        if not np.isfinite(array).all():
                            raise ValueError("nonfinite scalar one-point loop")
                        loops.setdefault(group,{}).setdefault(channel,{}).setdefault(row,{})[name]=array
    mean_id=save_correlator(Path(output)/"source_average",averaged,configuration=cfg,
        axes=("source_slot","dt"),coordinates={"source_slot":list(range(40)),"dt":list(range(1,11))},
        average_tsrc=True,source_axis="source_slot",relative_axis="dt",physical_lt=128,
        boundary="periodic",coordinate_domain="slot",selection=selection,
        time_layout="cyclic_difference",dt_coordinates=range(1,33),metadata=metadata,provenance=provenance)
    loop_id=None
    if loops:
        loop_id=save_correlator(Path(output)/"scalar_loops",loops,configuration=cfg,
            axes=("time_slot",),coordinates={"time_slot":list(range(40))},
            metadata=dict(metadata,physical_times=list(selection.times),
                normalization="I0=(ubar u+dbar d)/sqrt(2); vacuum=-sqrt(2)*Tr(V*S); source uses daggered V"),
            provenance=provenance)
    return dict(source_average=mean_id,scalar_loops=loop_id)


def run(args):
    from lqcdweave.adapters.mpi_dispatch import Dispatch, combine
    from lqcdweave.contraction.correlator import save_calculation_plan
    if any(os.environ.get(name,"1")!="1" for name in
            ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")):
        raise ValueError("set OMP/OPENBLAS/MKL_NUM_THREADS=1 before launching")
    args.scratch.mkdir(parents=True,exist_ok=True)
    tasks=Dispatch(str(args.cfg_file),args.suffix)
    if tasks.comm.Get_size()!=1:
        raise ValueError("one MPI rank per independent cfg consumer is required")
    template=None;fields=None;tags=None
    with ExitStack() as representative:
        try:
            for cfg in tasks:
                cfg=str(cfg);started=perf_counter()
                timings={}
                def phase(name):
                    print(json.dumps(dict(cfg=cfg,phase=name,elapsed_seconds=perf_counter()-started)),flush=True)
                with ExitStack() as current:
                    owner=representative if template is None else current
                    phase('input-start')
                    stage=perf_counter()
                    elemental,light,charm,selection,provenance=owner.enter_context(
                        inputs(args.data_root,cfg,args.scratch))
                    timings['input_seconds']=perf_counter()-stage
                    phase('inputs-ready')
                    if template is None:
                        stage=perf_counter()
                        requests,fields,tags=campaign.build_task(elemental=elemental)
                        timings['operator_definition_seconds']=perf_counter()-stage
                        phase('joint-planning-start')
                        stage=perf_counter()
                        template=campaign.plan(requests,elemental,light,charm,tags,configuration=cfg,
                            backend=args.backend,memory_limit=args.device_gib<<30 if args.backend=="cupy"
                                else args.host_gib<<30,host_memory_limit=args.host_gib<<30)
                        timings['joint_planning_seconds']=perf_counter()-stage
                        phase('joint-plan-ready')
                        archive=args.output/"plans"/f"{args.suffix}-{os.getpid()}-{cfg}.json.gz"
                        archive.parent.mkdir(parents=True,exist_ok=True)
                        stage=perf_counter()
                        save_calculation_plan(archive,template)
                        timings['archive_seconds']=perf_counter()-stage
                    pairs=campaign.periodic_pairs(selection.times)
                    # One call includes all 400 pairs and the entire operator basis.
                    # No channel/sink calculate loops and no OOM-to-scalar fallback.
                    # The existing Dataset entry binds bounded windows from the
                    # same joint template; it starts execution before binding
                    # all remaining coordinates. Pair payloads are job-local
                    # intermediates. Persistent products keep the same means
                    # and forty-time scalar loops as before.
                    with TemporaryDirectory(prefix=f"pairs-{cfg}-",dir=args.scratch) as temporary:
                        phase('fixed-window-replay-start')
                        stage=perf_counter()
                        result=template.run(output=temporary,configuration=cfg,vertex_map=elemental,
                            propagator_map=propagators(light,charm),time_pairs=pairs,
                            physical_times=selection.times,physical_lt=selection.Lt,
                            time_roles=("sink","source"))
                        timings['coordinate_binding_and_replay_seconds']=perf_counter()-stage
                        phase('all-400-pairs-assembled')
                        stage=perf_counter()
                        values=read_pair_results(temporary,cfg,requests)
                        timings['pair_result_read_seconds']=perf_counter()-stage
                        metadata=dict(definition="light-charm-SPD-BS-source-mean-v2",
                            operators=operator_metadata(fields),physical_times=list(selection.times),
                            light_edges=campaign.LIGHT_EDGES,charm_edges=campaign.CHARM_EDGES,
                            charm_disconnected=False,light_charm_mixing=False,vacuum_subtracted=False,
                            requested_physical_dt=[1,32],source_slots=list(range(40)),
                            fixed_replay=replay_statistics(result.report))
                        stage=perf_counter()
                        save_source_average(args.output,cfg,values,selection,
                            metadata=metadata,provenance=provenance)
                        timings['source_average_and_publication_seconds']=perf_counter()-stage
                        del values
                    timings['total_seconds']=perf_counter()-started
                    timing_dir=args.output/'timing';timing_dir.mkdir(parents=True,exist_ok=True)
                    (timing_dir/f'{cfg}.json').write_text(json.dumps(dict(configuration=cfg,
                        time_pairs=len(pairs),timings=timings,
                        fixed_replay=replay_statistics(result.report)),indent=2)+'\n')
                    del result
                    combine(str(args.output/f"completed-{args.suffix}.txt"),cfg)
                    print(json.dumps(dict(cfg=cfg,status="saved",seconds=perf_counter()-started)),flush=True)
        finally:
            if template is not None:
                template.close()


def main():
    logger=logging.getLogger('lqcdweave')
    logger.setLevel(logging.INFO)
    handler=logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(handler)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-file",type=Path,required=True)
    parser.add_argument("--suffix",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--scratch",type=Path,required=True)
    parser.add_argument("--data-root",type=Path,default=BASE)
    parser.add_argument("--backend",choices=("numpy","cupy"),default="numpy")
    parser.add_argument("--host-gib",type=int,default=80)
    parser.add_argument("--device-gib",type=int,default=24)
    run(parser.parse_args())


if __name__=="__main__":
    main()
