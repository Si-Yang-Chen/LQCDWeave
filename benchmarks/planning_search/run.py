"""Fixed-work CPU search benchmark; excludes storage scheduling and numerical IO."""
from pathlib import Path
import sys,os,time,json,gzip,socket,hashlib,traceback,functools,argparse,subprocess,platform
from dataclasses import replace
from collections import defaultdict
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,required=True)
parser.add_argument('--sha256',required=True)
parser.add_argument('--source',type=Path,required=True,help='checkout to benchmark')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--max-states',type=int,default=1536)
parser.add_argument('--deadline',type=float,default=600)
parser.add_argument('--first-path',action='store_true')
parser.add_argument('--profile',action='store_true',help='instrumented diagnostic; do not compare with formal timing')
args=parser.parse_args()
if args.max_states<1 or args.deadline<=0:parser.error('positive budget required')
SOURCE=args.source.resolve()
sys.path.insert(0,str(SOURCE/'src'))
from path_fingerprint import fingerprint
profiled=args.profile
OUT=args.output.resolve()
if OUT.exists():parser.error('output already exists')
OUT.parent.mkdir(parents=True,exist_ok=True)
def git(*arguments):
    return subprocess.check_output(['git','-C',str(SOURCE),*arguments],text=True).strip()
R=dict(status='starting',scope='first path' if args.first_path else 'continued DFS',
       phases={},timings={},cpu_affinity=[],source_commit=git('rev-parse','HEAD'),
       source_status=git('status','--porcelain'),python=sys.version,platform=platform.platform(),
       harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       max_states=args.max_states,deadline_seconds=args.deadline,profiled=profiled,
       settings=dict(root_width=4,branch_width=4,include_ties=True,scorer='time_per_term',component_workers=1,kernel_batching=False),
       environment={k:os.environ.get(k) for k in ('PYTHONHASHSEED','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})

def save():OUT.write_text(json.dumps(R,indent=2,default=str))
def rss():
    values={}
    if not Path('/proc/self/status').exists():return {'unavailable':True}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:','VmHWM:')):values[line.split(':')[0]+'_kib']=int(line.split()[1])
    return values
stack=[];timings=defaultdict(lambda:dict(calls=0,inclusive=0.0,exclusive=0.0))
def timed(name,func):
    @functools.wraps(func)
    def wrapped(*args,**kwargs):
        started=time.perf_counter();frame=[0.0];stack.append(frame)
        try:return func(*args,**kwargs)
        finally:
            elapsed=time.perf_counter()-started;stack.pop()
            row=timings[name];row['calls']+=1;row['inclusive']+=elapsed;row['exclusive']+=elapsed-frame[0]
            if stack:stack[-1][0]+=elapsed
    return wrapped
class Complete(Exception):pass
def main():
    if hasattr(os,'sched_getaffinity'):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
        R['cpu_affinity']=sorted(os.sched_getaffinity(0))
    R['host']=socket.gethostname()
    from lqcdweave.storage.contraction_plans.api import decode_plan_value
    from lqcdweave.contraction.planning.optimizers import graph_session as gs
    from lqcdweave.contraction.planning.optimizers.integer_space import IntegerSpace
    from lqcdweave.contraction.planning.optimizers.graph_dfs import FingerprintDFS
    from lqcdweave.contraction.planning.optimizers.budget import retained_size
    from lqcdweave.contraction.planning.optimizers.integer_ranking import CandidateRanking
    from lqcdweave.contraction.planning.optimizers import selected_path
    archive=args.archive
    R['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    if R['archive_sha256']!=args.sha256:raise ValueError('archive SHA256 mismatch')
    t=time.perf_counter();doc=json.load(gzip.open(archive,'rt'))['payload'];baseline=replace(decode_plan_value(doc['plans'][0]),blocks=decode_plan_value(doc['blocks']))
    R['phases']['load_seconds']=time.perf_counter()-t
    problem=baseline.state.problem
    R['inputs']=dict(terms=len(problem.terms),inputs=len(problem.specs),blocks=len(baseline.blocks),dtype=sorted({str(s.dtype) for s in problem.specs.values()}),model=baseline.cost_profile.hardware_fingerprint)
    R['memory_after_load']=rss()
    original_advance=FingerprintDFS.advance
    def first_only(self,*,first_solution=False):
        t=time.perf_counter()
        try:return original_advance(self,first_solution=(True if args.first_path else first_solution))
        finally:R['phases']['search_advance_seconds']=R['phases'].get('search_advance_seconds',0)+time.perf_counter()-t
    FingerprintDFS.advance=first_only
    original_finish=FingerprintDFS.finish
    def finish(self,*args,**kwargs):
        R['memory_at_search_end']=rss();R['live_search_depth']=max(0,len(self.frames)-1)
        if profiled:
            t=time.perf_counter()
            R['retained_graph_and_tail_cache_bytes']=retained_size((self.nodes,self.connections,self.frames,self.operations,getattr(self.space,'_terminal_completions',None)))
            R['memory_measurement_seconds']=time.perf_counter()-t
        return original_finish(self,*args,**kwargs)
    FingerprintDFS.finish=finish
    def coarse(name,func):
        def wrapped(*args,**kwargs):
            t=time.perf_counter()
            try:return func(*args,**kwargs)
            finally:R['phases'][name]=R['phases'].get(name,0)+time.perf_counter()-t
        return wrapped
    gs.shared_components=coarse('classification_seconds',gs.shared_components)
    selected_path.materialize_dense_path=coarse('action_materialization_seconds',selected_path.materialize_dense_path)
    if hasattr(FingerprintDFS,'selected_operations'):
        FingerprintDFS.selected_operations=coarse('winning_operation_expansion_seconds',FingerprintDFS.selected_operations)
    if profiled:
        for cls,names in ((IntegerSpace,('_index','_rebuild','apply','undo','fingerprint','rank_candidates')),(FingerprintDFS,('rank_candidates','_walk')),(CandidateRanking,('_refresh','select'))):
            for name in names:setattr(cls,name,timed(cls.__name__+'.'+name,getattr(cls,name)))
        original=IntegerSpace._part
        IntegerSpace._part=staticmethod(timed('IntegerSpace._part',original))
        from lqcdweave.contraction.planning.optimizers.terminal_completion import TerminalCompletions
        for name in ('complete','_family_options'):
            setattr(TerminalCompletions,name,timed('TerminalCompletions.'+name,getattr(TerminalCompletions,name)))
        TerminalCompletions._choose=staticmethod(timed('TerminalCompletions._choose',TerminalCompletions._choose))
    def capture(physical,audited,snippets,workspaces,*,stats,**kwargs):
        R['phases']['plan_until_audit_seconds']=time.perf_counter()-started
        assert audited.done and audited.problem.terms==problem.terms
        R.update(status='complete',statistics={k:v for k,v in stats.items() if k not in ('component_scenes','component_statistics')},compute_seconds=audited.seconds,steps=len(snippets),actions=len(audited.actions),memory_after_audit=rss())
        t=time.perf_counter();R['fingerprint']=fingerprint(problem,audited.actions)[0];R['phases']['fingerprint_seconds']=time.perf_counter()-t
        R['timings']=dict(timings);save();raise Complete()
    gs.finish_fixed_storage=capture
    R['status']='searching';save();started=time.perf_counter()
    try:gs.solve_graph_storage(problem,baseline.blocks,baseline.preparations,profile=baseline.cost_profile,memory_limit=baseline.memory_limit,host_memory_limit=baseline.host_memory_limit,reserved_host_bytes=baseline.reserved_host_bytes,reserved_device_bytes=baseline.reserved_device_bytes,component_workers=1,root_width=4,branch_width=4,include_ties=True,scorer='time_per_term',deadline_seconds=args.deadline,max_states=args.max_states,kernel_batching=False)
    except Complete:pass
    if R['status']!='complete':raise RuntimeError('benchmark did not reach audited delivery')
    print(json.dumps(dict(status=R['status'],phases=R['phases'],compute_seconds=R.get('compute_seconds'))),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:R.update(status='failed',error=traceback.format_exc());save();raise
