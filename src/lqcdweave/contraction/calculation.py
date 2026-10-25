"""The common W result and replay handle; numerical services stay in E/P/X/R."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from lqcdweave.contracts.correlator.api import CfgPlan
from lqcdweave.contraction.execution.api import ExecutionBindingCache, CfgExecutionCache, RunResources
from lqcdweave.contraction.results.api import OutputAssemblyCache
from lqcdweave.contraction.calculation_compat import _ValueCompatibility


class _Report(dict):
    """Mapping contract with attribute access to an older measured report."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            execution = self.get('execution')
            if execution is not None:
                return getattr(execution, name)
            raise AttributeError(name) from None


@dataclass(frozen=True, init=False)
class Calculation(_ValueCompatibility):
    """Selected work, its latest value and a mapping of execution evidence.

    ``run()`` returns another Calculation, sharing selected work, structural
    bindings and a bounded idle device pool. This handle's value is unchanged.
    The historical CompleteCalculation constructor is accepted here too.
    """

    expanded: object = None
    path: object = None
    value: object = None
    report: object = None
    options: object = field(default_factory=dict)
    _cfg_plan: object = field(default=None, repr=False)
    _execution_bindings: object = field(default=None, repr=False, compare=False, kw_only=True)
    _execution_resources: object = field(default=None, repr=False, compare=False, kw_only=True)
    _output_bindings: object = field(default=None, repr=False, compare=False, kw_only=True)
    _window_request: object = field(default=None, repr=False, compare=False, kw_only=True)
    _runner: object = field(default=None, repr=False, compare=False, kw_only=True)
    _legacy_stats: object = field(default=None, repr=False, compare=False, kw_only=True)
    _legacy_pair: bool = field(default=False, repr=False, compare=False, kw_only=True)
    _legacy_array: bool = field(default=False, repr=False, compare=False, kw_only=True)
    _session_plan: object = field(default=None, repr=False, compare=False, kw_only=True)

    def __init__(self, *args, **kwargs):
        # Keep both positional constructors and dataclasses.replace working.
        names = (('plan', 'value', 'report', 'options') if args and isinstance(args[0], CfgPlan)
                 else ('expanded', 'path', 'value', 'report', 'options'))
        if len(args) > len(names):
            raise TypeError('too many positional arguments for Calculation')
        for name, value in zip(names, args):
            if name in kwargs:
                raise TypeError(f'multiple values for {name}')
            kwargs[name] = value
        selected = kwargs.pop('plan', kwargs.get('_cfg_plan'))
        if selected is not None and not isinstance(selected, CfgPlan):
            raise TypeError('plan= requires a CfgPlan; fixed work uses path=')
        kwargs['_cfg_plan'] = selected
        unknown = kwargs.keys() - self.__dataclass_fields__.keys()
        if unknown:
            raise TypeError(f'unknown Calculation arguments: {sorted(unknown)}')
        for name, definition in self.__dataclass_fields__.items():
            object.__setattr__(self, name, kwargs.get(name, definition.default))
        object.__setattr__(self, 'options', MappingProxyType(dict(kwargs.get('options') or {})))
        gram_tiling = self.options.get('factorized_gram_tiling',False)
        if (self._execution_bindings is None
                or getattr(self._execution_bindings,'factorized_gram_tiling',False) != gram_tiling):
            cache_type = CfgExecutionCache if selected is not None else ExecutionBindingCache
            cache = cache_type(factorized_gram_tiling=gram_tiling)
            object.__setattr__(self, '_execution_bindings', cache)
        if self._output_bindings is None:
            object.__setattr__(self, '_output_bindings', OutputAssemblyCache())
        if self._execution_resources is None:
            object.__setattr__(self, '_execution_resources', RunResources())
        report = kwargs.get('report')
        if report is None:
            report = {}
        elif not isinstance(report, Mapping):
            report = _Report(execution=report)
        elif not isinstance(report, dict):
            report = dict(report)
        # A lazy cfg executor updates this same dictionary as it is consumed.
        report.setdefault('status', 'planned' if self.value is None else 'completed')
        report.setdefault('execution_mode', self.execution_mode)
        report.setdefault('output_mode', self.output_mode)
        object.__setattr__(self, 'report', report)

    @property
    def plan(self):
        return self._cfg_plan if self._cfg_plan is not None else (self.path.plan if self.path is not None else self._session_plan)

    @property
    def inputs(self):
        return self.expanded.inputs if self.expanded is not None else None

    @property
    def search(self):
        return self.path.search if self.path is not None else None

    @property
    def execution_mode(self):
        if self._window_request is not None:
            return 'cfg_windows'
        if self._cfg_plan is not None:
            return 'complete_cfg'
        return self.path.execution_mode if self.path is not None else 'session'

    @property
    def output_mode(self):
        if self._window_request is not None:
            return 'dataset'
        return self._cfg_plan.output_mode if self._cfg_plan is not None else 'memory'

    @property
    def prediction(self):
        if self._cfg_plan is not None:
            plan = self._cfg_plan
            return dict(plan.statistics, peak_host_bytes=plan.peak_host_bytes,
                output_host_bytes=plan.output_host_bytes, prefetch_host_bytes=plan.prefetch_host_bytes,
                peak_device_bytes=plan.peak_device_bytes, prefetch_device_bytes=plan.prefetch_device_bytes,
                execution_target=plan.target_summary)
        return self.path.prediction if self.path is not None else {}

    @property
    def planning_seconds(self):
        return self.path.planning_seconds if self.path is not None else self.options.get('planning_seconds', 0.0)

    @property
    def values(self):
        """Compatibility spelling of value."""
        return self.value

    @property
    def stats(self):
        """Compatibility access to the original session statistics."""
        return self._legacy_stats if self._legacy_stats is not None else self.report

    def run(self, *, backend=None, measure=False, debug=None, device_memory_limit=None, **bindings):
        """Replay through the common bounded resource owner."""
        effective_debug = self.options.get('debug', False) if debug is None else debug
        if effective_debug:
            self._execution_resources.close()
            return self._run(backend=backend, measure=measure, debug=debug,
                device_memory_limit=device_memory_limit, **bindings)
        with self._execution_resources.activate():
            return self._run(backend=backend, measure=measure, debug=debug,
                device_memory_limit=device_memory_limit, **bindings)

    def close(self):
        """Release this handle's idle device pool; existing values stay valid."""
        self._execution_resources.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _run(self, *, backend=None, measure=False, debug=None, device_memory_limit=None, **bindings):
        """Replay the selected work; return a Calculation with fresh results.

        Fixed work accepts current cfg output bindings. Complete cfg work uses
        its saved readers/options. A window result retains its full request and
        accepts replacements for those same runtime arguments, e.g. output.
        """
        if self._runner is not None:
            if bindings or measure or device_memory_limit is not None:
                raise ValueError('legacy session replay does not accept cfg bindings, calibration or a device limit override')
            if debug is None:
                return self._runner(backend=backend)
            if type(debug) is not bool:
                raise TypeError('debug must be bool')
            from lqcdweave.contraction.execution.api import execution_debug
            with execution_debug(debug):
                return self._runner(backend=backend)
        if self._window_request is not None:
            if not self._window_request:
                raise ValueError('this compatibility result has no selected work to replay')
            if measure or device_memory_limit is not None:
                raise ValueError('window replay does not support calibration or a device limit override')
            from lqcdweave.contraction.correlator_window_run import run_cfg_window_calculation
            request = dict(self._window_request, **bindings)
            if backend is not None:
                request['backend'] = backend
            if debug is not None:
                request['debug'] = debug
            return run_cfg_window_calculation(self, **request)
        if self._cfg_plan is not None:
            if bindings:
                raise ValueError('complete cfg replay uses its saved runtime bindings; bind them when loading')
            from lqcdweave.contraction.correlator_cfg import _run_complete_calculation
            return _run_complete_calculation(self, backend=backend, measure=measure,
                device_memory_limit=device_memory_limit, debug=debug)
        if device_memory_limit is not None:
            raise ValueError('fixed replay uses the selected path memory limits')
        if self.path is None:
            raise ValueError('this compatibility result has no selected work to replay')
        from lqcdweave.contraction.correlator import _run_fixed_calculation
        return _run_fixed_calculation(self, backend=backend, measure=measure, debug=debug, **bindings)


class _CfgValueStream:
    """Preserve laziness, explicit close and the live report identity."""

    def __init__(self, values, report):
        self._values, self._report = values, report
        self._finished = False
        report['status'] = 'pending'

    def __iter__(self):
        return self

    def __next__(self):
        if self._finished:
            raise StopIteration
        self._report['status'] = 'running'
        try:
            return next(self._values)
        except StopIteration:
            self._finished = True
            self._report['status'] = 'completed'
            raise
        except BaseException:
            self._finished = True
            self._report['status'] = 'failed'
            raise

    def close(self):
        try:
            self._values.close()
        finally:
            self._finished = True
            if self._report['status'] not in ('completed', 'failed'):
                self._report['status'] = 'closed'
