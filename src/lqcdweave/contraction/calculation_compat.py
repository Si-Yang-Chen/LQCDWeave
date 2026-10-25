"""Historical result constructors and value protocols, isolated from execution."""
import numpy as np


class _ValueCompatibility(np.lib.mixins.NDArrayOperatorsMixin):
    """Only old array-returning session calls opt into the ndarray protocols."""

    def __getattr__(self, name):
        if name.startswith('_') or not self._legacy_array:
            raise AttributeError(name)
        return getattr(self.value, name)

    def __array__(self, dtype=None, copy=None):
        if not self._legacy_array:
            raise TypeError('use Calculation.value for numerical array conversion')
        return np.array(self.value, dtype=dtype, copy=copy) if copy is not None else np.asarray(self.value, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        from lqcdweave.contraction.calculation import Calculation
        if not self._legacy_array:
            return NotImplemented
        inputs = tuple(item.value if isinstance(item, Calculation) else item for item in inputs)
        if 'out' in kwargs:
            kwargs['out'] = tuple(item.value if isinstance(item, Calculation) else item for item in kwargs['out'])
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __getitem__(self, key):
        if self._legacy_pair:
            return (self.value, self.stats)[key]
        if not self._legacy_array:
            raise TypeError('use Calculation.value for numerical indexing')
        return self.value[key]

    def __setitem__(self, key, value):
        if not self._legacy_array or self._legacy_pair:
            raise TypeError('use Calculation.value for numerical assignment')
        self.value[key] = value

    def __bool__(self):
        return bool(self.value) if self._legacy_array and not self._legacy_pair else True

    def __len__(self):
        if self._legacy_pair:
            return 2
        if not self._legacy_array:
            raise TypeError('use Calculation.value for numerical length')
        return len(self.value)

    def __iter__(self):
        if self._legacy_pair:
            return iter((self.value, self.stats))
        if not self._legacy_array:
            raise TypeError('iterate Calculation.value for streamed or numerical results')
        return iter(self.value)


def session_result(value, stats, runner, *, array=False, pair=False, plan=None):
    """Adapt W's old returns without changing X's internal ExecutionResult."""
    from lqcdweave.contraction.calculation import Calculation
    report = dict(stats) if isinstance(stats, dict) else stats.as_dict()
    return Calculation(value=value, report=report, _runner=runner,
        _legacy_stats=stats, _legacy_array=array, _legacy_pair=pair, _session_plan=plan)


class _WindowResultType(type):
    def __instancecheck__(cls, value):
        from lqcdweave.contraction.calculation import Calculation
        return isinstance(value, Calculation) and value.execution_mode == 'cfg_windows'


class CfgWindowResult(metaclass=_WindowResultType):
    """Compatibility constructor; newly executed windows retain replay as well."""

    def __new__(cls, value, report):
        from lqcdweave.contraction.calculation import Calculation
        # This old constructor contains no selected work to replay.
        return Calculation(value=value, report=report, _window_request={})


class _CfgResultType(type):
    def __instancecheck__(cls, value):
        from lqcdweave.contraction.calculation import Calculation
        return isinstance(value, Calculation) and value._legacy_stats is not None


class CfgExecutionResult(metaclass=_CfgResultType):
    """Compatibility constructor with the old values/stats names."""

    def __new__(cls, values, stats):
        return session_result(values, stats, None)
