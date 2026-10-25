"""Real GPU replay from CPU readers through neutral B-owned preparations."""
from dataclasses import replace

import numpy as np
import pytest

try:
    import cupy as cp
except ImportError as error:
    from unittest import SkipTest
    raise SkipTest('CuPy is required for real GPU replay tests') from error

from lqcdweave.contraction.execution.api import execute_contraction_path
from lqcdweave.contraction.expansion.api import BaryonDiagramBinding, expand_correlator
from lqcdweave.contraction.kernel_descriptions import attach_vertex_kernel_descriptions
from lqcdweave.contraction.planning.api import plan_direct_execution, retarget_expanded
from lqcdweave.contraction.results.api import assemble_correlator
from lqcdweave.contracts.correlator.api import DiagramBindings
from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
from lqcdweave.storage.contraction_plans.api import save_contraction_plan, load_contraction_plan
from test_baryon_diagram_frontend import _case, _independent_oracle, _spin_entries


@pytest.mark.parametrize('multiple', (False, True))
def test_real_host_baryon_source_adjoint_components_archive_and_current_data(tmp_path, multiple):
    case = _case((2, 0, 1))
    extra_case = None
    if multiple:
        sink = case['vertices']['sink']
        first = sink.operator.terms[0]
        extra = OperatorTerm.from_baryon(first.component, _spin_entries(8),
                                        momentum=first.momentum, coefficient=-.2+.3j)
        sink.operator = Operator.from_terms('sum-sink', (first, extra))
        extra_case = dict(case, sink_spin=extra.spin.dense(), sink_coefficient=extra.coefficient)
    # The B objects deliberately retain array_module=np and actual CPU readers.
    expanded = expand_correlator(case['diagram'], adapter=BaryonDiagramBinding(sector_scope='dense-v'),
        bindings=DiagramBindings(case['propagators'], case['vertices'], 'cfg-a'),
        time_axes=None, time_pairs=None, batch='auto')
    target = retarget_expanded(attach_vertex_kernel_descriptions(expanded), 'cupy')
    selected = replace(plan_direct_execution(target, memory_limit=1 << 27, host_memory_limit=1 << 27),
                       execution_mode='planned')
    hardware = {'backend': 'cupy'}
    destination = tmp_path / 'real-baryon-gpu.json.gz'
    save_contraction_plan(destination, selected, hardware=hardware)
    path = load_contraction_plan(destination, expected=target, hardware=hardware)
    previous = None
    for repeat in range(2):
        if repeat:
            for array in case['arrays']:
                array *= 1.1-.2j
        expected = _independent_oracle(case)
        if extra_case is not None:
            expected += _independent_oracle(extra_case)
        result = execute_contraction_path(target, path, backend=cp, io_overlap=True)
        actual = assemble_correlator(target, result.term_values)
        np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=1e-10)
        if previous is not None:
            assert not np.allclose(previous, actual)
        previous = np.array(actual, copy=True)
