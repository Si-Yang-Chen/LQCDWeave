"""Test-owned numerical comparison of a delivered route and independent oracle."""

from lqcdweave.contraction.planning.optimizers.verification import VerificationError

def verify_numeric_replay(state, loader, *, reference_state=None, oracle=None,
                          sources=None, rtol=1e-12, atol=1e-12, backend=None):
    """Replay one Action state with real NumPy and compare the arrays.

    Either reference_state (a second, independently built Action state) or
    oracle (a zero-argument callable returning the expected arrays) must be
    supplied.  A symbolic state built without explicit load actions needs its
    source tensors in sources; it is copied per replay and forwarded as the
    execution's initial array table.  Returns the replayed arrays.
    """
    import numpy as np
    from lqcdweave.contraction.execution.api import replay as execute
    if reference_state is None and oracle is None:
        raise ValueError("numeric verification needs a reference_state or an oracle")
    backend = np if backend is None else backend

    def run(target):
        shared = None if sources is None else dict(sources)
        return tuple(execute(target, loader, backend=backend, shared_arrays=shared))

    values = run(state)
    if reference_state is not None:
        other = run(reference_state)
        if len(values) != len(other):
            raise VerificationError(
                f"numeric replay produced {len(values)} outputs != reference {len(other)}")
        for index, (left, right) in enumerate(zip(values, other)):
            if not np.allclose(left, right, rtol=rtol, atol=atol):
                raise VerificationError(
                    f"numeric replay output {index} is not allclose to the reference state")
    if oracle is not None:
        expected = oracle()
        expected = (expected,) if isinstance(expected, np.ndarray) else tuple(expected)
        if len(values) != len(expected):
            raise VerificationError(
                f"numeric replay produced {len(values)} outputs != oracle {len(expected)}")
        for index, (left, right) in enumerate(zip(values, expected)):
            if not np.allclose(left, right, rtol=rtol, atol=atol):
                raise VerificationError(
                    f"numeric replay output {index} is not allclose to the oracle")
    return values
