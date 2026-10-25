"""Prepare one whole vertex using existing current-data member operations."""


def bind_linear_combination(backend, operations, coefficients, spec):
    """Bind ordered arithmetic, without reading or retaining any input array.

    Members already implement source adjoints, profile transforms and casts.
    The output and one reusable member buffer are owned by this preparation;
    no member value survives the call. There is no global component axis.
    """
    operations, coefficients = tuple(operations), tuple(coefficients)
    if len(operations) != len(coefficients) or not operations:
        raise ValueError('whole vertex requires matching members and coefficients')

    def operation(source, blocks, destination=None):
        direct = destination is not None and destination.dtype == spec.dtype
        result = destination if direct else backend.empty(spec.shape, dtype=spec.dtype)
        scratch = None
        for index, (member, coefficient) in enumerate(zip(operations, coefficients)):
            if index == 0:
                member(None, blocks, result)
                if coefficient != 1:
                    backend.multiply(result, coefficient, out=result)
            else:
                if scratch is None:
                    scratch = backend.empty(spec.shape, dtype=spec.dtype)
                member(None, blocks, scratch)
                if coefficient != 1:
                    backend.multiply(scratch, coefficient, out=scratch)
                backend.add(result, scratch, out=result)
        if destination is not None and not direct:
            backend.copyto(destination, result, casting='unsafe')
            return destination
        return result

    return operation
