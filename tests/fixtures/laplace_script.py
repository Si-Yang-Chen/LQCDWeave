"""Test-owned production script composition; not an installed production API."""

from numbers import Integral, Real
import numpy as np
from lqcdweave.contracts.time_selection import dataset_time_selection, dataset_physical_times

def _checked_solver_options(polynomial_degree, lambda_cut, *, backend_available):
    if (isinstance(polynomial_degree, (bool, np.bool_)) or
            not isinstance(polynomial_degree, Integral) or polynomial_degree < 0):
        raise ValueError("polynomial_degree must be a nonnegative integer")
    degree = int(polynomial_degree)
    if isinstance(lambda_cut, (bool, np.bool_)) or not isinstance(lambda_cut, Real):
        raise ValueError("lambda_cut must be finite and in [0,12)")
    cutoff = float(lambda_cut)
    if not np.isfinite(cutoff) or not 0 <= cutoff < 12:
        raise ValueError("lambda_cut must be finite and in [0,12)")
    if (degree > 0) != (cutoff > 0):
        raise ValueError("polynomial_degree and lambda_cut must both be positive or both zero")
    if degree and not backend_available:
        raise NotImplementedError("Chebyshev acceleration requires an explicit LaplaceBackend")
    return degree, cutoff

def _host_owned_array(value, to_host):
    host = value if to_host is None else to_host(value)
    if not isinstance(host, np.ndarray):
        raise TypeError("to_host must return a NumPy ndarray")
    return np.array(host, copy=True, order="C")

def _check_row_writer(writer, laplace, to_host, polynomial_degree, lambda_cut,
                      *, needs_calc):
    if to_host is not None and not callable(to_host):
        raise TypeError("to_host must be a synchronous host-array callback")
    xp = getattr(laplace, "xp", None)
    if needs_calc and to_host is None and xp is not np:
        raise ValueError("non-NumPy array modules require an explicit to_host callback")
    backend_available = (getattr(laplace, "backend", None) is not None
                         if needs_calc else True)
    return _checked_solver_options(polynomial_degree, lambda_cut,
                                   backend_available=backend_available)

def produce_basis(writer, laplace, times, configuration,
                             calculated_values, *, to_host=None,
                             polynomial_degree=0, lambda_cut=0.0):
    """Write missing full-time basis rows and retain owned eigenvalue copies."""
    if getattr(writer, "spec", None) is not None and tuple(times) != dataset_physical_times(writer.spec):
        raise ValueError("writer selection differs from ordered physical calculation times")
    missing = tuple(writer.missing)
    if not missing:
        return
    if laplace is None:
        raise TypeError("Laplace generator is required for missing basis rows")
    degree, cutoff = _check_row_writer(writer, laplace, to_host,
        polynomial_degree, lambda_cut, needs_calc=True)
    laplace.load(configuration)
    for position in missing:
        time = times[position]
        vectors, values = laplace.calc(time, polynomial_degree=degree,
                                       lambda_cut=cutoff)
        host_vectors = _host_owned_array(vectors, to_host)
        host_values = _host_owned_array(values, to_host)
        writer.write(position, host_vectors)
        calculated_values[time] = np.array(host_values, copy=True, order="C")
        # The writer and the detached cache have consumed these owned rows.
        # Drop the current solve/host references before the next time slice.
        del vectors, values, host_vectors, host_values

def produce_eigenvalues(writer, laplace, times, configuration, ne,
                                  saved_basis, calculated_values, *, to_host=None,
                                  polynomial_degree=0, lambda_cut=0.0):
    """Write missing eigenvalue rows, checking any newly solved basis overlap."""
    if getattr(writer, "spec", None) is not None and tuple(times) != dataset_physical_times(writer.spec):
        raise ValueError("writer selection differs from ordered physical calculation times")
    missing = tuple(writer.missing)
    if not missing:
        return
    needs_calc = any(times[position] not in calculated_values for position in missing)
    if needs_calc and laplace is None:
        raise TypeError("Laplace generator is required for uncached eigenvalue rows")
    degree, cutoff = _check_row_writer(writer, laplace, to_host,
        polynomial_degree, lambda_cut, needs_calc=needs_calc)
    if needs_calc:
        laplace.load(configuration)
    for position in missing:
        time = times[position]
        if time in calculated_values:
            host_values = _host_owned_array(calculated_values[time], None)
            recomputed = False
        else:
            candidate, values = laplace.calc(time, polynomial_degree=degree,
                                             lambda_cut=cutoff)
            host_candidate = _host_owned_array(candidate, to_host)
            host_values = _host_owned_array(values, to_host)
            selection = (None if getattr(saved_basis, "spec", None) is None else dataset_time_selection(saved_basis.spec))
            stored = np.asarray(saved_basis.read(time=time if selection is None else selection.slot(time)))
            if (host_candidate.shape != stored.shape or
                    host_candidate.dtype != stored.dtype):
                raise ValueError("recomputed Laplace basis does not match the stored basis layout")
            overlaps = np.sum(stored.reshape(ne, -1).conj() *
                              host_candidate.reshape(ne, -1), axis=1)
            if not np.allclose(np.abs(overlaps), 1, atol=1e-6):
                raise ValueError("recomputed eigenvalues do not match stored basis")
            recomputed = True
        writer.write(position, host_values)
        if recomputed:
            del candidate, values, host_candidate, stored, overlaps, host_values
        else:
            del host_values
