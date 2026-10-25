"""Pure CostProfile fitting, operation calibration, and weighted fusion."""

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import math
from numbers import Real
from statistics import median

import numpy as np
from lqcdweave.contracts.measurement import ReplayReport

from .model import CostProfile


@dataclass(frozen=True)
class ProfileFitReport:
    """One saved operation report with user-declared grouping metadata."""

    report: object
    source_id: str
    run_id: str | None = None
    shape_family: str | None = None
    weight: object = None
    reference: str | None = None


def _fit_nonnegative(rows, targets, sample_weights=None):
    """Small weighted NNLS; leave underdetermined models unchanged."""
    design = np.asarray(rows, dtype=np.float64)
    observed = np.asarray(targets, dtype=np.float64)
    if design.ndim != 2 or len(observed) != design.shape[0]:
        return None
    if sample_weights is None:
        weights = np.ones(len(observed), dtype=np.float64)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if (weights.ndim != 1 or len(weights) != len(observed)
                or np.any(~np.isfinite(weights)) or np.any(weights <= 0)):
            raise ValueError("NNLS sample weights must be finite, positive, and match the rows")
        weights = weights / np.max(weights)
    if design.shape[1] == 0 or len(observed) < design.shape[1]:
        return None
    scales = np.maximum(np.max(np.abs(design), axis=0), 1e-30)
    normalized = design / scales
    sqrt_weights = np.sqrt(weights)
    weighted_design = normalized * sqrt_weights[:, None]
    weighted_observed = observed * sqrt_weights
    if np.linalg.matrix_rank(weighted_design) < design.shape[1]:
        return None
    best = None
    for mask in range(1, 1 << design.shape[1]):
        columns = [index for index in range(design.shape[1]) if mask & (1 << index)]
        estimate = np.linalg.lstsq(weighted_design[:, columns], weighted_observed, rcond=None)[0]
        if np.any(estimate < 0):
            continue
        coefficients = np.zeros(design.shape[1])
        coefficients[columns] = estimate
        error = float(np.square(weighted_design @ coefficients - weighted_observed).sum())
        if best is None or error < best[0]:
            best = error, coefficients / scales
    return None if best is None else best[1]


def execution_cost_samples(report):
    """Extract preparation, snapshot, and numerical kernel measurements."""
    execution_profile = getattr(report, 'execution_profile', None)
    if isinstance(execution_profile, dict):
        calibration_eligible = execution_profile.get('calibration_eligible')
    else:
        calibration_eligible = getattr(execution_profile, 'calibration_eligible', None)
    if calibration_eligible is False:
        raise ValueError(
            'direct contraction reports are not calibration-eligible; '
            'fit only planned replay reports with independently validated predictors'
        )
    samples = []
    for timing in report.timings:
        features = timing.model_features
        if (timing.kind == 'sparse_bridge' and features
                and (features.get('sparse_operation') == 'bounded_coo_join'
                     or features.get('join_id') == 'bounded_coo_join')):
            # The bounded streaming join uses a conservative structural upper
            # bound until it has its own validated cost model. Never train the
            # legacy single/double-cycle coefficients from these timings.
            continue
        if timing.kind == 'load':
            for phase in timing.phases:
                if phase.name == 'materialize':
                    if not features or not features.get('storage_preparation'):
                        raise ValueError('materialization requires explicit preparation predictors')
                    fields = dict(preparation_call_seconds=1.,
                                  preparation_kernel_call_seconds=features['kernel_calls'],
                                  preparation_flop_seconds=features['flops'],
                                  preparation_traffic_seconds_per_byte=features['traffic_bytes'])
                    samples.append(dict(category='source_preparation', fields=fields,
                                        dtype=features.get('dtype'),
                                        observed_seconds=phase.observed_seconds))
                elif phase.name == 'snapshot':
                    if phase.bytes is None:
                        raise ValueError('snapshot calibration requires byte count')
                    samples.append(dict(category='logical_snapshot',
                        fields=dict(snapshot_call_seconds=1.,
                                    snapshot_seconds_per_byte=phase.bytes),
                        observed_seconds=phase.observed_seconds))
        elif timing.kind in ('contract', 'sparse_bridge', 'baryon_join'):
            if not features:
                raise ValueError('contraction calibration requires predictors')
            if features['kernel_calls'] == 0:
                continue  # Reused results launch no kernel.
            call_field = 'kernel_call_seconds_' + features['dtype']
            if call_field not in CostProfile.__dataclass_fields__:
                raise ValueError('unsupported calibration dtype')
            if timing.kind == 'contract':
                fields = {call_field: features['kernel_calls'],
                          'flop_seconds': features['flops'],
                          'traffic_seconds_per_byte': features['traffic_bytes']}
                category = 'dense_contraction'
            elif timing.kind == 'sparse_bridge':
                fields = dict(sparse_kernel_call_scale=features['kernel_calls'],
                              sparse_traffic_seconds_per_byte=features['traffic_bytes'])
                fields['sparse_cycle_flop_seconds' if len(timing.operands) == 4
                       else 'sparse_flop_seconds'] = features['flops']
                category = 'sparse_contraction'
            else:
                fields = {call_field: features['kernel_calls'],
                          'flop_seconds': features['flops'],
                          'traffic_seconds_per_byte': features['traffic_bytes']}
                category = 'baryon_join'
            samples.append(dict(category=category, fields=fields,
                                dtype=features['dtype'],
                                observed_seconds=timing.observed_seconds))
        elif timing.kind not in ('release',):
            raise ValueError(f'unsupported storage calibration action: {timing.kind}')
    return tuple(samples)


def evaluate_cost_profile(profile, samples):
    """Return per-operation model errors for the supplied compute samples."""
    result = {}
    for sample in samples:
        predicted = 0.
        for field, amount in sample['fields'].items():
            coefficient = getattr(profile, field)
            if field.startswith('preparation_') and coefficient is None:
                coefficient = {
                    'preparation_kernel_call_seconds': profile.kernel_call_for_dtype(
                        sample.get('dtype', 'complex128') or 'complex128'),
                    'preparation_flop_seconds': profile.flop_seconds,
                    'preparation_traffic_seconds_per_byte': profile.traffic_seconds_per_byte,
                }[field] * profile.contract_scale
            if field == 'sparse_kernel_call_scale':
                coefficient *= profile.kernel_call_for_dtype(sample['dtype'])
            predicted += amount * coefficient
        category = sample['category']
        scale = {
            'dense_contraction': profile.contract_scale,
            'sparse_contraction': profile.sparse_bridge_scale,
            'baryon_join': profile.baryon_join_scale,
            'source_preparation': 1.,
            'logical_snapshot': 1.,
        }.get(category)
        if scale is None:
            raise ValueError(f'unsupported compute-cost sample category: {category}')
        predicted *= scale
        observed = sample['observed_seconds']
        row = result.setdefault(category, dict(predicted_seconds=0., observed_seconds=0.,
                                               absolute_error_seconds=0., samples=0))
        row['predicted_seconds'] += predicted
        row['observed_seconds'] += observed
        row['absolute_error_seconds'] += abs(predicted - observed)
        row['samples'] += 1
    return result


_CALIBRATION_CATEGORIES = (
    'source_preparation', 'logical_snapshot', 'dense_contraction',
    'sparse_contraction', 'baryon_join',
)
_UNSUPPORTED_CALIBRATION_CATEGORIES = {'baryon_join'}


def _normalized_source_weights(sources):
    raw = {}
    for source in sources:
        if not isinstance(source, ProfileFitReport):
            raise TypeError('fit sources must be ProfileFitReport records')
        if not isinstance(source.source_id, str) or not source.source_id.strip():
            raise ValueError('training source_id must be nonempty')
        if isinstance(source.weight, bool) or not isinstance(source.weight, Real):
            raise TypeError('training source weights must be real numbers')
        try:
            weight = float(source.weight)
        except (OverflowError, ValueError) as exc:
            raise ValueError('training source weights must be finite') from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('training source weights must be finite and nonnegative')
        if source.source_id in raw and raw[source.source_id] != weight:
            raise ValueError('all reports for one source_id must use the same source weight')
        raw[source.source_id] = weight
    maximum = max(raw.values(), default=0.)
    if maximum <= 0:
        raise ValueError('at least one training source weight must be positive')
    scaled = {source_id: weight / maximum for source_id, weight in raw.items() if weight > 0}
    denominator = math.fsum(scaled.values())
    return raw, {source_id: weight / denominator for source_id, weight in scaled.items()}


def _validate_sample(sample):
    if not isinstance(sample.get('category'), str) or not sample['category']:
        raise ValueError('calibration samples need an operation category')
    observed = sample.get('observed_seconds')
    if not isinstance(observed, (int, float)) or not math.isfinite(observed) or observed < 0:
        raise ValueError('invalid measured time')
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
           for value in sample['fields'].values()):
        raise ValueError('invalid model predictor')


def _calibrate_source_samples(profile, source_samples, raw_source_weights,
                              normalized_source_weights, *, hardware_fingerprint):
    if profile.hardware_fingerprint not in ('unmeasured', hardware_fingerprint):
        raise ValueError('calibration hardware differs from profile')
    if any(value != 1 for value in (profile.contract_scale, profile.sparse_bridge_scale)):
        raise ValueError('calibration requires unit base scales')
    for samples in source_samples.values():
        for sample in samples:
            _validate_sample(sample)
    total_samples = sum(len(samples) for samples in source_samples.values())
    if total_samples == 0:
        raise ValueError('no calibration samples')
    updates = dict(
        preparation_kernel_call_seconds=(profile.kernel_call_for_dtype('complex128')
            if profile.preparation_kernel_call_seconds is None else profile.preparation_kernel_call_seconds),
        preparation_flop_seconds=(profile.flop_seconds
            if profile.preparation_flop_seconds is None else profile.preparation_flop_seconds),
        preparation_traffic_seconds_per_byte=(profile.traffic_seconds_per_byte
            if profile.preparation_traffic_seconds_per_byte is None else profile.preparation_traffic_seconds_per_byte),
    )
    per_source_categories = {
        source_id: Counter(sample['category'] for sample in samples)
        for source_id, samples in source_samples.items()
    }
    all_categories = set(_CALIBRATION_CATEGORIES)
    all_categories.update(category for counts in per_source_categories.values() for category in counts)
    diagnostics = {}
    coverage = {}
    ordered_categories = sorted(all_categories, key=lambda item: item == 'sparse_contraction')
    for category in ordered_categories:
        selected = [(source_id, sample)
                    for source_id, samples in source_samples.items()
                    for sample in samples if sample['category'] == category]
        sample_counts = Counter(source_id for source_id, _ in selected)
        active_sources = tuple(sorted(sample_counts))
        effective_weights = {source_id: normalized_source_weights[source_id]
                             for source_id in active_sources}
        fields = sorted({field for _, sample in selected
                         for field, amount in sample['fields'].items() if amount > 0})
        status = 'base-unobserved' if not selected else 'base-insufficient'
        fitted = False
        lower_bounds = {}
        if category in _UNSUPPORTED_CALIBRATION_CATEGORIES:
            status = 'unsupported'
        elif selected and fields:
            rows = [[sample['fields'].get(field, 0.) for field in fields]
                    for _, sample in selected]
            targets = np.asarray([sample['observed_seconds'] for _, sample in selected])
            sample_weights = [normalized_source_weights[source_id] / sample_counts[source_id]
                              for source_id, _ in selected]
            if category == 'sparse_contraction' and 'sparse_kernel_call_scale' in fields:
                fitted_dense = replace(profile, **updates)
                call_index = fields.index('sparse_kernel_call_scale')
                for row, (_, sample) in zip(rows, selected, strict=True):
                    row[call_index] *= fitted_dense.kernel_call_for_dtype(sample['dtype'])
            bounded = replace(profile, **updates)
            floors = np.asarray([
                getattr(bounded, field)
                if ('flop_seconds' in field or 'traffic_seconds_per_byte' in field) else 0.
                for field in fields
            ], dtype=np.float64)
            lower_bounds = dict(zip(fields, map(float, floors), strict=True))
            fit = _fit_nonnegative(rows, targets - np.asarray(rows) @ floors,
                                   sample_weights=sample_weights)
            if fit is not None:
                updates.update(zip(fields, map(float, fit + floors), strict=True))
                status = 'fitted'
                fitted = True
        diagnostics[category] = dict(samples=len(selected), fitted=fitted,
                                     fields=fields, lower_bounds=lower_bounds,
                                     status=status,
                                     source_weights=effective_weights,
                                     source_sample_counts=dict(sample_counts))
        coverage[category] = dict(
            status=status,
            prediction_available=True,
            sample_count=len(selected),
            source_count=len(active_sources),
            source_weights=effective_weights,
            source_sample_counts=dict(sample_counts),
            fields=fields,
            coefficient_basis='fitted' if fitted else 'base-profile',
        )
    fitted_profile = replace(
        profile, **updates, version=f'{profile.version}-cost-fit',
        hardware_fingerprint=hardware_fingerprint, sample_count=total_samples,
    )
    all_samples = tuple(sample for source_id in source_samples for sample in source_samples[source_id])
    training_by_source = {
        source_id: evaluate_cost_profile(fitted_profile, samples)
        for source_id, samples in source_samples.items()
    }
    for category, row in coverage.items():
        errors_by_source = {
            source_id: metrics[category]
            for source_id, metrics in training_by_source.items()
            if category in metrics
        }
        active_weight = math.fsum(normalized_source_weights[source_id]
                                  for source_id in errors_by_source)
        weighted_mae = (
            math.fsum(normalized_source_weights[source_id]
                      * metrics[category]['absolute_error_seconds']
                      / metrics[category]['samples']
                      for source_id, metrics in training_by_source.items()
                      if category in metrics) / active_weight
            if active_weight > 0 else None
        )
        row['training_errors_by_source'] = errors_by_source
        row['active_source_weight_sum'] = active_weight
        row['weighted_training_mean_absolute_error_seconds'] = weighted_mae
    return fitted_profile, dict(
        groups=diagnostics,
        operation_coverage=coverage,
        training=evaluate_cost_profile(fitted_profile, all_samples),
        training_by_source=training_by_source,
        source_weights={source_id: normalized_source_weights[source_id]
                        for source_id in sorted(normalized_source_weights)},
        raw_source_weights={source_id: raw_source_weights[source_id]
                            for source_id in sorted(normalized_source_weights)},
        source_sample_counts={source_id: len(source_samples[source_id])
                              for source_id in sorted(source_samples)},
        scope='Measured disjoint execution phases; excludes search, hashing, publication and unassigned overhead',
    )


def calibrate_cost_profile(profile, reports, *, verified, hardware_fingerprint,
                           source_weights=None, source_ids=None):
    """Fit saved operation reports with equal-report or explicit source weights.

    This compatibility API accepts reports directly. Every report is its own
    source unless source_ids groups multiple reports under one weight.
    """
    if verified is not True or not isinstance(hardware_fingerprint, str) or not hardware_fingerprint.strip():
        raise ValueError('independently verified results and hardware identity required')
    reports = tuple(reports)
    if not reports:
        raise ValueError('no calibration reports')
    if source_ids is None:
        source_ids = tuple(f'report-{index}' for index in range(len(reports)))
    else:
        source_ids = tuple(source_ids)
    if len(source_ids) != len(reports):
        raise ValueError('source_ids must have one entry per report')
    if source_weights is None:
        weights = (1.0,) * len(reports)
    elif isinstance(source_weights, dict):
        weights = tuple(source_weights[source_id] for source_id in source_ids)
    else:
        weights = tuple(source_weights)
    if len(weights) != len(reports):
        raise ValueError('source_weights must have one entry per report')
    sources = tuple(ProfileFitReport(report, source_id, run_id=f'legacy-run-{index}', weight=weight)
                    for index, (report, source_id, weight) in enumerate(zip(reports, source_ids, weights, strict=True)))
    raw_weights, normalized_weights = _normalized_source_weights(sources)
    source_samples = defaultdict(list)
    for source in sources:
        if raw_weights[source.source_id] > 0:
            source_samples[source.source_id].extend(execution_cost_samples(source.report))
    return _calibrate_source_samples(profile, source_samples, raw_weights,
                                     normalized_weights,
                                     hardware_fingerprint=hardware_fingerprint)


def fit_cost_profile_sources(base_profile, training_sources, validation_sources, *,
                             verified, hardware_fingerprint, validation_by='run'):
    """Weighted source fit with explicit run or shape-family holdout groups."""
    if not isinstance(base_profile, CostProfile):
        raise TypeError('base_profile must be CostProfile')
    if verified is not True:
        raise ValueError('independent numerical verification must be explicitly asserted')
    if not isinstance(hardware_fingerprint, str) or not hardware_fingerprint.strip():
        raise ValueError('hardware_fingerprint is required')
    if validation_by not in {'run', 'shape_family'}:
        raise ValueError("validation_by must be 'run' or 'shape_family'")
    training_sources = tuple(training_sources)
    validation_sources = tuple(validation_sources)
    if not training_sources or not validation_sources:
        raise ValueError('at least one training and one validation report are required')
    raw_weights, normalized_weights = _normalized_source_weights(training_sources)
    active_sources = tuple(source for source in training_sources
                           if raw_weights[source.source_id] > 0)
    for source in validation_sources:
        if not isinstance(source, ProfileFitReport):
            raise TypeError('validation sources must be ProfileFitReport records')
        if not isinstance(source.source_id, str) or not source.source_id.strip():
            raise ValueError('validation source_id must be nonempty')
    group_field = 'run_id' if validation_by == 'run' else 'shape_family'
    for source in (*active_sources, *validation_sources):
        value = getattr(source, group_field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{group_field} is required for every active report')
    training_groups = {getattr(source, group_field) for source in active_sources}
    validation_groups = {getattr(source, group_field) for source in validation_sources}
    overlap = training_groups & validation_groups
    if overlap:
        raise ValueError(f'training and validation {group_field} groups overlap: {sorted(overlap)}')
    training_samples_by_source = defaultdict(list)
    training_signatures = set()
    for source in active_sources:
        samples = execution_cost_samples(source.report)
        if not samples:
            continue
        signature = _sample_signature(samples)
        if signature in training_signatures:
            raise ValueError('duplicate operation evidence appears in training reports')
        training_signatures.add(signature)
        training_samples_by_source[source.source_id].extend(samples)
    if not any(training_samples_by_source.values()):
        raise ValueError('training reports contain no supported operation measurements')
    validation_samples_by_report = {}
    validation_signatures = set()
    for index, source in enumerate(validation_sources):
        samples = execution_cost_samples(source.report)
        if not samples:
            raise ValueError('validation report contains no supported operation measurements')
        signature = _sample_signature(samples)
        if signature in training_signatures or signature in validation_signatures:
            raise ValueError('duplicate operation evidence appears in validation reports')
        validation_signatures.add(signature)
        key = source.reference or f'{source.source_id}@{source.run_id or source.shape_family or index}'
        if key in validation_samples_by_report:
            raise ValueError(f'duplicate validation report label {key!r}')
        validation_samples_by_report[key] = (source, samples)
    fitted, diagnostics = _calibrate_source_samples(
        base_profile, training_samples_by_source, raw_weights, normalized_weights,
        hardware_fingerprint=hardware_fingerprint,
    )
    validation_by_report = {
        key: evaluate_cost_profile(fitted, samples)
        for key, (_, samples) in validation_samples_by_report.items()
    }
    for category, row in diagnostics['operation_coverage'].items():
        row['validation_errors_by_report'] = {
            report_key: metrics[category]
            for report_key, metrics in validation_by_report.items()
            if category in metrics
        }
    combined_validation = tuple(
        sample for _, samples in validation_samples_by_report.values() for sample in samples
    )
    diagnostics = dict(
        diagnostics,
        validation=evaluate_cost_profile(fitted, combined_validation),
        validation_by_report=validation_by_report,
        validation_by=validation_by,
        validation_groups=sorted(validation_groups),
        training_groups=sorted(training_groups),
        grouping_identity='user-declared; ReplayReport cannot authenticate run or shape-family metadata',
    )
    return fitted, diagnostics


def _sample_signature(samples):
    items = []
    for sample in samples:
        items.append((sample['category'], tuple(sorted(sample['fields'].items())),
                      sample.get('dtype'), float(sample['observed_seconds'])))
    return tuple(sorted(items, key=repr))


def fit_cost_profile(base_profile, training_report, validation_report, *, verified,
                     hardware_fingerprint, training_run_id, validation_run_id):
    """Backward-compatible single-training-source fit by run holdout."""
    if not isinstance(training_run_id, str) or not training_run_id.strip():
        raise ValueError('training_run_id must be a nonempty user-supplied run identity')
    if not isinstance(validation_run_id, str) or not validation_run_id.strip():
        raise ValueError('validation_run_id must be a nonempty user-supplied run identity')
    if training_run_id == validation_run_id:
        raise ValueError('training and validation run identities must differ')
    if training_report is validation_report:
        raise ValueError('training and validation reports must be separate run records')
    training = ProfileFitReport(training_report, training_run_id, run_id=training_run_id, weight=1.0)
    validation = ProfileFitReport(validation_report, validation_run_id, run_id=validation_run_id)
    return fit_cost_profile_sources(
        base_profile, (training,), (validation,), verified=verified,
        hardware_fingerprint=hardware_fingerprint, validation_by='run',
    )



_KERNEL_FIELDS = (
    "kernel_call_seconds",
    "kernel_call_seconds_float32",
    "kernel_call_seconds_float64",
    "kernel_call_seconds_complex64",
    "kernel_call_seconds_complex128",
)
_KERNEL_DTYPES = (None, "float32", "float64", "complex64", "complex128")
_SNAPSHOT_FIELDS = (
    "snapshot_call_seconds",
    "snapshot_seconds_per_byte",
)
_PREDICTION_STRUCTURE = "CostProfile.compute-model/v1"

def normalize_profile_weights(pairs):
    """Validate and normalize references without resolving or reading them.

    Return positive entries as ``(reference, input_weight, normalized_weight)``
    plus the input mode. This is also used by workflow code so zero-weight
    files are never loaded.
    """
    try:
        raw_items = tuple(pairs)
    except TypeError as exc:
        raise TypeError("profiles must be profile references or (profile, weight) pairs") from exc
    if not raw_items:
        raise ValueError("at least one profile is required")
    pair_flags = [isinstance(item, (tuple, list)) and len(item) == 2 for item in raw_items]
    if any(pair_flags) and not all(pair_flags):
        raise ValueError("do not mix equally weighted profiles with explicit (profile, weight) pairs")
    if all(pair_flags):
        raw_pairs = raw_items
        weight_mode = "explicit"
    else:
        raw_pairs = tuple((item, 1.0) for item in raw_items)
        weight_mode = "equal"
    parsed = []
    for item in raw_pairs:
        try:
            source, weight = item
        except (TypeError, ValueError) as exc:
            raise ValueError("each profile component must be a (profile, weight) pair") from exc
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise TypeError("profile weights must be real numbers")
        try:
            value = float(weight)
        except (OverflowError, ValueError) as exc:
            raise ValueError("profile weights must be finite real numbers") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError("profile weights must be finite and nonnegative")
        parsed.append((source, value))
    maximum = max(weight for _, weight in parsed)
    if maximum == 0:
        raise ValueError("at least one profile weight must be positive")
    scaled = [weight / maximum for _, weight in parsed]
    denominator = math.fsum(scaled)
    normalized = [value / denominator for value in scaled]
    components = [(source, weight, norm)
                  for (source, weight), norm in zip(parsed, normalized, strict=True)
                  if weight > 0]
    return components, weight_mode


def _weighted_value(values, weights):
    """Stable convex average for finite nonnegative prediction coefficients."""
    maximum = max(values)
    if maximum == 0:
        return 0.0
    normalized = math.fsum(weight * (value / maximum)
                           for value, weight in zip(values, weights, strict=True))
    return maximum * min(normalized, 1.0)


def _average(profiles, weights, field):
    return _weighted_value([float(getattr(profile, field)) for profile in profiles], weights)


def _same(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-15)


def _product(*values):
    result = 1.0
    for value in values:
        result *= float(value)
        if not math.isfinite(result):
            raise ValueError("effective CostProfile prediction coefficient overflowed")
    return result


def _common_factor(numerators, denominators, label):
    factors = []
    for numerator, denominator in zip(numerators, denominators, strict=True):
        if denominator == 0:
            if numerator != 0:
                raise ValueError(
                    f"weighted {label} predictions cannot be represented by CostProfile fields"
                )
            continue
        factor = numerator / denominator
        if not math.isfinite(factor):
            raise ValueError(f"weighted {label} feature ratio is not finite")
        factors.append(factor)
    if not factors:
        return 1.0
    factor = factors[0]
    if any(not _same(value, factor) for value in factors[1:]):
        raise ValueError(
            f"weighted {label} predictions require incompatible feature ratios; "
            "CostProfile cannot represent the fused model"
        )
    return factor


def _prep_coefficient(profiles, weights, field, *, dtype=None):
    values = []
    for profile in profiles:
        value = getattr(profile, field)
        if value is None:
            if field == "preparation_kernel_call_seconds":
                value = _product(profile.contract_scale, profile.kernel_call_for_dtype(dtype))
            elif field == "preparation_flop_seconds":
                value = _product(profile.contract_scale, profile.flop_seconds)
            elif field == "preparation_traffic_seconds_per_byte":
                value = _product(profile.contract_scale, profile.traffic_seconds_per_byte)
            else:  # pragma: no cover - all optional fields are enumerated above
                raise AssertionError(field)
        values.append(float(value))
    return _weighted_value(values, weights)


def _merge_prediction_models(profiles, weights, operation_weights=None,
                             operation_provenance=None):
    """Fuse actual linear prediction coefficients, resolving model factors."""
    operation_weights = operation_weights or {}
    operation_provenance = operation_provenance or {}

    def weights_for(category):
        return operation_weights.get(category, weights)

    updates = {name: 1.0 for name in (
        "contract_scale", "sparse_bridge_scale", "sparse_kernel_call_scale",
    )}

    for field in _SNAPSHOT_FIELDS:
        updates[field] = _average(profiles, weights_for("logical_snapshot"), field)

    # Dense contractions and inherited preparation coefficients share these
    # model terms. The result uses contract_scale=1, with effective rates.
    dense_kernel = {}
    for field, dtype in zip(_KERNEL_FIELDS, _KERNEL_DTYPES, strict=True):
        effective = [_product(profile.contract_scale, profile.kernel_call_for_dtype(dtype))
                     for profile in profiles]
        dense_kernel[field] = _weighted_value(effective, weights_for("dense_contraction"))
        updates[field] = dense_kernel[field]
    for field in ("flop_seconds", "traffic_seconds_per_byte"):
        updates[field] = _weighted_value(
            [_product(profile.contract_scale, getattr(profile, field)) for profile in profiles],
            weights_for("dense_contraction"),
        )

    # Sparse bridge calls reuse dtype-specific dense kernel rates, while the
    # sparse bridge and call scales multiply them. A single resulting ratio
    # must represent every supported dtype.
    sparse_call = [
        _weighted_value([
            _product(profile.sparse_bridge_scale, profile.sparse_kernel_call_scale,
                     profile.kernel_call_for_dtype(dtype))
            for profile in profiles
        ], weights_for("sparse_contraction"))
        for dtype in _KERNEL_DTYPES
    ]
    if operation_provenance.get("sparse_contraction", {}).get("weight_policy") == "base-fallback-global-weights":
        # There is no measured sparse coverage to fuse. Preserve a clearly
        # labeled fallback scale instead of rejecting a result because its
        # shared dense kernel coefficients came from a different covered set.
        updates["sparse_kernel_call_scale"] = _weighted_value(
            [_product(profile.sparse_bridge_scale, profile.sparse_kernel_call_scale)
             for profile in profiles], weights_for("sparse_contraction")
        )
    else:
        updates["sparse_kernel_call_scale"] = _common_factor(
            sparse_call, list(dense_kernel.values()), "sparse kernel-call"
        )
    for field in ("sparse_flop_seconds", "sparse_cycle_flop_seconds",
                  "sparse_traffic_seconds_per_byte"):
        updates[field] = _weighted_value(
            [_product(profile.sparse_bridge_scale, getattr(profile, field))
             for profile in profiles], weights_for("sparse_contraction")
        )

    # Baryon joins share dense component coefficients and can only be encoded
    # by one baryon_join_scale when their effective ratios agree.
    baryon_num = []
    baryon_den = []
    for dtype in _KERNEL_DTYPES:
        baryon_num.append(_weighted_value(
            [_product(profile.baryon_join_scale, profile.kernel_call_for_dtype(dtype))
             for profile in profiles], weights_for("baryon_join")
        ))
        baryon_den.append(dense_kernel[f"kernel_call_seconds{'' if dtype is None else '_' + dtype}"])
    for field in ("flop_seconds", "traffic_seconds_per_byte"):
        baryon_num.append(_weighted_value(
            [_product(profile.baryon_join_scale, getattr(profile, field))
             for profile in profiles], weights_for("baryon_join")
        ))
        baryon_den.append(updates[field])
    if operation_provenance.get("baryon_join", {}).get("weight_policy") == "base-fallback-global-weights":
        # As above, an unsupported/unobserved baryon domain keeps an explicit
        # fallback scale; it is not advertised as a fitted baryon predictor.
        updates["baryon_join_scale"] = _weighted_value(
            [profile.baryon_join_scale for profile in profiles], weights_for("baryon_join")
        )
    else:
        updates["baryon_join_scale"] = _common_factor(baryon_num, baryon_den, "baryon-join")

    updates["preparation_call_seconds"] = _average(
        profiles, weights_for("source_preparation"), "preparation_call_seconds"
    )
    optional_preparation = (
        ("preparation_kernel_call_seconds", _KERNEL_DTYPES,
         [dense_kernel[field] for field in _KERNEL_FIELDS]),
        ("preparation_flop_seconds", (None,), [updates["flop_seconds"]]),
        ("preparation_traffic_seconds_per_byte", (None,), [updates["traffic_seconds_per_byte"]]),
    )
    for field, dtypes, dense_values in optional_preparation:
        values = [_prep_coefficient(profiles, weights_for("source_preparation"), field, dtype=dtype)
                  for dtype in dtypes]
        if all(_same(value, dense) for value, dense in zip(values, dense_values, strict=True)):
            updates[field] = None
        elif all(_same(value, values[0]) for value in values[1:]):
            updates[field] = values[0]
        else:
            raise ValueError(
                f"weighted {field} predictions cannot be represented by the existing "
                "CostProfile field; optional None inheritance was resolved but produces "
                "dtype-dependent coefficients"
            )
    return updates


_MERGE_COVERAGE_CATEGORIES = (
    "source_preparation", "logical_snapshot", "dense_contraction",
    "sparse_contraction", "baryon_join",
)
_VALID_COVERAGE_STATUSES = {"fitted", "observed", "validated", "covered",
                            "legacy-complete-assumed"}
_NON_COVERAGE_STATUSES = {"base-unobserved", "base-insufficient", "unsupported"}
def _coverage_map(metadata):
    """Find fit/merge coverage without interpreting prediction_available as fit evidence."""
    if "operation_coverage" in metadata:
        coverage = metadata["operation_coverage"]
    else:
        provenance = metadata.get("prior_provenance")
        if not isinstance(provenance, dict):
            return None
        coverage = provenance.get("operation_coverage")
        if coverage is None:
            diagnostics = provenance.get("fit_diagnostics")
            if isinstance(diagnostics, dict):
                coverage = diagnostics.get("operation_coverage")
    if coverage is None:
        return None
    if not isinstance(coverage, dict):
        raise ValueError("profile operation_coverage metadata must be a mapping")
    return coverage


def _category_coverage_status(coverage, category):
    if coverage is None:
        return "legacy-complete-assumed"
    entry = coverage.get(category)
    if not isinstance(entry, dict):
        return "base-unobserved"
    status = entry.get("status", "base-unobserved")
    if not isinstance(status, str):
        raise ValueError(f"invalid operation coverage status for {category}")
    return status


def _has_valid_coverage(coverage, category, status):
    if coverage is None:
        return True  # Old complete profiles retain their compatibility behavior.
    if status in _NON_COVERAGE_STATUSES:
        return False
    if status in _VALID_COVERAGE_STATUSES:
        return True
    entry = coverage.get(category)
    return isinstance(entry, dict) and (
        entry.get("effective_coverage") is True or entry.get("calibrated") is True
    )


def _normalize_subset_weights(raw_weights, indices):
    selected = [raw_weights[index] for index in indices]
    maximum = max(selected)
    scaled = [weight / maximum for weight in selected]
    denominator = math.fsum(scaled)
    result = [0.0] * len(raw_weights)
    for index, value in zip(indices, scaled, strict=True):
        result[index] = value / denominator
    return result


def _coverage_weight_plan(metadata, raw_weights, global_weights):
    """Choose per-operation source weights and explain legacy/base fallbacks."""
    coverage_maps = [_coverage_map(item) for item in metadata]
    operation_weights = {}
    operation_provenance = {}
    for category in _MERGE_COVERAGE_CATEGORIES:
        statuses = [_category_coverage_status(coverage, category)
                    for coverage in coverage_maps]
        eligible = [
            index for index, (coverage, status) in enumerate(zip(coverage_maps, statuses, strict=True))
            if _has_valid_coverage(coverage, category, status)
        ]
        if eligible:
            operation_weights[category] = _normalize_subset_weights(raw_weights, eligible)
            if all(coverage_maps[index] is None for index in eligible):
                policy = "legacy-complete-assumed"
                merged_status = "legacy-complete-assumed"
            else:
                policy = "covered-sources-renormalized"
                has_fitted_coverage = any(
                    coverage_maps[index] is not None and statuses[index] != "legacy-complete-assumed"
                    and _has_valid_coverage(coverage_maps[index], category, statuses[index])
                    for index in eligible
                )
                merged_status = "fitted" if has_fitted_coverage else "legacy-complete-assumed"
            calibrated = any(
                coverage_maps[index] is not None
                and statuses[index] != "legacy-complete-assumed"
                and _has_valid_coverage(coverage_maps[index], category, statuses[index])
                for index in eligible
            )
        else:
            # Keep base predictions for domains with no fit evidence. This is an
            # explicit compatibility fallback, never labeled as calibrated.
            operation_weights[category] = list(global_weights)
            policy = "base-fallback-global-weights"
            if statuses and all(status == "unsupported" for status in statuses):
                merged_status = "unsupported"
            elif statuses and all(status == "base-unobserved" for status in statuses):
                merged_status = "base-unobserved"
            elif "base-insufficient" in statuses:
                merged_status = "base-insufficient"
            else:
                merged_status = "base-unobserved"
            calibrated = False

        contributors = [
            {"index": index,
             "source": metadata[index].get("source", f"input-{index}"),
             "normalized_weight": operation_weights[category][index],
             "coverage_status": statuses[index]}
            for index in range(len(metadata)) if operation_weights[category][index] > 0
        ]
        excluded = [
            {"index": index,
             "source": metadata[index].get("source", f"input-{index}"),
             "coverage_status": statuses[index]}
            for index in range(len(metadata)) if operation_weights[category][index] == 0
        ]
        operation_provenance[category] = {
            "status": merged_status,
            "effective_coverage": bool(calibrated or merged_status == "legacy-complete-assumed"),
            "calibrated": bool(calibrated),
            "weight_policy": policy,
            "contributors": contributors,
            "excluded_sources": excluded,
        }
    return operation_weights, operation_provenance


def weighted_profile_merge(profiles, *, allow_cross_hardware=False, return_provenance=False,
                           source_metadata=None, input_weight_mode=None):
    """Fuse compatible prediction models with explicit finite nonnegative weights.

    Weight normalization is overflow-safe. Each CostProfile predictor is
    expanded to the effective coefficient used by its prediction formula
    before averaging. If the merged coefficients cannot be represented by the
    existing shared scale fields, the merge is rejected. A cross-hardware
    result requires explicit opt-in and is labeled as an expected-cost model.

    When provenance has operation coverage, each operation renormalizes the
    requested input weights over fitted or explicitly valid sources. Unobserved,
    insufficient, and unsupported entries do not count as calibrated coverage.
    An operation with no covered source retains an explicitly labeled global
    base fallback. Profiles without coverage metadata keep the historical
    complete-profile merge behavior and are recorded as legacy assumptions.

    ``profiles`` accepts CostProfile objects (combined with equal weights) or
    ``(CostProfile, weight)`` pairs. File names and paths must be resolved by a
    workflow before calling this pure fusion function. Do not mix the two forms.
    """
    if not isinstance(allow_cross_hardware, bool):
        raise TypeError("allow_cross_hardware must be bool")
    if not isinstance(return_provenance, bool):
        raise TypeError("return_provenance must be bool")
    components, weight_mode = normalize_profile_weights(profiles)
    if input_weight_mode is not None:
        if input_weight_mode not in {"equal", "explicit"}:
            raise ValueError("input_weight_mode must be 'equal' or 'explicit'")
        weight_mode = input_weight_mode
    profile_models = [source for source, _, _ in components]
    if any(not isinstance(profile, CostProfile) for profile in profile_models):
        raise TypeError("weighted_profile_merge accepts CostProfile objects only; resolve file references in the contraction frontend")
    if source_metadata is None:
        metadata = [
            {"source": f"CostProfile object (version={profile.version!r}, hardware={profile.hardware_fingerprint!r})"}
            for profile in profile_models
        ]
    else:
        metadata = tuple(source_metadata)
        if len(metadata) != len(profile_models) or any(not isinstance(item, dict) for item in metadata):
            raise ValueError("source_metadata must provide one mapping for each positive-weight profile")
    raw_weights = [weight for _, weight, _ in components]
    weights = [norm for _, _, norm in components]

    fingerprints = {profile.hardware_fingerprint for profile in profile_models}
    if any(not isinstance(value, str) or not value.strip() for value in fingerprints):
        raise ValueError("positive-weight profiles require a nonempty hardware fingerprint")
    if len(fingerprints) > 1 and not allow_cross_hardware:
        raise ValueError("profiles use different hardware; pass allow_cross_hardware=True explicitly")
    operation_weights, operation_coverage = _coverage_weight_plan(
        metadata, raw_weights, weights
    )
    updates = _merge_prediction_models(
        profile_models, weights, operation_weights, operation_coverage
    )
    if len(fingerprints) == 1:
        fingerprint = next(iter(fingerprints))
        hardware_mode = "single-hardware-model"
    else:
        fingerprint = "expected-mix:" + "+".join(sorted(fingerprints))
        hardware_mode = "expected-cost-cross-hardware"
    merged = replace(
        profile_models[0], **updates, version="weighted-merge-v1",
        hardware_fingerprint=fingerprint, sample_count=0,
    )
    provenance = {
        "operation": "weighted-merge",
        "prediction_structure": _PREDICTION_STRUCTURE,
        "coefficient_units": {"time": "seconds", "work": "flops", "traffic": "bytes"},
        "weight_normalization": "positive input weights divided by their overflow-safe sum",
        "input_weight_mode": weight_mode,
        "hardware_mode": hardware_mode,
        "operation_coverage": operation_coverage,
        "inputs": [
            {
                "source": meta.get(
                    "source",
                    f"CostProfile object (version={model.version!r}, hardware={model.hardware_fingerprint!r})",
                ),
                "weight": raw_weight,
                "normalized_weight": normalized,
                "version": model.version,
                "hardware_fingerprint": model.hardware_fingerprint,
                "sample_count": model.sample_count,
                **({"prior_provenance": meta["prior_provenance"]}
                   if meta.get("prior_provenance") else {}),
            }
            for (model, meta), raw_weight, normalized
            in zip(zip(profile_models, metadata, strict=True), raw_weights, weights, strict=True)
        ],
        "coefficient_method": (
            "average effective coefficients using category-specific covered-source weights; "
            "renormalize over covered inputs, retain explicitly labeled global base fallbacks "
            "for uncovered categories, and reject nonrepresentable calibrated shared-scale ratios"
        ),
    }
    if return_provenance:
        return merged, provenance
    return merged


def _is_unmeasured_sparse_join(timing):
    features = timing.model_features or {}
    return (features.get("sparse_operation") == "bounded_coo_join"
            or features.get("join_id") == "bounded_coo_join")


def update_cost_profile(
    profile: CostProfile, report: ReplayReport, *, verified: bool, hardware_fingerprint: str, alpha: float = 0.2
) -> CostProfile:
    """Use verified action timings for future plans, never the active plan."""
    if not verified:
        raise ValueError("cost feedback requires independently verified results")
    if not hardware_fingerprint or not 0 < alpha <= 1:
        raise ValueError("hardware fingerprint and alpha are required")
    if profile.hardware_fingerprint not in ("unmeasured", hardware_fingerprint):
        raise ValueError("cost feedback hardware differs from the existing profile")
    measured = tuple(
        timing
        for timing in report.timings
        if timing.kind in ("contract", "sparse_bridge", "baryon_join")
        and timing.estimated_seconds > 0
        and math.isfinite(timing.observed_seconds)
        and timing.observed_seconds >= 0
        and not _is_unmeasured_sparse_join(timing)
    )
    if not measured:
        if any(_is_unmeasured_sparse_join(timing) for timing in report.timings):
            raise ValueError(
                "bounded_coo_join timings are explicitly unmeasured and cannot calibrate legacy sparse costs"
            )
    updates = {}
    snapshots = tuple(
        phase
        for timing in report.timings
        if timing.kind == "load"
        for phase in timing.phases
        if phase.name == "snapshot"
        and phase.bytes is not None
        and phase.bytes > 0
        and math.isfinite(phase.observed_seconds)
        and phase.observed_seconds >= 0
    )
    if not measured and not snapshots:
        raise ValueError("cost feedback has no measured snapshot or contraction phases")
    snapshot_fit = _fit_nonnegative(
        [(1.0, phase.bytes) for phase in snapshots],
        [phase.observed_seconds for phase in snapshots],
    )
    if snapshot_fit is not None:
        updates["snapshot_call_seconds"] = (1 - alpha) * profile.snapshot_call_seconds + alpha * float(snapshot_fit[0])
        updates["snapshot_seconds_per_byte"] = (1 - alpha) * profile.snapshot_seconds_per_byte + alpha * float(
            snapshot_fit[1]
        )
    dense = tuple(
        timing
        for timing in measured
        if timing.kind == "contract" and timing.model_features and timing.model_features.get("kernel_calls") == 1
    )
    dense_groups = {}
    for timing in dense:
        features = timing.model_features
        signature = (features.get("dtype"), features["flops"], features["traffic_bytes"])
        dense_groups.setdefault(signature, []).append(timing)
    dense = tuple(
        replace(group[0], observed_seconds=median(item.observed_seconds for item in group))
        for group in dense_groups.values()
    )
    hold_contract_scale = False
    supported_dtypes = ("float32", "float64", "complex64", "complex128")
    typed = tuple(item for item in dense if item.model_features.get("dtype") in supported_dtypes)
    if typed:
        present = tuple(
            dtype for dtype in supported_dtypes if any(item.model_features["dtype"] == dtype for item in typed)
        )
        flops_observed = {item.model_features["flops"] for item in typed}
        traffic_observed = {item.model_features["traffic_bytes"] for item in typed}
        fit_flops = len(flops_observed) > 1 and max(flops_observed) >= 1_000
        fit_traffic = len(traffic_observed) > 1 and max(traffic_observed) >= 1_024
        variable_fields = tuple(
            field
            for enabled, field in (
                (fit_flops, "flop_seconds"),
                (fit_traffic, "traffic_seconds_per_byte"),
            )
            if enabled
        )
        low_work = tuple(item for item in typed if item.model_features["flops"] <= 1_000)
        if low_work:
            for dtype in present:
                samples = [
                    max(
                        0.0,
                        item.observed_seconds / profile.contract_scale
                        - item.model_features["flops"] * profile.flop_seconds
                        - item.model_features["traffic_bytes"] * profile.traffic_seconds_per_byte,
                    )
                    for item in low_work
                    if item.model_features["dtype"] == dtype
                ]
                if samples:
                    field = f"kernel_call_seconds_{dtype}"
                    updates[field] = (1 - alpha) * getattr(profile, field) + alpha * median(samples)
        kernel_fit = (
            _fit_nonnegative(
                [
                    tuple(float(item.model_features["dtype"] == dtype) for dtype in present)
                    + tuple(
                        item.model_features["flops"]
                        if field == "flop_seconds"
                        else item.model_features["traffic_bytes"]
                        for field in variable_fields
                    )
                    for item in typed
                ],
                [
                    item.observed_seconds / profile.contract_scale
                    - (0 if fit_flops else item.model_features["flops"] * profile.flop_seconds)
                    - (
                        0
                        if fit_traffic
                        else item.model_features["traffic_bytes"] * profile.traffic_seconds_per_byte
                    )
                    for item in typed
                ],
            )
            if profile.contract_scale and not low_work
            else None
        )
        if kernel_fit is not None:
            for dtype, value in zip(present, kernel_fit[: len(present)], strict=True):
                field = f"kernel_call_seconds_{dtype}"
                updates[field] = (1 - alpha) * getattr(profile, field) + alpha * float(value)
            for field, value in zip(variable_fields, kernel_fit[len(present) :], strict=True):
                updates[field] = (1 - alpha) * getattr(profile, field) + alpha * float(value)
            for dtype in present:
                field = f"kernel_call_seconds_{dtype}"
                if updates[field] > 0:
                    continue
                dtype_items = [item for item in typed if item.model_features["dtype"] == dtype]
                least_flops = min(item.model_features["flops"] for item in dtype_items)
                floor = median(
                    max(
                        0.0,
                        item.observed_seconds / profile.contract_scale
                        - item.model_features["flops"] * updates.get("flop_seconds", profile.flop_seconds)
                        - item.model_features["traffic_bytes"]
                        * updates.get("traffic_seconds_per_byte", profile.traffic_seconds_per_byte),
                    )
                    for item in dtype_items
                    if item.model_features["flops"] == least_flops
                )
                if floor > 0:
                    updates[field] = (1 - alpha) * getattr(profile, field) + alpha * floor
        elif profile.contract_scale and not low_work:
            for dtype in present:
                samples = [
                    max(
                        0.0,
                        item.observed_seconds / profile.contract_scale
                        - item.model_features["flops"] * profile.flop_seconds
                        - item.model_features["traffic_bytes"] * profile.traffic_seconds_per_byte,
                    )
                    for item in typed
                    if item.model_features["dtype"] == dtype
                ]
                if samples:
                    field = f"kernel_call_seconds_{dtype}"
                    updates[field] = (1 - alpha) * getattr(profile, field) + alpha * median(samples)
        if low_work and variable_fields:
            work_fit = _fit_nonnegative(
                [
                    tuple(
                        item.model_features["flops"]
                        if field == "flop_seconds"
                        else item.model_features["traffic_bytes"]
                        for field in variable_fields
                    )
                    for item in typed
                ],
                [
                    max(
                        0.0,
                        item.observed_seconds / profile.contract_scale
                        - updates.get(
                            f"kernel_call_seconds_{item.model_features['dtype']}",
                            profile.kernel_call_for_dtype(item.model_features["dtype"]),
                        )
                        - (0 if fit_flops else item.model_features["flops"] * profile.flop_seconds)
                        - (
                            0
                            if fit_traffic
                            else item.model_features["traffic_bytes"] * profile.traffic_seconds_per_byte
                        ),
                    )
                    for item in typed
                ],
            )
            if work_fit is not None:
                for field, value in zip(variable_fields, work_fit, strict=True):
                    updates[field] = (1 - alpha) * getattr(profile, field) + alpha * float(value)
            else:
                hold_contract_scale = True
        elif low_work:
            hold_contract_scale = True
    else:
        kernel_fit = _fit_nonnegative(
            [(1.0, item.model_features["flops"], item.model_features["traffic_bytes"]) for item in dense],
            [item.observed_seconds for item in dense],
        )
        if kernel_fit is not None:
            for field, value in zip(
                ("kernel_call_seconds", "flop_seconds", "traffic_seconds_per_byte"), kernel_fit, strict=True
            ):
                updates[field] = (1 - alpha) * getattr(profile, field) + alpha * float(value)
    dense_fitted = replace(profile, **updates)
    sparse_groups = {}
    for timing in measured:
        features = timing.model_features
        if timing.kind != "sparse_bridge" or not features or features.get("kernel_calls", 0) < 1:
            continue
        signature = (
            max(1, len(timing.operands) - 2),
            features["dtype"],
            features["kernel_calls"],
            features["flops"],
            features["traffic_bytes"],
        )
        sparse_groups.setdefault(signature, []).append(timing.observed_seconds)
    if sparse_groups and profile.sparse_bridge_scale:
        single_groups = [
            (signature, observations)
            for signature, observations in sparse_groups.items()
            if signature[0] == 1
        ]
        low_groups = [item for item in single_groups if item[0][3] <= 1_000]
        call_estimates = [
            max(
                0.0,
                median(observations) / profile.sparse_bridge_scale
                - flops * profile.sparse_flop_seconds
                - traffic * dense_fitted.sparse_traffic_seconds_per_byte,
            )
            / (calls * dense_fitted.kernel_call_for_dtype(dtype))
            for (_records, dtype, calls, flops, traffic), observations in low_groups
            if dense_fitted.kernel_call_for_dtype(dtype) > 0
        ]
        if call_estimates:
            updates["sparse_kernel_call_scale"] = (
                (1 - alpha) * profile.sparse_kernel_call_scale + alpha * median(call_estimates)
            )
        call_scale = updates.get("sparse_kernel_call_scale", profile.sparse_kernel_call_scale)
        for records, field in ((1, "sparse_flop_seconds"), (2, "sparse_cycle_flop_seconds")):
            ratios = [
                max(
                    0.0,
                    median(observations) / profile.sparse_bridge_scale
                    - calls * dense_fitted.kernel_call_for_dtype(dtype) * call_scale
                    - traffic * dense_fitted.sparse_traffic_seconds_per_byte,
                )
                / flops
                for (group_records, dtype, calls, flops, traffic), observations in sparse_groups.items()
                if group_records == records and flops > 0
            ]
            if ratios:
                estimate = min(ratios) if records == 1 else median(ratios)
                updates[field] = (1 - alpha) * getattr(profile, field) + alpha * estimate
    fitted = replace(profile, **updates)

    scales = {}
    for kind, field in (
        ("contract", "contract_scale"),
        ("sparse_bridge", "sparse_bridge_scale"),
        ("baryon_join", "baryon_join_scale"),
    ):
        if kind == "contract" and hold_contract_scale:
            scales[field] = profile.contract_scale
            continue
        ratios = []
        observed_total = 0.0
        baseline_total = 0.0
        for timing in measured:
            if timing.kind != kind:
                continue
            features = timing.model_features
            if features and kind == "sparse_bridge":
                records = max(1, len(timing.operands) - 2)
                baseline = (
                    features["kernel_calls"]
                    * fitted.kernel_call_for_dtype(features["dtype"])
                    * fitted.sparse_kernel_call_scale
                    + features["flops"]
                    * (fitted.sparse_cycle_flop_seconds if records == 2 else fitted.sparse_flop_seconds)
                    + features["traffic_bytes"] * fitted.sparse_traffic_seconds_per_byte
                )
            elif features and kind != "load":
                baseline = (
                    features.get("kernel_calls", 1) * fitted.kernel_call_for_dtype(features.get("dtype"))
                    + features["flops"] * fitted.flop_seconds
                    + features["traffic_bytes"] * fitted.traffic_seconds_per_byte
                )
            else:
                baseline = timing.estimated_seconds / getattr(profile, field) if getattr(profile, field) else 0
            if baseline > 0:
                ratios.append(timing.observed_seconds / baseline)
                observed_total += timing.observed_seconds
                baseline_total += baseline
        current = getattr(profile, field)
        measured_scale = median(ratios) if ratios else current
        scales[field] = (1 - alpha) * current + alpha * measured_scale if ratios else current
    return replace(
        fitted,
        **scales,
        hardware_fingerprint=hardware_fingerprint,
        sample_count=profile.sample_count + len(measured) + len(snapshots),
        version=f"calibrated-{profile.sample_count + len(measured) + len(snapshots)}",
    )



__all__ = [
    "ProfileFitReport", "_fit_nonnegative", "calibrate_cost_profile",
    "evaluate_cost_profile", "execution_cost_samples", "fit_cost_profile", "fit_cost_profile_sources",
    "normalize_profile_weights", "update_cost_profile",
    "weighted_profile_merge",
]
