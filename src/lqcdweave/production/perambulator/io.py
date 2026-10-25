"""Six-representative PR IO over injected, existing DatasetReader/Writer objects.

No format selection, publication, file discovery or implicit solves live here.
Each first-axis row is one explicit physical (sink,source) time pair, allowing
noncanonical products to be saved at their actual reversed coordinates.
"""

from numbers import Integral
import numpy as np

from lqcdweave.contracts.dataset import DatasetSpec, canonical_json
from lqcdweave.contracts.propagator import PropagatorEndpoint, PropagatorBlock
from lqcdweave.contracts.mode_partition import ModePartition
from .products import (SAVED_PRODUCTS, normalize_product,
                       product_shape, reverse_product)
from .symmetry import reverse_propagator


def _endpoint(endpoint):
    if not isinstance(endpoint, PropagatorEndpoint) or endpoint.data_id is None:
        raise ValueError('storage requires an actual PropagatorEndpoint data_id')
    if endpoint.kind == 'D':
        metadata = endpoint.metadata
        required = {'vector_labels', 'mode_groups', 'parent_basis_id',
                    'realization_id', 'noise_pool_id', 'normalization'}
        if not required <= metadata.keys():
            raise ValueError('D storage requires explicit groups, labels and random provenance')
        labels = metadata['vector_labels']
        if (not isinstance(labels, list) or len(labels) != len(endpoint.indices) or
                any(not isinstance(label, list) or len(label) != 2 or
                    any(type(i) is not int or i < 0 for i in label) for label in labels) or
                len({tuple(label) for label in labels}) != len(labels)):
            raise ValueError('D vector_labels must bind each selected vector to a unique block/vector pair')
        if metadata['normalization'] != 'raw':
            raise ValueError('PR storage preserves raw dilution amplitudes')
    projection = endpoint.metadata.get('high_mode_projection')
    if projection is not None:
        if endpoint.kind != 'P':
            raise ValueError('high projection belongs only to a point endpoint')
        ModePartition.from_document(projection)
    return endpoint


def _physical_time(time, lt):
    if isinstance(time, bool) or not isinstance(time, Integral) or not 0 <= time < lt:
        raise ValueError('time pairs require physical integer times in [0,Lt)')
    return int(time)


def _gamma_matrix(gamma5):
    gamma = np.asarray(gamma5, dtype=np.complex128)
    if (gamma.shape != (4, 4) or not np.isfinite(gamma).all() or
            not np.allclose(gamma, gamma.conj().T, rtol=0, atol=1e-12) or
            not np.allclose(gamma @ gamma, np.eye(4), rtol=0, atol=1e-12)):
        raise ValueError('gamma5 must be a finite Hermitian 4x4 involution')
    return gamma


def propagator_pair_spec(product, *, configuration, sink, source, time_pairs,
                         gauge_id, dirac, gamma5, spin_basis,
                         gamma5_hermitian, dtype='complex128'):
    """Define one canonical PR dataset, with explicit physical pair coordinates.

    The action's Gamma5 capability is supplied by the caller, not inferred from
    its label. Actual D identity and all estimator provenance are retained.
    The resulting spec uses the existing generic whole-array NPY/HDF5 writer.
    """
    product = normalize_product(product)
    if product not in SAVED_PRODUCTS:
        raise ValueError('storage products must be ' + ', '.join(SAVED_PRODUCTS))
    if gamma5_hermitian is not True:
        raise ValueError('six-representative storage requires declared gamma5 hermiticity')
    sink, source = _endpoint(sink), _endpoint(source)
    if (sink.kind, source.kind) != (product[0], product[2]):
        raise ValueError('storage product differs from its actual endpoints')
    if not isinstance(gauge_id, str) or not gauge_id or not isinstance(dirac, dict) or not dirac:
        raise ValueError('actual gauge_id and explicit Dirac definition are required')
    if not isinstance(spin_basis, str) or not spin_basis:
        raise ValueError('an explicit spin_basis is required')
    if dtype not in ('complex64', 'complex128'):
        raise ValueError('propagator storage requires complex64 or complex128')
    gamma = _gamma_matrix(gamma5)
    lt = configuration['lattice'][3]
    pairs = []
    for pair in time_pairs:
        if len(pair) != 2:
            raise ValueError('time_pairs must contain (sink_time,source_time) pairs')
        pairs.append([_physical_time(t, lt) for t in pair])
    axes = [dict(name='time_pair', coordinates=pairs),
            dict(name='sink_spin', coordinates=list(range(4))),
            dict(name='source_spin', coordinates=list(range(4)))]
    bases = {}
    for side, endpoint in (('sink', sink), ('source', source)):
        name = side + ('_point' if endpoint.kind == 'P' else
                       '_mode' if endpoint.kind == 'V' else '_dilution_vector')
        axes.append(dict(name=name, coordinates=list(endpoint.indices)))
        bases[name] = dict(data_id=endpoint.data_id, indices=list(endpoint.indices))
        if endpoint.kind == 'P':
            axes.append(dict(name=side+'_color', coordinates=list(range(3))))
    return DatasetSpec(dict(schema=1, kind='perambulator_pairs',
        configuration=configuration, axes=axes, dtype=dtype, bases=bases,
        definition=dict(encoding='pr-six-pairs-v1', product=product,
                        saved_products=list(SAVED_PRODUCTS), gauge_id=gauge_id,
                        dirac=dirac, sink_endpoint=sink.document, source_endpoint=source.document),
        conventions=dict(normalization='unscaled', time_coordinates='physical_pairs',
                         spin_basis=spin_basis, gamma5_hermitian=True,
                         gamma5=dict(real=gamma.real.tolist(), imag=gamma.imag.tolist())),
        parents=list(dict.fromkeys((gauge_id, sink.data_id, source.data_id,
            *(endpoint.metadata['high_mode_projection']['parent_basis_id']
              for endpoint in (sink, source) if 'high_mode_projection' in endpoint.metadata))))))


class _PairLayout:
    """Validated, configuration-local metadata; no borrowed numeric input."""

    def __init__(self, spec):
        document = spec.document
        if (document['kind'] != 'perambulator_pairs' or
                document['definition'].get('encoding') != 'pr-six-pairs-v1'):
            raise ValueError('expected a pr-six-pairs-v1 propagator dataset')
        definition, conventions = document['definition'], document['conventions']
        self.product = definition['product']
        self.sink = PropagatorEndpoint.from_document(definition['sink_endpoint'])
        self.source = PropagatorEndpoint.from_document(definition['source_endpoint'])
        encoded = conventions['gamma5']
        self.gamma5 = np.asarray(encoded['real']) + 1j*np.asarray(encoded['imag'])
        self.lt = document['configuration']['lattice'][3]
        self.pairs = tuple(tuple(pair) for pair in spec.coordinates('time_pair'))
        expected = propagator_pair_spec(self.product, configuration=document['configuration'],
            sink=self.sink, source=self.source, time_pairs=self.pairs,
            gauge_id=definition['gauge_id'], dirac=definition['dirac'],
            gamma5=self.gamma5, spin_basis=conventions['spin_basis'],
            gamma5_hermitian=conventions['gamma5_hermitian'], dtype=document['dtype'])
        if expected.document != document:
            raise ValueError('propagator dataset axes or convention metadata differ')
        self.positions = {pair: i for i, pair in enumerate(self.pairs)}
        self.context = canonical_json(dict(configuration=document['configuration'],
            gauge_id=definition['gauge_id'], dirac=definition['dirac'], conventions=conventions))


class PropagatorWriter:
    """Normalize one raw physical-pair block into an injected canonical writer.

    The caller owns writer lifetime, resume and publication. This facade never
    changes the existing writer's dtype, scientific spec or covered rows.
    """

    def __init__(self, writer, *, array_module=np):
        self.writer, self.array_module = writer, array_module
        self._layout = _PairLayout(writer.spec)

    def write(self, block):
        layout = self._layout
        product = normalize_product(block.product)
        pair = (_physical_time(block.sink_time, layout.lt),
                _physical_time(block.source_time, layout.lt))
        # Validate before converting or modifying the underlying writer.
        if product not in SAVED_PRODUCTS:
            product = reverse_product(product)
            sink, source, pair = block.source, block.sink, pair[::-1]
        else:
            sink, source = block.sink, block.source
        if product != layout.product or sink != layout.sink or source != layout.source:
            raise ValueError('block differs from the canonical writer endpoint records')
        if pair not in layout.positions:
            raise KeyError('canonical physical time pair is outside the declared coverage')
        if tuple(block.values.shape) != product_shape(block.product,
                len(block.sink.indices), len(block.source.indices)):
            raise ValueError('block shape differs from its endpoint records')
        if normalize_product(block.product) not in SAVED_PRODUCTS:
            block = reverse_propagator(block, layout.gamma5, gamma5_hermitian=True,
                                       array_module=self.array_module)
        values = block.values
        if self.array_module is not np:
            values = self.array_module.asnumpy(values)
        # No implicit precision change or estimator weighting at this boundary.
        self.writer.write(layout.positions[pair], values)


class PropagatorReader:
    """Nine product views over injected six-representative readers.

    Multiple datasets of the same product may bind distinct actual endpoints.
    They must share one configuration/action/Gamma5 context. Exact requests
    resolve actual reverse time pairs; missing data never starts a solver.
    Reader lifetime remains with the caller.
    """

    def __init__(self, readers, *, array_module=np):
        self.array_module = array_module
        self._readers = {}
        context = None
        for reader in readers:
            layout = _PairLayout(reader.spec)
            if context is None:
                context = layout.context
            elif context != layout.context:
                raise ValueError('reader configuration, action or Gamma5 conventions differ')
            key = (layout.product, layout.sink, layout.source)
            entry = self._readers.setdefault(key, [])
            covered = {pair for _, prior in entry for pair in prior.pairs}
            if covered.intersection(layout.pairs):
                raise ValueError('ambiguous duplicate physical-pair coverage')
            entry.append((reader, layout))
        if context is None:
            raise ValueError('at least one canonical propagator reader is required')
        self.lt = layout.lt
        self._coverage = {
            key: tuple(pair for _, item in entries for pair in item.pairs)
            for key, entries in self._readers.items()
        }

    def coverage(self, product, *, sink, source):
        """Return available physical pairs without reading numerical data.

        Reverse products exchange actual endpoints and physical times. Coverage
        belongs to these exact endpoint records, not just their vector counts.
        VSV/DSD/PSP also cover opposite pairs through declared Gamma5
        hermiticity, with the actual source and sink endpoints exchanged.
        """
        product = normalize_product(product)
        if (product[0], product[2]) != (sink.kind, source.kind):
            raise ValueError('requested product differs from its actual endpoints')
        reverse = product not in SAVED_PRODUCTS
        key = ((reverse_product(product), source, sink) if reverse else
               (product, sink, source))
        pairs = self._coverage.get(key, ())
        if reverse:
            return tuple(pair[::-1] for pair in pairs)
        if product[0] == product[2]:
            opposite = self._coverage.get((product, source, sink), ())
            return tuple(dict.fromkeys((*pairs, *(pair[::-1] for pair in opposite))))
        return pairs

    def missing(self, product, time_pairs, *, sink, source):
        """Return absent requested pairs in request order; never solve or write.

        Use this before creating an additional disjoint dataset. Published
        datasets remain unchanged; partial writer recovery is caller-owned.
        """
        covered = set(self.coverage(product, sink=sink, source=source))
        missing, seen = [], set()
        for pair in time_pairs:
            if len(pair) != 2:
                raise ValueError('time_pairs must contain (sink_time,source_time) pairs')
            pair = tuple(_physical_time(t, self.lt) for t in pair)
            if pair in seen:
                raise ValueError('duplicate requested physical time pair')
            seen.add(pair)
            if pair not in covered:
                missing.append(pair)
        return tuple(missing)

    def read(self, product, sink_time, source_time, *, sink, source):
        """Return a recorded block; direct reads may borrow, reverse reads own."""
        product = normalize_product(product)
        if (product[0], product[2]) != (sink.kind, source.kind):
            raise ValueError('requested product differs from its actual endpoints')
        reverse = product not in SAVED_PRODUCTS
        parent_product = reverse_product(product) if reverse else product
        parent_sink, parent_source = (source, sink) if reverse else (sink, source)
        orientations = [(parent_product, parent_sink, parent_source, reverse)]
        if not reverse and product[0] == product[2]:
            orientations.append((product, source, sink, True))
        pair = (_physical_time(sink_time, self.lt), _physical_time(source_time, self.lt))
        for parent_product, parent_sink, parent_source, reverse in orientations:
            entries = self._readers.get((parent_product, parent_sink, parent_source), ())
            for reader, layout in entries:
                parent_pair = pair[::-1] if reverse else pair
                if parent_pair not in layout.positions:
                    continue
                values = self.array_module.asarray(reader.read(time_pair=list(parent_pair)))
                block = PropagatorBlock(parent_product, *parent_pair, parent_sink, parent_source,
                                        values, parent_id=reader.data_id)
                return (reverse_propagator(block, layout.gamma5, gamma5_hermitian=True,
                                          array_module=self.array_module) if reverse else block)
        raise KeyError('missing actual endpoint or reverse physical-pair coverage')


def source_time_blocks(product, values, source_time, sink_times, *, sink, source):
    """Yield borrowed fixed-pair views from a dense generator source-time row."""
    product = normalize_product(product)
    if (sink.kind, source.kind) != (product[0], product[2]):
        raise ValueError('product differs from actual source-time row endpoints')
    if tuple(values.shape) != (len(sink_times),) + product_shape(
            product, len(sink.indices), len(source.indices)):
        raise ValueError('row shape differs from its ordered sink times/endpoints')
    for slot, time in enumerate(sink_times):
        yield PropagatorBlock(product, time, source_time, sink, source, values[slot])
