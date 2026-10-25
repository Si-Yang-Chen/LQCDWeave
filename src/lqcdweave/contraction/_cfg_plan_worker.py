"""Private local-process transport for pure P descriptors, never archive input.

Only the parent-created worker and its private pipes use pickle. Persisted or
user-supplied plans always go through A's whitelisted archive codec.
"""
from contextlib import redirect_stdout
from contextvars import ContextVar
import copyreg
from dataclasses import fields
import io
import pickle
import struct
import sys
from time import perf_counter
import traceback
from types import MappingProxyType

from lqcdweave.contracts.correlator.api import SourcePreparation, TensorSpec


_reference_resolver = ContextVar('cfg_window_reference_resolver', default=None)


def _restore_reference(reference):
    resolver = _reference_resolver.get()
    if resolver is None:
        raise pickle.UnpicklingError('private cfg descriptor reference outside receive scope')
    return resolver(reference)


def _restore_mapping(value):
    return MappingProxyType(value)


def _reduce_mapping(value):
    return _restore_mapping, (dict(value),)


def _restore_state(values):
    from lqcdweave.contraction.planning.api import ContractionState
    return ContractionState(**values)


def _reduce_state(value):
    # Only the public frozen fields cross the process boundary. Reconstruct
    # the immutable catalog through P; exclude search-local cached hashes.
    values = {field.name: getattr(value, field.name) for field in fields(value)}
    values['catalog'] = dict(value.catalog)
    return _restore_state, (values,)


def _same_spec(current, previous):
    # TensorSpec's numerical equality deliberately excludes description.
    # Transport must also preserve its complete scientific metadata.
    return (current is previous or
            (type(current) is TensorSpec and type(previous) is TensorSpec
             and current == previous and current.description == previous.description))


def _same_template(current, previous):
    if current is previous:
        return True
    if current != previous:
        return False
    for left, right in ((current.plan.state.catalog, previous.plan.state.catalog),
                        (current.preparation_specs or {}, previous.preparation_specs or {})):
        if any(not _same_spec(spec, right[key]) for key, spec in left.items()):
            return False
    return True


class _TemplatePickler(pickle.Pickler):
    def __init__(self, buffer, references):
        super().__init__(buffer, protocol=pickle.HIGHEST_PROTOCOL)
        self.references = references

    def reducer_override(self, value):
        reference = self.references.get(id(value))
        # A reducer participates in pickle's ordinary object memo. A persistent
        # ID instead invokes the resolver for every occurrence of the object.
        return ((_restore_reference, (reference,)) if reference is not None
                else NotImplemented)


class _TemplateUnpickler(pickle.Unpickler):
    def __init__(self, buffer, templates, preparations=None):
        super().__init__(buffer)
        self.templates = templates
        self.preparations = preparations if preparations is not None else {}

    def load(self):
        token = _reference_resolver.set(self.persistent_load)
        try:
            return super().load()
        finally:
            _reference_resolver.reset(token)

    def persistent_load(self, reference):
        if isinstance(reference, tuple) and len(reference) == 2:
            kind, key = reference
            if (kind == 'cfg-template-v1' and type(key) is int
                    and 0 <= key < len(self.templates)):
                return self.templates[key]
            if kind == 'cfg-preparation-v1' and type(key) is str:
                value = self.preparations.get(key)
                if type(value) is SourcePreparation:
                    return value
            if (kind == 'cfg-spec-v1' and isinstance(key, tuple) and len(key) == 2
                    and type(key[0]) is int and 0 <= key[0] < len(self.templates)
                    and type(key[1]) is str):
                specs = self.templates[key[0]].preparation_specs
                value = specs.get(key[1]) if specs is not None else None
                if type(value) is TensorSpec:
                    return value
        raise pickle.UnpicklingError('invalid private cfg descriptor reference')


def pack(value, *, template_refs=None):
    from lqcdweave.contraction.planning.api import ContractionState
    buffer = io.BytesIO()
    writer = (_TemplatePickler(buffer, template_refs) if template_refs else
              pickle.Pickler(buffer, protocol=pickle.HIGHEST_PROTOCOL))
    writer.dispatch_table = dict(copyreg.dispatch_table)
    writer.dispatch_table[MappingProxyType] = _reduce_mapping
    writer.dispatch_table[ContractionState] = _reduce_state
    writer.dump(value)
    return buffer.getvalue()


def send(stream, value, *, template_refs=None):
    payload = pack(value, template_refs=template_refs)
    stream.write(struct.pack('!Q', len(payload)))
    stream.write(payload)
    stream.flush()


def _receive_payload(stream):
    header = stream.read(8)
    if len(header) != 8:
        raise EOFError('planning worker closed its response pipe')
    size, = struct.unpack('!Q', header)
    payload = stream.read(size)
    if len(payload) != size:
        raise EOFError('planning worker returned a partial response')
    return payload


def receive(stream):
    return pickle.loads(_receive_payload(stream))


class WindowSender:
    """Reference value-equal immutable descriptors from the last response."""
    def __init__(self):
        self.templates = ()
        self.preparations = {}

    def send(self, stream, value):
        kind, window = value
        references = {}
        current = window.templates if kind == 'window' else ()
        preparations = window.preparations if kind == 'window' else {}
        for template in current:
            for index, previous in enumerate(self.templates):
                if _same_template(template, previous):
                    references[id(template)] = ('cfg-template-v1', index)
                    # The first full template memoizes these specs implicitly.
                    # A later template reference does not; reference
                    # its already restored specs instead of resending metadata.
                    for key, spec in (template.preparation_specs or {}).items():
                        if type(key) is str and type(spec) is TensorSpec:
                            references.setdefault(id(spec), ('cfg-spec-v1', (index, key)))
                    break
        for key, preparation in preparations.items():
            previous = self.preparations.get(key)
            if (type(key) is str and type(preparation) is SourcePreparation
                    and (preparation is previous or preparation == previous)):
                references[id(preparation)] = ('cfg-preparation-v1', key)
        send(stream, value, template_refs=references)
        self.templates = current
        self.preparations = preparations


class WindowReceiver:
    def __init__(self):
        self.templates = ()
        self.preparations = {}
        self.received_windows = 0
        self.received_bytes = 0
        self.receive_seconds = 0.0
        self.restore_seconds = 0.0

    def receive(self, stream):
        start = perf_counter()
        payload = _receive_payload(stream)
        received = perf_counter()
        value = _TemplateUnpickler(io.BytesIO(payload), self.templates,
                                  self.preparations).load()
        self.receive_seconds += received-start
        self.restore_seconds += perf_counter()-received
        self.received_bytes += len(payload)
        self.received_windows += value[0] == 'window'
        self.templates = value[1].templates if value[0] == 'window' else ()
        self.preparations = value[1].preparations if value[0] == 'window' else {}
        return value

    def clear(self):
        self.templates = ()
        self.preparations = {}


def main():
    from lqcdweave.contraction.planning.api import iter_cfg_plan_windows
    incoming, outgoing = sys.stdin.buffer, sys.stdout.buffer
    sender = WindowSender()
    # Normal P diagnostics may not corrupt the private binary response pipe.
    try:
        with redirect_stdout(sys.stderr):
            args, options = receive(incoming)
            windows = iter_cfg_plan_windows(*args, **options)
            while incoming.read(1) == b'N':
                try:
                    window = next(windows)
                except StopIteration:
                    sender.send(outgoing, ('end', None))
                    return
                sender.send(outgoing, ('window', window))
                del window
    except BaseException as error:
        try:
            send(outgoing, ('error', f'{type(error).__name__}: {error}\n{traceback.format_exc()}'))
        except (BrokenPipeError, OSError):
            pass


if __name__ == '__main__':
    # Keep reducer names importable in the parent instead of serializing
    # references to its unrelated frontend's __main__ module.
    from lqcdweave.contraction._cfg_plan_worker import main as run_worker
    run_worker()
