"""Price independent dense tails without mutating the expression workspace.

Only compact calculation identities are retained: each repeated block is one
deduplicated tuple of calculation IDs, and each tail's ordered block sequence
shares one retained base except at the positions where it differs. Integer
operations and per-Term Actions are expanded for the finally selected route,
not for each candidate tail. Ambiguous overlapping families keep the stepwise
reference.
"""
from collections import OrderedDict
from dataclasses import dataclass, replace
import math

from lqcdweave.contracts.correlator.api import Factor, PendingTerm
from ..canonical_operations import describe


@dataclass(frozen=True, slots=True)
class SharedChunks:
    """An ordered block sequence sharing one retained base, except overrides.

    Successive retained tails repeat almost the same block sequence, so a tail
    keeps only the positions where its sequence actually differs. Overrides are
    ascending and replace base positions one by one; length, indexing and
    iteration are those of the plain sequence, so expansion, bindings and the
    final cost are unchanged. It is a dataclass so the retained-size accounting
    follows the shared base and the overrides instead of stopping at a
    representation the walk cannot see.
    """
    base: tuple
    overrides: tuple

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        if index < 0:
            index += len(self.base)
        for position, chunk in self.overrides:
            if position == index:
                return chunk
        return self.base[index]

    def __iter__(self):
        overrides = self.overrides
        if not overrides:
            return iter(self.base)
        return self._iterate(overrides)

    def _iterate(self, overrides):
        pending, position = list(overrides), 0
        pending.reverse()
        for chunk in self.base:
            if pending and pending[-1][0] == position:
                chunk = pending.pop()[1]
            yield chunk
            position += 1


@dataclass(frozen=True, slots=True)
class CalculationChunks:
    chunks: object
    size: int

    def __len__(self):
        return self.size

    def __iter__(self):
        for chunk in self.chunks:
            yield from chunk


@dataclass(frozen=True, slots=True)
class TerminalCompletion:
    seconds: float
    calculations: CalculationChunks


class TerminalCompletions:
    # A tail keeps a shared base only while its patch stays this many times
    # smaller than the plain sequence it replaces. The base is always one
    # tail's own retained sequence, so a shared tail costs only its patch and
    # the representation can never exceed the unshared one by a whole sequence.
    sequence_divisor = 8

    def __init__(self):
        self.families, self.tails, self.chunks = OrderedDict(), OrderedDict(), OrderedDict()
        self.requests = self.hits = self.family_hits = self.family_builds = 0
        self.builds = self.fallbacks = self.logical_steps = self.expansions = 0
        self.sequence = None
        self.shared_sequences = self.reused_sequences = 0
        # The scalar index that prices the complete independent tails of this
        # session. It owns the incremental fold and block bookkeeping the scalar
        # branch reuses, and reports them through statistics(). The reference
        # outlives clear() so a released session still reports what it reused.
        self.scalar = None

    @staticmethod
    def _remember(cache, key, value, limit):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > limit:
            cache.popitem(last=False)
        return value

    def _share(self, chunks, debug=False, overrides=None):
        """Share one tail's ordered block sequence with the retained base.

        Only positions where the sequence differs from the base are retained,
        and only while that patch stays small against the sequence it replaces.
        A producer that already tracks those positions hands them over instead
        of paying for the comparison again; debug re-walks the sequence and
        rejects a patch that does not describe it.
        """
        base = self.sequence
        if base is None or len(base) != len(chunks):
            self.sequence = chunks
            return chunks
        if overrides is None:
            overrides = [(position, chunk) for position, (old, chunk)
                         in enumerate(zip(base, chunks)) if old is not chunk]
        elif debug:
            walked = [(position, chunk) for position, (old, chunk)
                      in enumerate(zip(base, chunks)) if old is not chunk]
            if walked != list(overrides):
                raise RuntimeError("supplied terminal block overrides do not describe the sequence")
        if not overrides:
            self.reused_sequences += 1
            return base
        if len(overrides) * self.sequence_divisor > len(chunks):
            self.sequence = chunks
            return chunks
        shared = SharedChunks(base, tuple(overrides))
        # Debug re-opens the representation check the release path skips: the
        # shared sequence must still replay the exact retained order.
        if debug and tuple(shared) != chunks:
            raise RuntimeError("shared terminal block sequence changed order or identity")
        self.shared_sequences += 1
        return shared

    def _shared(self, value, debug=False, overrides=None):
        """Share one already priced tail's block sequence with the retained base."""
        recipe = value.calculations
        chunks = self._share(recipe.chunks, debug, overrides)
        if chunks is recipe.chunks:
            return value
        return TerminalCompletion(value.seconds, CalculationChunks(chunks, recipe.size))

    def _family_options(self, space, family, checkpoint):
        signatures, representatives = [], {}
        for identity in sorted(space.terminal_members[family]):
            if not space.registry.calculations[identity].recipe.equation.output:
                # A classified scalar-output family cannot need an output
                # permutation. Its consumer count does not change its cost.
                signatures.append((identity, ()))
                representatives[identity] = {}
                continue
            permutations = {}
            for scene, occurrence in space.groups[identity].items():
                checkpoint()
                output = space.registry.patterns[space.terms[scene].pattern].output
                permutation = tuple(occurrence.output.index(a) for a in output)
                if permutation != tuple(range(len(output))):
                    permutations.setdefault(permutation, (scene, occurrence, output))
            signatures.append((identity, tuple(sorted(permutations))))
            representatives[identity] = permutations
        key = tuple(signatures)
        saved = self.families.get(key)
        if saved is not None:
            self.family_hits += 1
            self.families.move_to_end(key)
            return saved
        self.family_builds += 1
        choices = []
        for identity, _ in signatures:
            calculation = space.registry.calculations[identity]
            result = space.registry.data[calculation.result]
            normalizers = []
            for scene, occurrence, output in representatives[identity].values():
                checkpoint()
                state = space._state(scene)
                state = replace(state,
                    pending=(PendingTerm((Factor(result.name, occurrence.output),), output),),
                    loaded=state.loaded | {result.name},
                    catalog=state.catalog.updated(((result.name, result.spec),)))
                action, spec, workspace, *_ = describe(state, 0, (0,), profile=space.profile)
                # Keep an intrinsic recipe cost even when this particular
                # prefix already has the normalizer cached.
                if action.reused:
                    state = replace(state, loaded=state.loaded - {action.tensor})
                    action, spec, workspace, *_ = describe(state, 0, (0,), profile=space.profile)
                normalizers.append(space.registry.calculation(action, spec, workspace))
            choices.append((identity, tuple(dict.fromkeys(normalizers))))
        return self._remember(self.families, key, tuple(choices), 65536)

    @staticmethod
    def _choose(space, choices, checkpoint):
        best = None
        for identity, normalizers in choices:
            checkpoint()
            seconds = 0.0
            for index in (identity, *normalizers):
                calc = space.registry.calculations[index]
                if calc.result not in space.available:
                    if not math.isfinite(calc.seconds) or calc.seconds < 0:
                        raise ValueError('terminal increments must be finite and nonnegative')
                    seconds += calc.seconds
            candidate = (seconds, identity, normalizers)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        return best

    def complete(self, space, checkpoint):
        self.requests += 1
        scalar = space._scalar_terminals
        key = None if scalar is None else scalar.key(space)
        if key is None:
            key = space.fingerprint()
        if key in self.tails:
            self.hits += 1
            self.tails.move_to_end(key)
            return self.tails[key]
        if scalar is not None and key[0] == 'scalar':
            value = scalar.completion(space, checkpoint)
            self.builds += 1
            self.logical_steps += len(value.calculations)
            self.scalar = scalar
            # The index tracked every rebuilt block since the retained base was
            # captured, so it hands the patch over instead of letting the
            # sharing walk re-derive it.
            return self._remember(self.tails, key,
                self._shared(value, space.debug, scalar.overrides(self.sequence)), 16)
        occupied, owners, main, normalizers = set(), {}, [], []
        seconds = 0.0
        for family in sorted(space.terminal_members):
            checkpoint()
            scenes = space.groups[family].keys()
            if occupied.intersection(scenes):
                self.fallbacks += 1
                return None
            occupied.update(scenes)
            choices = self._family_options(space, family, checkpoint)
            # Distinct families must have independent results, including
            # alternate layouts and required output normalizers. A result
            # alias across families can change later layout pricing.
            for identity, tails in choices:
                for index in (identity, *tails):
                    result = space.registry.calculations[index].result
                    previous = owners.setdefault(result, family)
                    if previous != family:
                        self.fallbacks += 1
                        return None
            cost, identity, tails = self._choose(space, choices, checkpoint)
            seconds += cost
            main.append((family, identity))
            normalizers.extend(tails)
        if len(occupied) != space.remaining:
            self.fallbacks += 1
            return None
        # All base families first, as in the stable-family stepwise domain;
        # the new unary family IDs are appended after the current families.
        blocks = {}
        for family, identity in main:
            blocks.setdefault(('main', family // 64), []).append(identity)
        for identity in sorted(set(normalizers)):
            blocks.setdefault(('normalizer', identity // 64), []).append(identity)
        chunks = []
        size = 0
        for bucket, identities in blocks.items():
            checkpoint()
            values = tuple(identities)
            key_chunk = (bucket, values)
            chunk = self.chunks.get(key_chunk)
            if chunk is None:
                chunk = self._remember(self.chunks, key_chunk, values, 8192)
            else:
                self.chunks.move_to_end(key_chunk)
            chunks.append(chunk)
            size += len(chunk)
        recipe = CalculationChunks(self._share(tuple(chunks), space.debug), size)
        value = TerminalCompletion(seconds, recipe)
        self.builds += 1
        self.logical_steps += len(recipe)
        return self._remember(self.tails, key, value, 16)

    def statistics(self):
        summary = dict(terminal_summary_requests=self.requests, terminal_summary_hits=self.hits,
            terminal_family_cache_hits=self.family_hits, terminal_family_cache_builds=self.family_builds,
            terminal_summary_builds=self.builds, terminal_summary_fallbacks=self.fallbacks,
            terminal_summary_logical_steps=self.logical_steps, terminal_summary_expansions=self.expansions)
        if self.scalar is not None:
            summary.update(self.scalar.statistics())
        return summary

    def clear(self):
        self.families.clear()
        self.tails.clear()
        self.chunks.clear()
        # The shared base is dropped with the caches; every retained tail keeps
        # its own reference to the sequence it was built from.
        self.sequence = None


def expand_completion(completion, prefix, registry):
    """Expand only the winner, using exactly its prefix's cache availability."""
    from .integer_space import IntegerOperation
    available = {op.result for op in prefix}
    result = list(prefix)
    for identity in completion.calculations:
        calc = registry.calculations[identity]
        reused = calc.result in available
        result.append(IntegerOperation(identity, calc.result, reused, 0.0 if reused else calc.seconds))
        available.add(calc.result)
    return tuple(result)
