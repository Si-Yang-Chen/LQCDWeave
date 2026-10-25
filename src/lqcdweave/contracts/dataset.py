"""Versioned scientific array descriptions; no I/O or backend dependencies."""

import hashlib
import json
from collections.abc import Mapping


def canonical_json(value):
    """JSON-native values only, with deterministic ordering and no NaN."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def stable_digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class DatasetSpec:
    """Detached, immutable-by-interface snapshot of an explicit array contract.

    axes is an ordered list of {name, coordinates}; coordinates retain order.
    bases maps vector/point-axis names to {data_id, indices}. data_id
    refers to the actual basis/point realization, not generation parameters or
    a file path. Signed integer storage supports explicit coordinate tables.
    Unknown provenance/identities may be registered but cannot pass a strict
    measurement compatibility check. definition excludes execution settings.
    """

    def __init__(self, document):
        self._json = canonical_json(document)
        d = self.document
        required = {"schema", "kind", "configuration", "axes", "dtype", "bases",
                    "definition", "conventions", "parents"}
        if not isinstance(d, dict) or set(d) != required or type(d["schema"]) is not int or d["schema"] != 1:
            raise ValueError("expected dataset schema 1 and its exact fields")
        if not isinstance(d["kind"], str) or not d["kind"]:
            raise ValueError("dataset kind is required")
        cfg = d["configuration"]
        if not isinstance(cfg, dict) or set(cfg) != {"ensemble", "configuration", "lattice"}:
            raise ValueError("configuration requires ensemble, configuration and lattice")
        if len(cfg["lattice"]) != 4 or any(type(x) is not int or x < 1 for x in cfg["lattice"]):
            raise ValueError("lattice must be positive [Lx,Ly,Lz,Lt]")
        if any(cfg[k] is not None and (not isinstance(cfg[k], str) or not cfg[k])
               for k in ("ensemble", "configuration")):
            raise ValueError("configuration identities must be nonempty strings or null")
        names = []
        for axis in d["axes"]:
            if set(axis) != {"name", "coordinates"} or not isinstance(axis["name"], str) or not axis["name"]:
                raise ValueError("each axis needs a name and explicit coordinates")
            coords = axis["coordinates"]
            if not isinstance(coords, list) or not coords or any(c is None for c in coords):
                raise ValueError("axis coordinates must be nonempty and known")
            if len({canonical_json(c) for c in coords}) != len(coords):
                raise ValueError("duplicate axis coordinates")
            names.append(axis["name"])
        collection = (d['kind'] == 'correlator' and not names and d['dtype'] is None
                      and isinstance(d['definition'], dict)
                      and d['definition'].get('encoding') == 'native-hdf5-fields-v1')
        if (not names and not collection) or len(set(names)) != len(names):
            raise ValueError("nonempty unique axis names required")
        if not collection and d["dtype"] not in ("int32", "int64", "float32", "float64", "complex64", "complex128"):
            raise ValueError("unsupported storage dtype")
        if any(not isinstance(d[k], dict) for k in ("bases", "definition", "conventions")):
            raise ValueError("bases, definition and conventions must be mappings")
        for name, basis in d["bases"].items():
            if name not in names or set(basis) != {"data_id", "indices"}:
                raise ValueError("basis must bind an existing axis to an actual dataset and indices")
            identity = basis["data_id"]
            if identity is not None and (not isinstance(identity, str) or not identity):
                raise ValueError("basis identity must be a nonempty string or null")
            indices = basis["indices"]
            if (indices != self.coordinates(name) or
                    any(type(i) is not int or i < 0 for i in indices)):
                raise ValueError("basis indices must equal ordered mode-axis coordinates")
        if not isinstance(d["parents"], list) or any(not isinstance(p, str) or not p for p in d["parents"]):
            raise ValueError("parents must be dataset references")
        from lqcdweave.contracts.time_selection import validate_time_coordinates
        validate_time_coordinates(d)

    @property
    def document(self):
        return json.loads(self._json)

    def _layout(self):
        # The public document/coordinates accessors stay detached. Cache only
        # immutable derivations, tied to the exact serialized snapshot.
        if getattr(self, '_layout_json', None) is not self._json:
            axes = self.document['axes']
            self._axis_names = tuple(axis['name'] for axis in axes)
            self._shape = tuple(len(axis['coordinates']) for axis in axes)
            self._coordinates_json = {
                axis['name']: canonical_json(axis['coordinates']) for axis in axes}
            self._definition_id = hashlib.sha256(self._json.encode('utf-8')).hexdigest()
            self._layout_json = self._json

    @property
    def definition_id(self):
        self._layout()
        return self._definition_id

    @property
    def axis_names(self):
        self._layout()
        return self._axis_names

    @property
    def shape(self):
        self._layout()
        return self._shape

    def coordinates(self, name):
        self._layout()
        try:
            return json.loads(self._coordinates_json[name])
        except KeyError:
            raise ValueError(f"missing axis {name}") from None


_DATASET_FIELDS = frozenset({"schema", "kind", "configuration", "axes", "dtype",
                              "bases", "definition", "conventions", "parents"})
_DROP_REFERENCE = object()


def _is_dataset_document(value):
    return isinstance(value, dict) and set(value) == _DATASET_FIELDS


def _reference_node(value, path, *, allowed):
    if not isinstance(value, dict) or "request" not in value:
        return None
    if set(value) != {"request"}:
        raise ValueError(f"request reference at {path} must have only the 'request' field")
    node = value["request"]
    if not isinstance(node, str) or not node:
        raise ValueError(f"request reference at {path} needs a nonempty node name")
    if not allowed:
        raise ValueError(f"logical request reference is not allowed at {path}")
    return node


def _display_path(path):
    result = ""
    for part in path:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            result += ("." if result else "") + str(part)
    return result or "<root>"


def _scan_request_references(document):
    references = {}

    def visit(value, path, *, semantic=False, id_slot=False,
              reference_kind="id", input_view=False, input_roles_allowed=False):
        location = _display_path(path)
        node = _reference_node(value, location, allowed=id_slot)
        if node is not None:
            references[path] = (node, reference_kind)
            return
        if isinstance(value, dict):
            if id_slot and "request" in value:
                raise ValueError(f"dataset ID at {location} must be a string, null, or request reference")
            if semantic:
                for key, child in value.items():
                    child_path = path + (key,)
                    if key == "input_roles" and input_roles_allowed and isinstance(child, dict):
                        for role, identity in child.items():
                            visit(identity, child_path + (role,), semantic=True,
                                  id_slot=True, reference_kind="id", input_view=input_view)
                    elif key == "input_views" and isinstance(child, dict):
                        for role, embedded in child.items():
                            embedded_path = child_path + (role,)
                            if _is_dataset_document(embedded):
                                visit_spec(embedded, embedded_path)
                            else:
                                visit(embedded, embedded_path, semantic=True,
                                      input_view=True)
                    elif key == "parents" and input_view and isinstance(child, list):
                        for index, parent in enumerate(child):
                            visit(parent, child_path + (index,), semantic=True,
                                  id_slot=True, reference_kind="parent", input_view=True)
                    elif key == "definition":
                        visit(child, child_path, semantic=True,
                              input_view=input_view, input_roles_allowed=True)
                    elif key == "conventions":
                        visit(child, child_path, semantic=True, input_view=input_view)
                    elif key in ("data_id",) or key.endswith("_id"):
                        visit(child, child_path, semantic=True, id_slot=True,
                              reference_kind="id", input_view=input_view)
                    else:
                        visit(child, child_path, semantic=True, input_view=input_view)
            else:
                for key, child in value.items():
                    if key == "request" and set(value) == {"request"}:
                        # This is handled above; keep the branch defensive if
                        # a future mapping implementation changes its type.
                        continue
                    visit(child, path + (key,), input_view=input_view)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + (index,), semantic=semantic,
                      input_view=input_view)

    def visit_spec(spec_document, prefix):
        for key, value in spec_document.items():
            field_path = prefix + (key,)
            if key == "parents" and isinstance(value, list):
                for index, parent in enumerate(value):
                    visit(parent, field_path + (index,), semantic=True,
                          id_slot=True, reference_kind="parent")
            elif key == "bases" and isinstance(value, dict):
                for axis, basis in value.items():
                    if not isinstance(basis, dict):
                        visit(basis, field_path + (axis,))
                        continue
                    for basis_field, item in basis.items():
                        visit(item, field_path + (axis, basis_field),
                              semantic=(basis_field == "data_id"),
                              id_slot=(basis_field == "data_id"),
                              reference_kind="id")
            elif key in ("definition", "conventions"):
                visit(value, field_path, semantic=True,
                      input_roles_allowed=(key == "definition"))
            else:
                visit(value, field_path)

    visit_spec(document, ())
    return references, tuple(sorted({node for node, _ in references.values()}))


def _transform_request_references(value, path, references, *, actual_ids=None,
                                  for_validation=False):
    reference = references.get(path)
    if reference is not None:
        node, kind = reference
        if for_validation:
            return _DROP_REFERENCE if kind == "parent" else None
        return actual_ids[node]
    if isinstance(value, dict):
        return {
            key: transformed
            for key, child in value.items()
            if (transformed := _transform_request_references(
                child, path + (key,), references, actual_ids=actual_ids,
                for_validation=for_validation)) is not _DROP_REFERENCE
        }
    if isinstance(value, list):
        transformed = []
        for index, child in enumerate(value):
            item = _transform_request_references(
                child, path + (index,), references, actual_ids=actual_ids,
                for_validation=for_validation)
            if item is not _DROP_REFERENCE:
                transformed.append(item)
        return transformed
    return value


def _validate_embedded_input_views(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "input_views" and isinstance(child, dict):
                for embedded in child.values():
                    if _is_dataset_document(embedded):
                        _validate_embedded_input_views(DatasetSpec(embedded).document)
                    else:
                        _validate_embedded_input_views(embedded)
            else:
                _validate_embedded_input_views(child)
    elif isinstance(value, list):
        for child in value:
            _validate_embedded_input_views(child)


def scientific_input_bindings(spec):
    """Collect IDs only from the explicit reference slots accepted by requests.

    This mirrors ``_scan_request_references``: top-level parents and basis IDs,
    semantic ``data_id``/``*_id`` fields, input roles, and embedded input
    specifications. It intentionally does not search arbitrary strings.
    """
    if not isinstance(spec, DatasetSpec):
        raise TypeError("spec must be a DatasetSpec")
    identities = set()
    role_views = {}

    def add_identity(value):
        if isinstance(value, str) and value:
            identities.add(value)

    def visit_semantic(value, *, input_view=False, input_roles_allowed=False):
        if isinstance(value, Mapping):
            roles = value.get("input_roles") if input_roles_allowed else None
            if isinstance(roles, Mapping):
                for identity in roles.values():
                    add_identity(identity)
            for key, child in value.items():
                if key == "input_roles" and input_roles_allowed:
                    continue
                if key == "input_views" and isinstance(child, Mapping):
                    for embedded in child.values():
                        if _is_dataset_document(embedded):
                            visit_spec(DatasetSpec(embedded).document)
                        else:
                            visit_semantic(embedded, input_view=True)
                elif key == "parents" and input_view and isinstance(child, list):
                    for parent in child:
                        add_identity(parent)
                elif key == "definition":
                    visit_semantic(child, input_view=input_view,
                                   input_roles_allowed=True)
                elif key == "conventions":
                    visit_semantic(child, input_view=input_view)
                elif key == "data_id" or key.endswith("_id"):
                    add_identity(child)
                    # Keep traversing the semantic ID slot just as
                    # _scan_request_references does. DatasetRequest permits
                    # a logical reference nested below an ID-named field, so
                    # the resolved output may carry its actual ID at a deeper
                    # path (for example definition.gauge_id.nested_id).
                    visit_semantic(child, input_view=input_view)
                else:
                    visit_semantic(child, input_view=input_view)
        elif isinstance(value, list):
            for child in value:
                visit_semantic(child, input_view=input_view)

    def visit_spec(document):
        for key, value in document.items():
            if key == "parents" and isinstance(value, list):
                for parent in value:
                    add_identity(parent)
            elif key == "bases" and isinstance(value, Mapping):
                for basis in value.values():
                    if isinstance(basis, Mapping):
                        add_identity(basis.get("data_id"))
            elif key == "definition":
                visit_semantic(value, input_roles_allowed=True)
            elif key == "conventions":
                visit_semantic(value)

    visit_spec(spec.document)
    # Runtime evidence names roles from this DatasetSpec's top-level
    # definition. Embedded input specifications contribute explicit actual-ID
    # references above, but their local role names must not enter this scope:
    # an embedded ``basis`` view cannot stand in for this output's own
    # ``basis`` binding.
    definition = spec.document["definition"]
    roles = definition.get("input_roles")
    views = definition.get("input_views")
    if isinstance(roles, Mapping):
        for role, identity in roles.items():
            add_identity(identity)
            if not isinstance(identity, str) or not identity:
                continue
            declared_views = role_views.setdefault(role, {}).setdefault(identity, set())
            if (isinstance(views, Mapping) and role in views and
                    _is_dataset_document(views[role])):
                view = DatasetSpec(views[role])
                declared_views.add(canonical_json(view.document))
    return identities, role_views


class DatasetRequest:
    """Detached unresolved request template for one eventual ``DatasetSpec``.

    Logical references use exactly ``{"request": node_name}`` and are accepted
    only in parent or scientific-ID fields. The template is validated through
    ``DatasetSpec`` after references are removed from a private validation
    copy; no placeholder identity is created or exposed as a scientific spec.
    """

    def __init__(self, template, *, subset_axes=(), data_id=None):
        if isinstance(template, DatasetSpec):
            document = template.document
        elif isinstance(template, Mapping):
            document = json.loads(canonical_json(template))
        else:
            raise TypeError("template must be a DatasetSpec or JSON mapping")
        self._json = canonical_json(document)
        document = json.loads(self._json)
        self._references, self._dependencies = _scan_request_references(document)
        validation_document = _transform_request_references(
            document, (), self._references, for_validation=True)
        validated = DatasetSpec(validation_document)
        _validate_embedded_input_views(validation_document)

        if isinstance(subset_axes, str):
            raise TypeError("subset_axes must be an iterable of axis names")
        try:
            subset_axes = tuple(subset_axes)
        except TypeError as exc:
            raise TypeError("subset_axes must be an iterable of axis names") from exc
        if any(not isinstance(axis, str) or not axis for axis in subset_axes):
            raise TypeError("subset_axes entries must be nonempty axis names")
        if len(set(subset_axes)) != len(subset_axes):
            raise ValueError("subset_axes contains a duplicate axis")
        unknown_axes = set(subset_axes) - set(validated.axis_names)
        if unknown_axes:
            raise ValueError(f"unknown subset axis: {sorted(unknown_axes)[0]}")
        if data_id is not None and (not isinstance(data_id, str) or not data_id):
            raise ValueError("data_id must be a nonempty string or None")
        self._subset_axes = subset_axes
        self._data_id = data_id

    @property
    def dependencies(self):
        return self._dependencies

    @property
    def subset_axes(self):
        return self._subset_axes

    @property
    def data_id(self):
        return self._data_id

    @property
    def template(self):
        """Return a detached copy retaining logical references as request data."""
        return json.loads(self._json)

    def resolve(self, actual_ids):
        """Resolve every dependency to a real, nonempty parent dataset ID."""
        if not isinstance(actual_ids, Mapping):
            raise TypeError("actual_ids must be a mapping from request names to dataset IDs")
        missing = [
            name for name in self.dependencies
            if name not in actual_ids or not isinstance(actual_ids[name], str) or not actual_ids[name]
        ]
        if missing:
            raise ValueError("missing or invalid actual dataset IDs for requests: " + ", ".join(missing))
        document = self.template
        resolved = _transform_request_references(
            document, (), self._references, actual_ids=actual_ids)
        spec = DatasetSpec(resolved)
        _validate_embedded_input_views(resolved)
        return spec

    def resolved_bindings(self, spec):
        """Recover real dependency IDs from an exactly resolved specification.

        This is pure validation of a stored definition, not evidence that its
        parents are available or complete. Storage observations must establish
        that separately before planning or execution can use the bindings.
        """
        if not isinstance(spec, DatasetSpec):
            raise TypeError("resolved spec must be a DatasetSpec")
        document = spec.document
        bindings = {}
        for path, (node, _) in self._references.items():
            value = document
            try:
                for key in path:
                    value = value[key]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError("resolved dependency reference is missing") from exc
            if not isinstance(value, str) or not value:
                raise ValueError("resolved dependency requires a nonempty actual dataset ID")
            if node in bindings and bindings[node] != value:
                raise ValueError("resolved dependency has conflicting actual dataset IDs")
            bindings[node] = value
        if self.resolve(bindings).definition_id != spec.definition_id:
            raise ValueError("resolved definition or input dataset differs from request")
        return bindings
