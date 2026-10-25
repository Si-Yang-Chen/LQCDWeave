"""Compose structured outputs using the existing immutable OutputNode IR."""
from dataclasses import replace

from lqcdweave.contracts.correlator.api import OutputRecipe


def compose_output_recipe(root, replacements, term_ids, coordinates):
    """Replace construction-time basis leaves with validated numeric recipes."""
    from .core import _validate_output_node

    nodes = {}

    def resolve(node):
        key = id(node)
        if key not in nodes:
            if node.kind == 'term':
                try:
                    result = replacements[node.value.term_id]
                except KeyError as exc:
                    raise ValueError(f'unknown structured basis result {node.value.term_id!r}') from exc
            else:
                result = replace(node, children=tuple(resolve(child) for child in node.children))
            nodes[key] = result
        return nodes[key]

    resolved = resolve(root)
    _validate_output_node(resolved, set(term_ids))
    return OutputRecipe(resolved, tuple(term_ids), coordinates)


__all__ = ['compose_output_recipe']
