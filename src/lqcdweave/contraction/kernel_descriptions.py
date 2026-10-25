"""W delivers X's pure declaration through C; P never imports X."""
from dataclasses import replace


def attach_vertex_kernel_descriptions(expanded):
    if expanded.inputs is None:
        return expanded
    from lqcdweave.contraction.execution.api import describe_vertex_kernel
    descriptor = describe_vertex_kernel('baryon')
    meson_descriptor = describe_vertex_kernel('meson')
    memo = {}

    def inputs(value):
        if id(value) in memo:
            return memo[id(value)]
        problem = value.problem
        if problem is not None and problem.baryon_joins:
            problem = replace(problem, baryon_joins=tuple(
                replace(join, kernel_description=descriptor) for join in problem.baryon_joins))
        if problem is not None and problem.meson_cores:
            problem = replace(problem, terms=tuple(replace(term,
                kernel=replace(term.kernel, kernel_description=meson_descriptor))
                for term in problem.terms))
        batches = tuple(inputs(item) for item in value.task_batches)
        candidates = {key: tuple(inputs(item) for item in group)
                      for key, group in value.batch_candidates.items()}
        unchanged = (problem is value.problem
            and all(a is b for a, b in zip(batches, value.task_batches))
            and all(all(a is b for a, b in zip(group, value.batch_candidates[key]))
                    for key, group in candidates.items()))
        result = value if unchanged else replace(value, problem=problem,
            task_batches=batches, batch_candidates=candidates)
        memo[id(value)] = result
        return result

    return replace(expanded, inputs=inputs(expanded.inputs))
