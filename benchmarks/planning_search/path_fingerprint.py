"""Compare computation DAGs and contraction order without temporary names."""
import hashlib
import json
from collections import Counter

def digest(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()

def fingerprint(problem, actions):
    shapes = {key:tuple(spec.shape) for key,spec in problem.specs.items()}
    names = {key:digest(('input',key,shapes[key],spec.dtype.str))
             for key,spec in problem.specs.items()}
    sequence, records = [], {}
    for action in actions:
        if action.kind != 'contract':
            if action.kind not in ('load','release'):
                raise ValueError('unexpected action in this dense-only experiment: '+action.kind)
            continue
        equation = action.equation.normalized()
        dimensions = {}
        input_shapes = tuple(shapes[key] for key in action.operands)
        for axes, shape in zip(equation.inputs,input_shapes,strict=True):
            for axis,extent in zip(axes,shape,strict=True):
                if dimensions.setdefault(axis,extent) != extent:
                    raise ValueError('inconsistent contraction dimensions')
        output_shape = tuple(dimensions[axis] for axis in equation.output)
        record = (equation.inputs,equation.output,
                  tuple(names[key] for key in action.operands),input_shapes,output_shape,
                  action.used_np,action.kernel_path)
        identity = digest(record)
        names[action.tensor],shapes[action.tensor] = identity,output_shape
        if not action.reused:
            sequence.append(identity)
            records[identity] = record
    return dict(dag_sha256=digest(sorted(Counter(sequence).items())),
        contraction_order_sha256=digest(sequence),
        executed_contracts=len(sequence),unique_contract_nodes=len(records)),sequence,records
