"""List the proposed D SPD + L=0 magnetic basis, partitioned by reflection.

This inventory does not change the SPD production script or submit jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lqcdweave.symmetry import restrict_irrep

from lqcdweave.operators.single_hadron.api import gauge_recipe, meson55_operators


REPRESENTATIVES = ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1), (0, 0, 2))
DIMENSIONS = {"A1": 1, "A2": 1, "B1": 1, "B2": 1, "E": 2, "T1": 3, "T2": 3}


def inventory():
    spatial_rows = list(gauge_recipe("v2").rows)
    linked = {row['id']: any(term['path'] for term in row['terms']) for row in spatial_rows}
    candidates = tuple(op for op in meson55_operators(version="v2")
        if (op.kernel_origin[0] == "ordinary" and op.kernel_origin[1] <= 2)
        or (op.kernel_origin[0] == "magnetic" and op.kernel_origin[1] == 0))
    recipes = tuple(op for op in candidates
        if not any(linked[kernel] for kernel in op.kernel_ids)
        or op.gamma_name in ('gamma5', 'gammai'))
    partitions = []
    for momentum in REPRESENTATIVES:
        p2 = sum(p*p for p in momentum)
        irreps = (("A1", "A2", "E", "T1", "T2") if p2 == 0 else
                  ("A1", "A2", "B1", "B2", "E") if p2 in (1, 4) else
                  ("A1", "A2", "B1", "B2") if p2 == 2 else ("A1", "A2", "E"))
        for irrep in irreps:
            for parity in ((-1, 1) if p2 == 0 else (None,)):
                for reflection in (1, -1):
                    entries = []
                    for op in recipes:
                        if op.reflection != reflection:
                            continue
                        multiplicity = (int(op.irrep == irrep and op.parity == parity) if p2 == 0
                            else restrict_irrep(momentum, op.irrep, irrep, op.parity)["multiplicity"])
                        entries.extend(dict(name=op.name, copy_index=i, gamma=op.gamma_name,
                            rest_irrep=op.irrep, rest_parity=op.parity, c_label=op.c_factor,
                            reflection=op.reflection, kernel_ids=op.kernel_ids,
                            has_gauge_link=any(linked[kernel] for kernel in op.kernel_ids),
                            kernel_origin=op.kernel_origin) for i in range(multiplicity))
                    name = f"p2_{p2}_{irrep}" + (f"_P{'m' if parity < 0 else 'p'}" if parity else "")
                    name += f"_r{'p' if reflection > 0 else 'm'}"
                    partitions.append(dict(name=name, momentum=momentum, p2=p2, irrep=irrep,
                        parity=parity, reflection=reflection, rows=DIMENSIONS[irrep],
                        operators_per_row=len(entries), radial_operators_per_row=20*len(entries),
                        ordinary_per_row=sum(o['kernel_origin'][0] == 'ordinary' for o in entries),
                        magnetic_per_row=sum(o['kernel_origin'][0] == 'magnetic' for o in entries),
                        operators=entries))
    elements = sum(p['rows']*p['radial_operators_per_row']**2 for p in partitions)
    return dict(version='meson55-v2', flavor='ubar c',
        definition='Any nonempty gauge-link path: gamma5/gammai only; empty paths: all Dirac structures; grouped by catalog reflection r',
        spatial_catalog_sha256=gauge_recipe("v2").catalog_sha256,
        reflection_semantics='Linear time-reflection metadata; grouping does not establish vanishing cross-r correlators.',
        c_mixing=True, equivalent_directions_repeated=False,
        light_edges=[0,16,32,48,70], charm_edges=[0,16,32,48,70,120],
        radial_pairs=20, catalog_sha256=gauge_recipe("v2").coupling_sha256,
        rest_multiplets=len(recipes), partition_count=sum(bool(p['operators']) for p in partitions),
        empty_partition_count=sum(not p['operators'] for p in partitions),
        fields=sum(p['rows'] for p in partitions if p['operators']),
        diagonal_r_blocks_complex_elements_per_time_pair=elements,
        diagonal_r_blocks_complex64_bytes_per_1600_pairs=elements*8*1600,
        scope='Inventory proposal only; current running SPD benchmark retains its original definition.',
        partitions=partitions)


def markdown(document):
    lines = ['# D 介子 SPD + L=0 磁激发的 r 分区清单', '',
        '由 `tools/campaigns/d_meson_spd_r_magnetic_inventory.py` 从 meson55-v2 目录生成。', '',
        '保留普通空间目录行0–8，并加入磁型 B_S 行25–27。这里 L=0 指 B 场外的轨道部分，'
        'B 自身按 T1 变换，不将整个磁型空间核称为标量。每个动量壳仅一个代表方向。', '',
        '按实际路径判断：任一空间核项含非空 gauge-link 路径时，只保留 gamma5、gammai；'
        '全部路径为空时，保留全部八类 Dirac 结构。当前目录的局域 S（行0）为空路径，'
        'P、D_E、D_T2、B_S 均含链接；闭合回路即使位移为零也属于含链接。'
        '含链接的 r=+1 组只含 gammai，r=−1 组只含 gamma5。空分区仅列作核对，不生成矩阵。', '',
        'r 是目录 `reflection`（线性时间反射）标签。下列按 r 分组本身不证明跨 r 关联为零；'
        '若只计算两个对角块，就会省略跨 r 混合。不同 C 标签在各组内继续混合。', '',
        'light 边界 `[0,16,32,48,70]`，charm 边界 `[0,16,32,48,70,120]`；'
        '每个条目仍有20个有序径向算符。表示行不是额外算符副本，等价动量方向不重复计数。', '',
        '| 分区 | 代表动量 | 行数 | 普通 | 磁型 | N/行 | 径向扩展矩阵/行 |',
        '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for p in document['partitions']:
        n=p['radial_operators_per_row']
        lines.append(f"| {p['name']} | {tuple(p['momentum'])} | {p['rows']} | {p['ordinary_per_row']} | {p['magnetic_per_row']} | {p['operators_per_row']} | {n}×{n} |")
    lines += ['', f"共{document['partition_count']}个非空 r 分区、{document['fields']}个表示行字段；另有{document['empty_partition_count']}个空分区。", '',
        '以下每行是一个独立投影算符；moving-frame 的 copy_index 全部保留。'
        '名称中的 A1/E/T1/T2 是母体静止表示，分区标题是最终小群表示。', '']
    for p in document['partitions']:
        lines += [f"## {p['name']}", '', '| 目录名称 | copy_index | 静止 P | C标签 | 来源 |',
            '| --- | ---: | ---: | ---: | --- |']
        for op in p['operators']:
            origin=op['kernel_origin']
            lines.append(f"| `{op['name']}` | {op['copy_index']} | {op['rest_parity']:+d} | {op['c_label']:+d} | {origin[0]}, L={origin[1]} |")
        lines.append('')
    return '\n'.join(lines).rstrip()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path)
    parser.add_argument('--markdown', type=Path)
    args=parser.parse_args()
    document=inventory()
    if args.json:
        args.json.write_text(json.dumps(document,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    if args.markdown:
        args.markdown.write_text(markdown(document)+'\n',encoding='utf-8')
    if not args.json and not args.markdown:
        print(json.dumps(document,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
