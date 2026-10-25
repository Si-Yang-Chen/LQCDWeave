# 临时慢测清单

这些用例从常规测试中移出，因为单个用例在 CSNS CPU 上耗时超过约半分钟，
拖慢完整 pytest/unittest 回归。文件名使用 `tmp_*.py`，`unittest discover`、
默认 pytest 和 `tools/test_tiers.py --tier all` 均不收集；只在显式指定
`--tier temporary` 时运行。它们只是暂时移出，覆盖内容不变：改动对应模块时
仍须运行，原测试缩短到可接受耗时后应移回原文件并删除这里的副本。

```sh
python tools/test_tiers.py --tier temporary --list
python tools/test_tiers.py --tier temporary
```

每个文件继承原 `TestCase`，只运行移出的方法；夹具与辅助函数仍从原模块导入。

| 临时文件 | 移出的方法（原文件） | CSNS CPU 耗时 |
|---|---|---|
| `tmp_baryon_factor_network_scope.py` | `rows-mode-blocked`、`exchange-control` 两个范围用例（`correlator/expansion/test_baryon_factor_network_scope.py`） | 约 710 s |
| `tmp_baryon_row_structures.py` | 除常规三例外的全部行族结构（`correlator/expansion/test_baryon_row_structures.py`） | 约 1020 s |
| `tmp_dependency_tiers.py` | `test_wheel_and_sdist_include_all_symmetry_modules_and_mit_license`（`test_dependency_tiers.py`） | 约 204 s |
| `tmp_device_output_program.py` | `test_23056_term_segmented_ordered_sum_matches_cpu_recipe`（`correlator/execution/test_device_output_program.py`） | 约 70 s |
| `tmp_three_sector_closed_loop.py` | `test_joint_random_medium_and_high_average_to_full_wilson_trace`（`test_three_sector_closed_loop.py`） | 约 64 s |
| `tmp_baryon_spin_catalog.py` | `test_catalog_recipes_couple_through_existing_spatial_and_moving_interfaces`（`correlator/operators/test_baryon_spin_catalog.py`） | 约 44 s |
| `tmp_baryon_diquark_catalog.py` | `test_recipes_use_catalog_insertion_and_moving_interfaces`（`correlator/operators/test_baryon_diquark_catalog.py`） | 约 44 s |
| `tmp_exchange_independence.py` | `test_identical_vertices_transport_components_even_without_flavor_preserving_leg_swaps`（`correlator/symbolic/test_exchange_independence.py`） | 约 41 s |
| `tmp_baryon_mixed_exchange.py` | `test_complex_changed_basis_keeps_source_conjugation_and_nonzero_matrix_mixing`（`test_baryon_mixed_exchange.py`） | 约 37 s |
