"""Native glueball data terms, compared to direct complex scalar products."""
from unittest.mock import patch

import numpy as np
import pytest

from lqcdweave.adapters.glueball import ArrayGlueball
from lqcdweave.contraction.binding.api import Glueball
from lqcdweave.contraction.correlator import calculate
from lqcdweave.operators.correlator.api import Flavor, Hadron, HadronIrrep, Tag, gen_correlator


def particle(name="G", flavor="", slot=0):
    return Hadron(HadronIrrep(name, (0, 0, 0), "A_1", 1, Tag(slot, 0))[0], Flavor(flavor))


def glue_input(values, cfg="a"):
    return ArrayGlueball(np.asarray(values, dtype=complex).reshape(1, 1, -1),
                         components=("G",), momentum_list=((0, 0, 0),),
                         phase_convention="site", cfg=cfg)


def binding(reader):
    return lambda row: Glueball(reader, "G", source=row.dagger)


@pytest.mark.parametrize("structured", [False, True])
def test_pure_glue_is_owned_current_data_without_contraction(structured):
    reader = glue_input([2+3j, 5-7j])
    g = particle()
    correlator = gen_correlator([[g], [g]], ["sink", "source"], [False, True],
                               structured=structured)
    plan = calculate(correlator, vertex_map=binding(reader), configuration="a",
                     time_axes={"sink": 1, "source": 0}, optimize="none", execute=False)
    with patch("numpy.einsum", side_effect=AssertionError("glueball must not contract")):
        first = plan.run().value
        np.testing.assert_allclose(first, [[(5-7j)*(2-3j)]])
        reader.data[0, 0, 0] = 11-2j
        second = plan.run().value
    np.testing.assert_allclose(second, [[(5-7j)*(11+2j)]])
    np.testing.assert_allclose(first, [[(5-7j)*(2-3j)]])


@pytest.mark.parametrize("structured", [False, True])
def test_scan_and_replay_match_outer_product(structured):
    reader = glue_input([2+3j, 5-7j, -3+1j])
    g = particle()
    correlator = gen_correlator([[g], [g]], ["sink", "source"], [False, True],
                               structured=structured)
    fixed = calculate(correlator, vertex_map=binding(reader), optimize="none",
                      time_axes={"sink": 0, "source": 0}, execute=False)
    expected = reader.data[0, 0, :, None] * reader.data[0, 0, None, :].conj()
    with patch("numpy.einsum", side_effect=AssertionError("glueball must not contract")):
        scanned = calculate(correlator, vertex_map=binding(reader), optimize="none",
                            time_axes={"sink": range(3), "source": range(3)}).value
        replay = fixed.run(time_axes={"sink": range(3), "source": range(3)}, batch="auto").value
    np.testing.assert_allclose(np.asarray(scanned).reshape(3, 3), expected)
    np.testing.assert_allclose(np.asarray(replay).reshape(3, 3), expected)


@pytest.mark.parametrize("structured", [False, True])
def test_archive_reads_new_configuration(tmp_path, structured):
    from lqcdweave.contraction.correlator import save_calculation_plan
    from lqcdweave.contraction.correlator_cfg import complete_calculation
    g = particle()
    reader = glue_input([2+3j, 5-7j])
    correlator = gen_correlator([[g], [g]], ["sink", "source"], [False, True],
                               structured=structured)
    plan = calculate(correlator, vertex_map=binding(reader), configuration="a",
                     time_axes={"sink": 1, "source": 0}, optimize="none", execute=False)
    archive = tmp_path / "glueball.json.gz"
    save_calculation_plan(archive, complete_calculation(plan))
    current = glue_input([1-4j, -2+3j], cfg="b")
    with patch("lqcdweave.contraction.correlator.expand_correlator",
               side_effect=AssertionError("archive must not expand")):
        value = calculate(plan_archive=archive, configuration="b", vertex_map=current).value
    np.testing.assert_allclose(value, [[(-2+3j)*(1+4j)]])


@pytest.mark.parametrize("flavor", ["ss", "uds"])
def test_mixed_glue_quark_factors_keep_existing_contraction(flavor):
    from lqcdweave.adapters.elemental import ArrayBaryonElemental, ArrayElemental
    from lqcdweave.adapters.propagator import ArrayPropagator
    from lqcdweave.contraction.binding.api import Baryon, Meson
    from lqcdweave.contracts.elemental import BaryonComponent, ElementalComponent
    from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm, SparseSpinTensor
    rng = np.random.default_rng(50)
    def random(shape):
        return rng.normal(size=shape) + 1j*rng.normal(size=shape)
    if flavor == "uds":
        component = BaryonComponent.paths()
        reader = ArrayBaryonElemental({"vvv": random((1, 1, 2, 1, 1, 1))}, (component,), cfg="a")
        term = OperatorTerm.from_baryon(component, SparseSpinTensor(((0, 1, 2, 1+.3j),)))
        cls = Baryon
    else:
        component = ElementalComponent(0)
        reader = ArrayElemental({"v2v": random((1, 1, 2, 1, 1))}, (component,),
                                phase_convention="anchor", cfg="a")
        term = OperatorTerm(5, component, 0, .3+.7j)
        cls = Meson
    op = Operator.from_terms("H", (term,))
    vertices = {dagger: cls(reader, op, dagger, phase_convention="anchor") for dagger in (False, True)}
    prop = ArrayPropagator({"vsv": random((2, 2, 4, 4, 1, 1))}, time_convention="absolute", cfg="a")
    glue = glue_input([2+3j, 5-7j])
    h, g = particle("H", flavor), particle("G", "", 1)
    mixed = Hadron(h.irrep_row*g.irrep_row, h.flavor_structure*g.flavor_structure)
    options = dict(configuration="a", usedNe=1, optimize="none",
                   time_axes={"sink": 1, "source": 0},
                   propagator_map={name: prop for name in ("S^q", "S^s", r"S^q_\mathrm{local}", r"S^s_\mathrm{local}")})
    base = calculate(gen_correlator([[h], [h]], ["sink", "source"], [False, True]),
                     vertex_map=lambda row: vertices[row.dagger], **options).value
    result = calculate(gen_correlator([[mixed], [mixed]], ["sink", "source"], [False, True]),
                       vertex_map=lambda row: Glueball(glue, "G", source=row.dagger)
                       if row.hadron_name == "G" else vertices[row.dagger], **options).value
    np.testing.assert_allclose(result, base*(5-7j)*(2-3j), rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("format", ["npy", "hdf5"])
def test_dataset_template_and_archive_definition(tmp_path, format):
    from lqcdweave.contracts.dataset import DatasetSpec
    from lqcdweave.storage.dataset import register_npy, register_hdf5, DatasetReader
    from lqcdweave.adapters.preset import Dataset
    from lqcdweave.contraction.correlator import save_calculation_plan
    from lqcdweave.contraction.correlator_cfg import complete_calculation
    data = np.array([[[2+3j], [5-7j]]])  # stored momentum,time,component
    root = tmp_path / "input"
    def register(cfg, scale=1, definition="measured-Wilson-loop"):
        path = tmp_path / f"{cfg}.{format}"
        if format == "npy":
            np.save(path, data*scale)
        else:
            h5py = pytest.importorskip("h5py")
            with h5py.File(path, "w") as handle:
                handle["data"] = data*scale
        spec = DatasetSpec(dict(
            schema=1, kind="glueball",
            configuration=dict(ensemble="test", configuration=cfg, lattice=[1, 1, 1, 2]),
            axes=[dict(name="momentum", coordinates=[[1, 0, 0]]),
                  dict(name="time", coordinates=[0, 1]), dict(name="component", coordinates=["G"])],
            dtype="complex128", bases={}, definition=dict(operator=definition),
            conventions=dict(phase_convention="site"), parents=[]))
        register = register_npy if format == "npy" else register_hdf5
        register(root, path, spec, provenance={"test": "glueball"})
    register("a")
    register("b", 2)
    register("bad", 3, "different-loop")
    g = particle()
    correlator = gen_correlator([[g], [g]], ["sink", "source"], [False, True])
    source = Dataset(root)
    vertices = lambda row: Glueball(source, "G", source=row.dagger)
    with patch.object(DatasetReader, "read", side_effect=AssertionError("planning reads data")):
        plan = calculate(correlator, vertex_map=vertices, configuration="a", optimize="none",
                         time_axes={"sink": 1, "source": 0}, execute=False)
    expected = (5-7j)*(2-3j)
    np.testing.assert_allclose(plan.run().value, [[expected]])
    archive = tmp_path / "dataset-plan.json.gz"
    save_calculation_plan(archive, complete_calculation(plan))
    with source.load("b") as current:
        replay = calculate(plan_archive=archive, configuration="b", vertex_map=current).value
        np.testing.assert_allclose(replay, [[4*expected]])
    with source.load("bad") as current, pytest.raises(ValueError, match="conventions"):
        calculate(plan_archive=archive, configuration="bad", vertex_map=current)
    for reader in plan.inputs.loader.input_readers.values():
        reader.close()


def test_npy_transpose_scalar_and_lifetime(tmp_path):
    data = (np.arange(24).reshape(2, 4, 3)+1j).astype(complex)
    path = tmp_path / "glue.npy"
    np.save(path, data)
    with ArrayGlueball(path, components=("a", "b", "c"), momentum_list=((1, 0, 0), (-1, 0, 0)),
                      phase_convention="site", axis_order=("momentum", "time", "component")) as reader:
        assert reader.describe_block("b", 1, 2).shape == ()
        np.testing.assert_allclose(reader.block("b", 1, 2), data[1, 2, 1])
        np.testing.assert_allclose(reader.read_slab("b", 1, (3, 0, 2)), data[1, [3, 0, 2], 1])
        assert np.shares_memory(reader.data, reader._mapped)
    with pytest.raises(RuntimeError, match="closed"):
        reader.block("b", 1, 2)


@pytest.mark.parametrize("optimize", ["none", "anytime", "meson_matrix"])
def test_dataset_output_and_saved_replay(tmp_path, optimize):
    from lqcdweave.contraction.correlator import save_calculation_plan
    from lqcdweave.contraction.correlator_output import load_correlator_output
    reader = glue_input([2+3j, 5-7j, 1+2j])
    g = particle()
    correlator = gen_correlator([[g], [g]], ["sink", "source"], [False, True])
    output = tmp_path / "results"
    selected = calculate(correlator, vertex_map=binding(reader), configuration="a",
                         output=output, time_pairs=[{"sink": 1, "source": 0}, {"sink": 0, "source": 2}],
                         optimize=optimize, execute=False,
                         search=None if optimize == "none" else
                         dict(root_width=1, branch_width=1, max_states=1000, component_workers=1))
    archive = tmp_path / "output-plan.json.gz"
    save_calculation_plan(archive, selected)
    result = calculate(plan_archive=archive, vertex_map=reader, configuration="a", output=output)
    assert result.report["completed"] == ("a",)
    actual = load_correlator_output(output, configuration="a")
    np.testing.assert_allclose(actual, [[[(5-7j)*(2-3j), (2+3j)*(1-2j)]]])


def test_compact_times_preserve_declared_slots():
    from lqcdweave.contracts.time_selection import TimeSelectionRecord
    reader = ArrayGlueball(np.array([[[2+3j, 5-7j]]]), components=("G",),
                          momentum_list=((0, 0, 0),), phase_convention="site",
                          time_selection=TimeSelectionRecord(16, (3, 11)))
    g = particle()
    c = gen_correlator([[g], [g]], ["sink", "source"], [False, True])
    value = calculate(c, vertex_map=binding(reader), time_axes={"sink": 1, "source": 0},
                      optimize="none").value
    np.testing.assert_allclose(value, [[(5-7j)*(2-3j)]])


def test_binding_rejects_quark_edges_before_reads():
    reader = glue_input([2+3j, 5-7j])
    h = particle("M", "ss")
    c = gen_correlator([[h], [h]], ["sink", "source"], [False, True])
    with patch.object(reader, "block", side_effect=AssertionError("invalid binding read")):
        with pytest.raises(ValueError, match="quark edge"):
            calculate(c, vertex_map=binding(reader), time_axes={"sink": 1, "source": 0},
                      optimize="none")


def test_zero_degree_requires_glueball_and_real_edges_still_need_propagators():
    from lqcdweave.adapters.elemental import ArrayElemental
    from lqcdweave.contraction.binding.api import Meson
    from lqcdweave.contracts.elemental import ElementalComponent
    from lqcdweave.operators.single_hadron.api import Operator, OperatorTerm
    component = ElementalComponent(0)
    reader = ArrayElemental({"v2v": np.ones((1, 1, 2, 1, 1), complex)}, (component,),
                            phase_convention="anchor")
    op = Operator.from_terms("M", (OperatorTerm(5, component, 0),))
    bind = lambda row: Meson(reader, op, row.dagger, phase_convention="anchor")
    with patch.object(reader, "block", side_effect=AssertionError("invalid binding read")):
        for flavor, error, message in (("", ValueError, "zero-quark"), ("ss", KeyError, "propagator_map")):
            h = particle("M", flavor)
            c = gen_correlator([[h], [h]], ["sink", "source"], [False, True])
            with pytest.raises(error, match=message):
                calculate(c, vertex_map=bind, time_axes={"sink": 1, "source": 0}, optimize="none")


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_complex_linear_combination_and_repeated_insertions(dtype):
    import sympy as sp
    raw = np.array([[[2, 5]], [[3, -4]]], dtype=dtype)
    if np.dtype(dtype).kind == "c":
        raw += 1j
    reader = ArrayGlueball(raw, components=("G", "H"), momentum_list=((1, 0, 0),),
                          phase_convention="site")
    a = HadronIrrep("G", (1, 0, 0), "A_1", 1, Tag(0, 0))[0]
    b = HadronIrrep("H", (1, 0, 0), "A_1", 1, Tag(0, 0))[0]
    linear = Hadron((1+sp.I)*a + (2-sp.I)*b, Flavor(""))
    repeated = Hadron(linear.irrep_row**2, linear.flavor_structure**2)
    c = gen_correlator([[repeated], [linear]], ["sink", "source"], [False, True])
    bind = lambda row: Glueball(reader, row.hadron_name, source=row.dagger)
    with patch.object(reader, "block", side_effect=AssertionError("planning read")):
        plan = calculate(c, vertex_map=bind, time_axes={"sink": 1, "source": 0},
                         optimize="none", execute=False)
    amplitude = (1+1j)*raw[0, 0].astype(complex) + (2-1j)*raw[1, 0].astype(complex)
    np.testing.assert_allclose(plan.run().value, [[amplitude[1]**2*amplitude[0].conj()]],
                               rtol=2e-6, atol=2e-6)


def test_archive_runs_in_fresh_process(tmp_path):
    import subprocess
    import sys
    from lqcdweave.contraction.correlator import save_calculation_plan
    from lqcdweave.contraction.correlator_cfg import complete_calculation
    g = particle()
    c = gen_correlator([[g], [g]], ["sink", "source"], [False, True])
    plan = calculate(c, vertex_map=binding(glue_input([2+3j, 5-7j])), configuration="a",
                     time_axes={"sink": 1, "source": 0}, optimize="none", execute=False)
    archive = tmp_path / "fresh-process.json.gz"
    save_calculation_plan(archive, complete_calculation(plan))
    script = """
import sys
import numpy as np
from contextlib import ExitStack
from unittest.mock import patch
from lqcdweave.adapters.glueball import ArrayGlueball
from lqcdweave.contraction.correlator import calculate
reader = ArrayGlueball(np.array([[[1-4j, -2+3j]]]), components=("G",),
    momentum_list=((0,0,0),), phase_convention="site", cfg="b")
with ExitStack() as stack:
    for name in ("expand_correlator", "find_contraction_path"):
        stack.enter_context(patch("lqcdweave.contraction.correlator."+name,
                                  side_effect=AssertionError(name)))
    value = calculate(plan_archive=sys.argv[1], configuration="b", vertex_map=reader).value
np.testing.assert_allclose(value, [[(-2+3j)*(1+4j)]])
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, str(archive)],
                            capture_output=True, text=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert result.returncode == 0, result.stdout + result.stderr
