"""Full-time production, scientific bindings, recovery and measurement handoff."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

from production.elemental.test_elemental import fixture, reference_matrix
from lqcdweave.contracts.dataset import DatasetSpec
from lqcdweave.contracts.elemental import ElementalComponent, GaugePath
from lqcdweave.operators.single_hadron import Operator, OperatorTerm
from lqcdweave.production.elemental import ElementalGenerator
from lqcdweave.storage.dataset import DatasetReader, inspect_dataset, register_npy
from lqcdweave.production.elemental.planning import prepare_elemental_stage
from lqcdweave.contraction.meson_measurement import prepare_meson_stage, write_meson_rows
from lqcdweave.storage.dataset import DatasetWriter
from lqcdweave.runtime.dataset_stage import completed_dataset
from lqcdweave.adapters.production_dataset import DatasetFieldHandle
from lqcdweave.production.elemental.api import ElementalKernel as ElementalGenerator


def axis(name, values):
    return dict(name=name, coordinates=list(values))


class _ProductionDatasetFixture(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sizes, self.gauge, self.vectors, _ = fixture()
        lx,ly,lz,lt = self.sizes
        self.cfg = dict(ensemble="test", configuration="cfg", lattice=list(self.sizes))
        common = [axis("z",range(lz)), axis("y",range(ly)), axis("x",range(lx))]
        self.gspec = self.spec("gauge", [axis("time",[2,0,1]), *common,
            axis("direction",range(4)), axis("color_row",range(3)), axis("color_column",range(3))])
        self.bspec = self.spec("explicit_basis", [axis("time",[1,2,0]), axis("mode",[4,1,9]),
            *common, axis("color",range(3))])
        self.gid = self.save("gauge",self.gauge[[2,0,1]],self.gspec)
        self.bid = self.save("basis",self.vectors[[1,2,0]],self.bspec)
    def spec(self, kind, axes, **kwargs):
        d = dict(schema=1, kind=kind, configuration=self.cfg, axes=axes,
                 dtype="complex128", bases={}, definition={}, conventions={}, parents=[])
        d.update(kwargs)
        return DatasetSpec(d)
    def save(self, name, values, spec):
        path=self.root/(name+".npy")
        np.save(path,values)
        return register_npy(self.root/name,path,spec,provenance={"producer":"test"})
    def run_production(self, name="out", *, resume=False, **kwargs):
        with DatasetReader(self.root/"gauge") as gauge, DatasetReader(self.root/"basis") as basis:
            generator, spec, provenance, times, cfg = prepare_elemental_stage(gauge, basis, **kwargs)
            output = self.root/name
            if resume:
                existing = completed_dataset(output, spec, provenance)
                if existing is not None:
                    return existing
            with DatasetWriter(output, spec, provenance=provenance, resume=resume) as writer:
                if writer.missing:
                    generator.load(cfg["configuration"])
                for row in writer.missing:
                    writer.write(row, generator.calc_path_blocks(times[row])["v2v"])
                return writer.publish()
    def interrupt(self):
        original=ElementalGenerator.calc_path_blocks
        def calc(generator,t):
            if t==1:
                raise RuntimeError("interrupted")
            return original(generator,t)
        with patch.object(ElementalGenerator,"calc_path_blocks",calc):
            with self.assertRaises(RuntimeError):
                self.run_production(usedNe=2)
        output = self.root / "out"
        self.assertEqual(inspect_dataset(output, cfg="cfg")["state"], "partial")
        self.assertFalse((output / "cfg.npy").exists())
        self.assertFalse((output / "cfg.h5").exists())


class ProductionDatasetTests(_ProductionDatasetFixture):
    def test_all_times_reference_and_actual_parent_bindings(self):
        paths=((),(0,1,3,4))
        momenta=((0,0,0),(1,-1,1))
        identity=self.run_production(paths=paths,momentum_list=momenta,usedNe=2,ne_block_size=1)
        with DatasetReader(self.root/"out") as dataset:
            self.assertEqual(dataset.data_id,identity)
            self.assertEqual(dataset.spec.coordinates("time"),[0,1,2])
            doc=dataset.spec.document
            self.assertEqual(doc["parents"],[self.gid,self.bid])
            self.assertEqual(doc["definition"]["gauge_id"],self.gid)
            for name in ("sink_mode","source_mode"):
                self.assertEqual(doc["bases"][name],dict(data_id=self.bid,indices=[4,1]))
            for t in range(3):
                vectors=self.vectors[t,:2].reshape(2,-1)
                for pi,path in enumerate(paths):
                    for mi,momentum in enumerate(momenta):
                        reference=vectors.conj()@reference_matrix(self.gauge,path,t,momentum,phase="anchor")@vectors.T
                        np.testing.assert_allclose(dataset.read(time=t)[pi,mi],reference,atol=2e-12)
        self.assertEqual(self.run_production(resume=True,paths=paths,momentum_list=momenta,
                                             usedNe=2,ne_block_size=1),identity)

    def test_resume_skips_only_saved_times_and_rejects_changed_definition(self):
        self.interrupt()
        with self.assertRaises(ValueError):
            self.run_production(usedNe=1,resume=True)
        original=ElementalGenerator.calc_path_blocks
        calls=[]
        def calc(generator,t):
            calls.append(t)
            return original(generator,t)
        with patch.object(ElementalGenerator,"calc_path_blocks",calc):
            identity=self.run_production(usedNe=2,resume=True)
        self.assertEqual(calls,[1,2])
        clean_identity = self.run_production("clean",usedNe=2)
        self.assertNotEqual(identity, clean_identity)
        with DatasetReader(self.root/"out") as resumed, \
             DatasetReader(self.root/"clean") as clean:
            for time in resumed.spec.coordinates("time"):
                np.testing.assert_array_equal(resumed.read(time=time), clean.read(time=time))

    def test_completed_block_layout_mismatch_rejected_before_calculation(self):
        self.interrupt()
        path=self.root/"out"/".state"/"cfg.npy"
        np.save(path, np.ones(1, dtype=np.complex128), allow_pickle=False)
        self.assertEqual(inspect_dataset(self.root/"out", cfg="cfg")["state"], "partial")
        self.assertFalse((self.root/"out"/"cfg.npy").exists())
        with patch.object(ElementalGenerator,"calc_path_blocks") as calc:
            with self.assertRaisesRegex(ValueError,"shape/dtype"):
                self.run_production(usedNe=2,resume=True)
            calc.assert_not_called()
        self.assertEqual(inspect_dataset(self.root/"out", cfg="cfg")["state"], "partial")
        self.assertFalse((self.root/"out"/"cfg.npy").exists())
        self.assertFalse((self.root/"out"/"cfg.h5").exists())

    def test_configuration_and_layout_rejected_before_output(self):
        d=self.bspec.document
        d["configuration"]["configuration"]="other"
        self.save("wrong",self.vectors[[1,2,0]],DatasetSpec(d))
        with self.assertRaisesRegex(ValueError,"mismatch"):
            with DatasetReader(self.root/"gauge") as gauge, DatasetReader(self.root/"wrong") as basis:
                prepare_elemental_stage(gauge, basis)
        self.assertFalse((self.root/"out").exists())
        with self.assertRaisesRegex(ValueError,"spatial"):
            self.run_production(paths=[(6,)])
        self.assertFalse((self.root/"out").exists())

    def test_time_first_output_consumed_by_existing_meson_workflow(self):
        self.run_production(usedNe=2)
        with DatasetReader(self.root/"out") as dataset:
            doc=dataset.spec.document
            data=np.stack([dataset.read(time=t).copy() for t in range(3)],axis=2)
            doc["axes"]=doc["axes"][1:3]+doc["axes"][:1]+doc["axes"][3:]
        self.save("canonical",data,DatasetSpec(doc))
        mode=dict(data_id=self.bid,indices=[4,1])
        spec=self.spec("vsv",[axis("sink_time",range(3)),axis("source_time",range(3)),
            axis("sink_spin",range(4)),axis("source_spin",range(4)),
            axis("sink_mode",[4,1]),axis("source_mode",[4,1])],
            bases=dict(sink_mode=mode,source_mode=mode),parents=[self.bid,self.gid],
            definition=dict(gauge_id=self.gid,dirac=dict(action="test",mass=.1),solver_tolerance=1e-12),
            conventions=dict(normalization="unscaled",spin_basis="degrand_rossi",temporal_boundary="periodic"))
        rng=np.random.default_rng(34)
        self.save("vsv",rng.normal(size=spec.shape)+1j*rng.normal(size=spec.shape),spec)
        operator=Operator.from_terms("local",[OperatorTerm(15,ElementalComponent.path(GaugePath()),0)])
        for name in ("out","canonical"):
            with DatasetReader(self.root/"vsv") as prop, DatasetReader(self.root/name) as elemental:
                inputs = (prop, prop, elemental, elemental)
                spec, _, _, _ = prepare_meson_stage(inputs, operator, operator)
                with DatasetWriter(self.root/(name+"-corr"), spec, provenance={"producer": "fixture"}) as writer:
                    write_meson_rows(inputs, operator, operator, writer)
                    writer.publish()
        with DatasetReader(self.root/"out-corr") as a, DatasetReader(self.root/"canonical-corr") as b:
            for t in range(3):
                np.testing.assert_allclose(a.read(source_time=t),b.read(source_time=t))
