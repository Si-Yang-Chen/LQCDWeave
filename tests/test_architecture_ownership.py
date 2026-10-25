"""Responsibility gates beyond directory-level public API import checks."""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]/'src'/'lqcdweave'


class ArchitectureOwnershipTests(unittest.TestCase):
    def test_execution_cannot_compile_join_or_prepare_new_resource_plans(self):
        forbidden={'compile_plan','compile_plan_cached','compile_baryon_terms',
            'compile_baryon_terms_cached','join_storage_batches',
            'iter_join_storage_batches','prepare_coordinate_batch_workspace',
            'configure_cfg_output_layout'}
        violations=[]
        for path in (ROOT/'contraction'/'execution').rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                names=[]
                if isinstance(node,ast.ImportFrom): names=[a.name for a in node.names]
                elif isinstance(node,ast.Name): names=[node.id]
                elif isinstance(node,ast.Attribute): names=[node.attr]
                if forbidden.intersection(names):
                    violations.append((str(path.relative_to(ROOT)),node.lineno,sorted(forbidden.intersection(names))))
        self.assertEqual(violations,[], 'Public P.api access cannot bypass X pure execution semantics')

    def test_storage_read_view_has_no_workflow_or_planning_dependency(self):
        path=ROOT/'storage'/'dataset.py'
        tree=ast.parse(path.read_text())
        imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): imports += [x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom):
                name=(importlib.util.resolve_name('.'*node.level+(node.module or ''),'lqcdweave.storage')
                      if node.level else node.module or '')
                imports.append(name)
        self.assertFalse(any(x.startswith(('lqcdweave.planning','lqcdweave.runtime','lqcdweave.production','lqcdweave.contraction')) for x in imports))
        classes={x.name for x in tree.body if isinstance(x,ast.ClassDef)}
        self.assertIn('DatasetView',classes)
        workflow=ast.parse((ROOT/'runtime'/'datasets.py').read_text())
        admitted=next(x for x in workflow.body if isinstance(x,ast.ClassDef) and x.name=='DatasetView')
        self.assertEqual({x.name for x in admitted.body if isinstance(x,ast.FunctionDef)},{'__init__'})

    def test_retention_graph_decisions_are_not_in_workflow(self):
        workflow=ast.parse((ROOT/'runtime'/'project_datasets.py').read_text())
        function=next(x for x in workflow.body if isinstance(x,ast.FunctionDef) and x.name=='_build_project_retention_plan')
        names={x.id for x in ast.walk(function) if isinstance(x,ast.Name)}
        self.assertIn('plan_dataset_retention',names)
        self.assertFalse(names & {'parents_by_child','indegree','retained_ids','protected_by'})

    def test_workflow_output_orchestration_owns_no_numeric_codec_or_reduction(self):
        for name in ('correlator_output.py','correlator_dt_output.py'):
            tree=ast.parse((ROOT/'contraction'/name).read_text())
            names={x.id for x in ast.walk(tree) if isinstance(x,ast.Name)}
            definitions={x.name for x in tree.body if isinstance(x,(ast.ClassDef,ast.FunctionDef))}
            self.assertFalse(names & {'np','numpy','DatasetWriter','DatasetSpec'})
            self.assertFalse(definitions & {'_pack','_spec','_field_document','PackedDtWriter','_CfgWriter'})
