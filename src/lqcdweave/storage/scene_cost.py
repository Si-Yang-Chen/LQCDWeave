"""Persist compute-cost profiles and selected-plan reports as JSON documents."""

from dataclasses import asdict
import json
from pathlib import Path

from .dataset import write_json_document


def save_cost_prediction(path, prediction):
    """Save a plan cost report independently of execution results or checkpoints."""
    if prediction.get('format') != 'easy-distillation.scene-compute-cost-prediction':
        raise ValueError('expected a compute-cost plan report')
    write_json_document(path, prediction)


def load_cost_prediction(path):
    return _load(path, 'easy-distillation.scene-compute-cost-prediction')


def _load(path, expected_format):
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(document, dict) or document.get('format') != expected_format:
        raise ValueError('cost document format is invalid')
    return document


def read_cost_profile_document(path):
    """Read the complete profile document, including provenance."""
    return _load(path, 'easy-distillation.scene-compute-cost-profile')


def load_cost_profile(path):
    """Read profile fields; the caller constructs its CostProfile explicitly."""
    return read_cost_profile_document(path)['profile']


def save_cost_profile(path, profile, *, provenance=None):
    """Atomically save a CostProfile and optional provenance."""
    document = dict(format='easy-distillation.scene-compute-cost-profile', profile=asdict(profile))
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise TypeError('profile provenance must be a dict')
        document['provenance'] = provenance
    write_json_document(path, document)
