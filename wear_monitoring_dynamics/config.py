"""Configuration, portable paths, and content-addressed provenance."""
import copy
import hashlib
import json
from pathlib import Path

import yaml

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_hash():
    paths = sorted(PACKAGE.rglob('*.py'))
    paths.append(PROJECT / 'release/CondWear-ODE/models/vb_sensor_condition_ode.py')
    return digest({str(p.relative_to(PROJECT)): file_hash(p) for p in paths})


def merge(base, updates):
    result = copy.deepcopy(base)
    for key, value in updates.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def load_config(path=None):
    cfg = yaml.safe_load((PACKAGE / 'configs/default.yaml').read_text(encoding='utf-8'))
    if path:
        cfg = merge(cfg, yaml.safe_load(Path(path).read_text(encoding='utf-8')))
    return cfg


def project_path(path):
    p = Path(path)
    return p if p.is_absolute() else PROJECT / p


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)
