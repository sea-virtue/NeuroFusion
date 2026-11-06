"""Locate pinned checkouts without installing their complete robot stacks."""
import ast
import importlib
import os
from pathlib import Path
import sys


def project_root():
    override = os.environ.get('NEUROFUSION_ROOT')
    if override:
        return Path(override).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / 'third_party' / 'sources.lock.json').is_file():
            return parent
    raise FileNotFoundError('Use the source checkout or set NEUROFUSION_ROOT')


def upstream_path(name):
    target = project_root() / 'third_party' / name
    if not (target / '.git').exists():
        raise FileNotFoundError(f'Missing official checkout: {target}; see third_party/README.md')
    return target


def import_upstream(repository, module):
    folder = str(upstream_path(repository))
    if folder not in sys.path:
        sys.path.insert(0, folder)
    imported = importlib.import_module(module)
    if not Path(imported.__file__).resolve().is_relative_to(Path(folder)):
        raise ImportError(f'{module} resolves outside the pinned {repository} checkout')
    return imported


def act_definitions(relative, namespace, names=None):
    """Execute upstream definitions verbatim, omitting unrelated import-time baggage.

    ACT's original module imports camera/DETR utilities and IPython even for its
    tiny mlp() helper. AST selection keeps those dependencies out; no network
    definition is rewritten. Original files and licenses remain in third_party.
    """
    source = upstream_path('act') / relative
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and (names is None or node.name in names)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace
