"""Keep provided scientific scorers behind the counted host MCP interface."""
import ast
import hashlib
import json
import shutil
from pathlib import Path

from biosyn.benchmarks.runner import command as base_command

HOST_ONLY = (
    'applications/science.py',
    'applications/computational_sequence.py',
    'applications/computational_metabolic.py',
    'applications/computational_regulatory.py',
    'mcp/computational.py',
)


def prepare_source(root, destination, *, expose_biosyn_helpers=True):
    source = Path(root) / 'src'
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
    for relative in HOST_ONLY:
        (destination / 'biosyn' / relative).write_text(
            'raise ImportError("Scientific evaluation is host-only; submit candidates through the counted MCP interface")\n')
    circuit = destination / 'biosyn/applications/computational_circuit.py'
    if expose_biosyn_helpers:
        # Preserve the two existing deterministic topology functions verbatim, including imports.
        # The physical response evaluator is the only class in this explicitly checked module.
        original = circuit.read_text();module = ast.parse(original)
        classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
        if len(classes) != 1 or classes[0].name != 'CircuitEvaluator':
            raise ValueError('Circuit module changed; review worker-visible helpers before proceeding')
        cls = classes[0];lines = original.splitlines(keepends=True)
        circuit.write_text(''.join(lines[:cls.lineno-1]+lines[cls.end_lineno:]))
    else:
        circuit.write_text(
            'raise ImportError("BioSyn circuit helpers are unavailable in the native Biomni baseline")\n'
        )
    files = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(destination.rglob('*.py'))}
    (destination.parent / 'agent-source-hashes.json').write_text(json.dumps(files, indent=2)+'\n')


def command(root, public, work, socket_path, agent_source):
    cmd = base_command(root, public, work, socket_path)
    target = str(Path(root) / 'src')
    matches = [i for i in range(len(cmd)-2)
               if cmd[i:i+3] == ['--ro-bind',target,target]]
    if len(matches) != 1:
        raise ValueError('Expected exactly one worker source mount')
    cmd[matches[0]+1] = str(agent_source)
    return cmd


def upstream_biomni_command(root, public, work, socket_path, runner):
    """Run Biomni with no BioSyn source tree or evaluation socket in its namespace."""
    cmd = base_command(root, public, work, socket_path)
    target = str(Path(root) / 'src')
    matches = [i for i in range(len(cmd)-2)
               if cmd[i:i+3] == ['--ro-bind', target, target]]
    if len(matches) != 1:
        raise ValueError('Expected exactly one worker source mount')
    del cmd[matches[0]:matches[0]+3]
    proc = cmd.index('--proc')
    cmd[proc:proc] = ['--dir', '/runner', '--ro-bind', str(runner), '/runner/worker.py']
    if cmd[-2:] != ['-m', 'biosyn.benchmarks.worker']:
        raise ValueError('Unexpected benchmark worker command')
    cmd[-2:] = ['/runner/worker.py']
    if target in cmd or '/proxy/eval.sock' in cmd:
        raise ValueError('Upstream Biomni isolation still exposes BioSyn source or evaluation')
    return cmd
