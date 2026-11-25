import json
import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from biosyn.agent.state import RunStore
from biosyn.agent.planner import BiomniPlanner, MAX_CALLS_WITHOUT_ARTIFACT_PER_ROUND
from biosyn.applications.comparison import ScientificLedger, select_fixed
from biosyn.applications.comparison_protocol import unpack_design, run_construction, run_design_rounds
from biosyn.applications.comparison_sandbox import prepare_source, upstream_biomni_command
from biosyn.applications.computational_circuit import validate_truth_table
from biosyn.applications.retrospective import BUDGET


@pytest.mark.parametrize('wrapped', [True, False])
def test_output_contract_preserves_biological_fields(wrapped):
    design = {'cds': 'ATGTAA', 'extra_annotation': 'unaltered'}
    value = dict(id='x', parent_id=None, design=design, rationale='example') if wrapped else design
    parsed, form = unpack_design(value, ['cds'])
    assert parsed == design
    assert form == ('candidate_envelope' if wrapped else 'bare_design')


@pytest.mark.parametrize('value', [{'selected': {'cds': 'ATGTAA'}}, {'design': {'cds': 'ATGTAA'}}, [], {'cds': 'ATGTAA', 'design': {}}])
def test_ambiguous_or_unsupported_wrappers_are_not_repaired(value):
    if isinstance(value, dict) and 'cds' in value:
        # A bare schema object remains unchanged; no nested fields are preferred.
        assert unpack_design(value, ['cds'])[0] == value
    else:
        with pytest.raises(ValueError):
            unpack_design(value, ['cds'])


def test_plan_only_completion_continues_until_artifact(tmp_path):
    state = {'final': None, 'history': [], 'remaining_evaluations': 12,
             'remaining_llm_requests': 32, 'finalization_due': False}
    rounds = []
    def run(number, **kwargs):
        rounds.append(number)
        state['remaining_llm_requests'] -= 1
        if number == 2:
            state['history'].append({'id': 'actual-artifact'})
    episode = SimpleNamespace(summary=lambda: state, store=RunStore(tmp_path / 'store'))
    run_design_rounds(episode, SimpleNamespace(run_round=run), 32)
    assert rounds == [1, 2]
    assert 'no_artifact_after_round' in (tmp_path / 'store/events.jsonl').read_text()


def test_plan_only_cannot_spend_finalization_reserve(tmp_path):
    state = {'final': None, 'history': [], 'remaining_evaluations': 12,
             'remaining_llm_requests': 4, 'finalization_due': False}
    def run(number, **kwargs):
        state.update(remaining_llm_requests=3, finalization_due=True)
    episode = SimpleNamespace(summary=lambda: state, store=RunStore(tmp_path / 'store'))
    with pytest.raises(RuntimeError, match='No candidate'):
        run_design_rounds(episode, SimpleNamespace(run_round=run), 32)
    assert state['remaining_llm_requests'] == 3


def test_planner_stops_resource_loop_and_adds_submission_recovery():
    class Store:
        def __init__(self):
            self.events = []
            self.writes = {}

        def event(self, kind, **payload):
            self.events.append({'kind': kind, **payload})

        def write(self, name, value):
            self.writes[name] = value

    class Task:
        @staticmethod
        def model_dump_json(indent=2):
            return '{"id": "example"}'

    class Runtime:
        def __init__(self):
            self.remaining = 32
            self.store = Store()
            self.state = SimpleNamespace(task=Task(), run_id='watchdog-test')

        def summary(self):
            return {'final': None, 'history': [], 'remaining_evaluations': 12,
                    'remaining_llm_requests': self.remaining, 'finalization_due': False}

    class Message:
        def __init__(self, content):
            self.type = 'ai'
            self.content = content
            self.usage_metadata = None
            self.response_metadata = {}

    class LoopingApp:
        def __init__(self, runtime, calls):
            self.runtime = runtime
            self.calls = calls
            self.prompt = None

        def stream(self, value, **kwargs):
            self.prompt = value['messages'][0].content
            for index in range(self.calls):
                self.runtime.remaining -= 1
                action = Message(f'<execute>print({index})</execute>')
                yield {'messages': [action], 'next_step': 'execute'}
                observation = Message(f'<observation>{index}</observation>')
                yield {'messages': [observation], 'next_step': None}

    runtime = Runtime()
    planner = object.__new__(BiomniPlanner)
    planner.runtime = runtime
    planner.config = SimpleNamespace(recursion_limit=200)
    planner.instructions = 'Submit a design.'
    first_app = LoopingApp(runtime, calls=20)
    planner.agent = SimpleNamespace(use_tool_retriever=False, app=first_app)

    planner.run_round(1)

    assert runtime.remaining == 32 - MAX_CALLS_WITHOUT_ARTIFACT_PER_ROUND
    assert {'kind': 'planner_stopped', 'reason': 'no_artifact_call_limit',
            'round': 1, 'calls_used': MAX_CALLS_WITHOUT_ARTIFACT_PER_ROUND} in runtime.store.events

    second_app = LoopingApp(runtime, calls=1)
    planner.agent.app = second_app
    planner.run_round(2)
    assert 'SUBMISSION RECOVERY' in second_app.prompt
    assert 'call biosyn_submit in this round' in second_app.prompt


def test_no_goal_feedback_masks_submit_and_state_but_preserves_host_evidence(tmp_path):
    def call(*args):
        gateway.calls += 1
        return {'call_id': 'c', 'data': {'valid': True, 'feasible': True, 'metrics': {'secret_goal': 987},
                                       'model_function_satisfied': True, 'artifact': {'x': 1}}}
    gateway = SimpleNamespace(calls=0, call=call)
    ledger = ScientificLedger({}, RunStore(tmp_path / 'store'), gateway, BUDGET, goal_feedback=False)
    feedback = ledger.dispatch('submit', {'candidate': {'id': 'c', 'parent_id': None,
                                                       'design': {'x': 1}, 'rationale': 'reason'}})
    assert feedback['valid'] is True
    assert 'secret_goal' not in json.dumps(feedback)
    assert 'feasible' not in json.dumps(ledger.dispatch('state', {}))
    assert ledger.history[0]['full_evaluation']['metrics']['secret_goal'] == 987
    with pytest.raises(ValueError, match='approved'):
        ledger.dispatch('tool', {'name': 'science.new_goal_predictor', 'arguments': {}})
    with pytest.raises(ValueError, match='biosyn_submit'):
        ledger.dispatch('tool', {'name': 'science.evaluate_design', 'arguments': {}})
    ledger.dispatch('finalize', {'candidate_id': 'c'})
    assert json.loads((tmp_path / 'store/final.json').read_text())['evaluation']['metrics']['secret_goal'] == 987


def test_numerical_noise_does_not_outweigh_discrete_complexity():
    rows = [{'candidate': {'id': str(i)}, 'evaluation': {'valid': True, 'metrics': {
        'product_yield': y, 'gene_count': n}}} for i, (y, n) in enumerate([
            (1.1443435036136902, 3), (1.1443435036136755, 1)])]
    assert select_fixed(rows, [{'metric': 'product_yield', 'direction': 'maximize'},
                               {'metric': 'gene_count', 'direction': 'minimize'}]) is rows[1]


def test_all_requested_circuit_outputs_must_be_supported():
    rows = [{'inputs': {'A': i}, 'outputs': {'output_YFP': i, 'output_RFP': 1-i}} for i in (0, 1)]
    with pytest.raises(ValueError, match='multi-output'):
        validate_truth_table(rows)


def test_truth_table_requires_every_input_state():
    rows = [{'inputs': {'A': 0, 'B': i}, 'outputs': {'output_YFP': i}} for i in (0, 1)]
    with pytest.raises(ValueError, match='every input state'):
        validate_truth_table(rows)


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process-group cleanup integration')
def test_construction_timeout_kills_shell_children(tmp_path):
    marker = tmp_path / 'late-write'
    code = ('import subprocess,time; subprocess.Popen([' + repr(sys.executable) + ", '-c', " +
            repr('import time; from pathlib import Path; time.sleep(1); Path(' + repr(str(marker)) + ').touch()') +
            ']); time.sleep(20)')
    with (tmp_path / 'log').open('w') as log:
        with pytest.raises(subprocess.TimeoutExpired):
            run_construction([sys.executable, '-c', code], tmp_path, log, .25)
    import time
    time.sleep(1.2)
    assert not marker.exists()


def test_native_biomni_source_hides_biosyn_scientific_helpers(tmp_path):
    root = Path(__file__).resolve().parents[2]
    destination = tmp_path / 'agent-source'
    prepare_source(root, destination, expose_biosyn_helpers=False)
    circuit = (destination / 'biosyn/applications/computational_circuit.py').read_text()
    assert 'native Biomni baseline' in circuit
    assert 'synthesize_nor' not in circuit
    for relative in (
        'biosyn/applications/science.py',
        'biosyn/applications/computational_sequence.py',
        'biosyn/applications/computational_metabolic.py',
        'biosyn/applications/computational_regulatory.py',
        'biosyn/mcp/computational.py',
    ):
        assert 'Scientific evaluation is host-only' in (destination / relative).read_text()


def test_upstream_biomni_worker_has_no_biosyn_imports_or_evaluator_socket():
    path = Path(__file__).resolve().parents[2] / 'src/biosyn/applications/biomni_upstream_worker.py'
    source = path.read_text()
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name == 'biosyn' or name.startswith('biosyn.') for name in imported)
    assert '/proxy/eval.sock' not in source
    assert 'biosyn_submit' not in source


def test_upstream_biomni_command_hides_biosyn_source_and_feedback(tmp_path):
    root = tmp_path / 'project'
    public, work = tmp_path / 'public', tmp_path / 'work'
    socket, runner = tmp_path / 'llm.sock', tmp_path / 'worker.py'
    for path in (root / 'src', root / 'Biomni', root / '.venv-biomni', public, work):
        path.mkdir(parents=True, exist_ok=True)
    socket.touch(); runner.write_text('print("worker")\n')
    cmd = upstream_biomni_command(root, public, work, socket, runner)
    assert str(root / 'src') not in cmd
    assert '/proxy/eval.sock' not in cmd
    assert '/runner/worker.py' in cmd
    assert cmd[-2:] == [str(root / '.venv-biomni/bin/python'), '/runner/worker.py']
