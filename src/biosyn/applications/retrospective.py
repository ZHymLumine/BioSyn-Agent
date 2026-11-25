"""Run Biomni on public retrospective cases; compare hidden references only after finalization."""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import jsonschema
from dotenv import dotenv_values

from biosyn.agent.state import RunStore
from biosyn.applications.runner import ledger_server
from biosyn.benchmarks.proxy import chat_proxy
from biosyn.benchmarks.runner import command, write
from biosyn.config import ServerConfig
from biosyn.mcp.client import ToolGateway

MODEL = 'zai-org/GLM-5.3-Flash'
BUDGET = {'llm_requests': 32, 'candidate_evaluations': 12, 'mcp_calls': 36,
          'model_tokens': 240000, 'max_output_tokens': 8192, 'wall_seconds': 5400,
          'finalization_call_reserve': 3, 'finalization_token_reserve': 48000,
          'finalization_wall_reserve': 1800}
PROXY_TIMEOUT = 1200
CLIENT_TIMEOUT = 1260


def comparator(root):
    # This host-only import is not mounted in the agent namespace.
    sys.path.insert(0, str(root / 'scripts'))
    spec = importlib.util.spec_from_file_location('retrospective_comparator',
                                                 root / 'scripts/evaluate_retrospective_design.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare


class RetrospectiveLedger:
    def __init__(self, task, store, gateway, compare, budget):
        self.task, self.store, self.gateway, self.compare = task, store, gateway, compare
        self.budget = budget
        self.history, self.final = [], None
        self.lock = threading.RLock()
        self.llm_usage = {'calls': 0, 'total_tokens': 0}
        self.started_at = time.monotonic()

    def state(self):
        remaining_calls = max(0, self.budget['llm_requests'] - self.llm_usage['calls'])
        remaining_tokens = max(0, self.budget['model_tokens'] - self.llm_usage['total_tokens'])
        remaining_seconds = max(0, self.budget['wall_seconds'] - (time.monotonic() - self.started_at))
        reserve = (remaining_calls <= self.budget['finalization_call_reserve']
                   or remaining_tokens <= self.budget['finalization_token_reserve']
                   or remaining_seconds <= self.budget['finalization_wall_reserve'])
        return {'history': [{'id': row['candidate']['id'], 'evaluation': row['evaluation'],
                            'rationale': row['candidate']['rationale'][:500]}
                            for row in self.history], 'final': self.final,
                'remaining_evaluations': self.budget['candidate_evaluations'] - len(self.history),
                'remaining_mcp_calls': self.budget['mcp_calls'] - self.gateway.calls,
                'remaining_llm_requests': remaining_calls, 'remaining_model_tokens': remaining_tokens,
                'remaining_wall_seconds': round(remaining_seconds, 1), 'finalization_due': reserve}

    def dispatch(self, action, payload):
        with self.lock:
            if action == 'state':
                return self.state()
            if self.final is not None:
                raise ValueError('Episode already finalized')
            if action == 'tool':
                return self.gateway.call(payload['name'], payload['arguments'])
            if action == 'submit':
                return self.submit(payload['candidate'])
            if action == 'finalize':
                row = next((r for r in self.history if r['candidate']['id'] == payload['candidate_id']), None)
                if row is None:
                    raise ValueError('Select an existing submitted candidate')
                self.final = {'candidate_id': row['candidate']['id'], 'design_sha256': row['design_sha256']}
                self.final['selection_source'] = payload.get('selection_source', 'biomni_tool')
                self.store.write('final.json', row)
                self.store.write('ledger.json', self.state())
                return self.final
            raise ValueError('Unknown operation')

    def submit(self, candidate):
        if len(self.history) >= self.budget['candidate_evaluations']:
            raise ValueError('Candidate evaluation budget exhausted')
        if not isinstance(candidate, dict) or set(candidate) != {'id', 'parent_id', 'design', 'rationale'}:
            raise ValueError('Candidate requires exactly id, parent_id, design, rationale')
        cid = candidate['id']
        if not isinstance(cid, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,59}', cid):
            raise ValueError('Invalid candidate id')
        ids = {row['candidate']['id'] for row in self.history}
        if cid in ids or (candidate['parent_id'] is not None and candidate['parent_id'] not in ids):
            raise ValueError('Duplicate id or unknown parent')
        if not isinstance(candidate['design'], dict) or not isinstance(candidate['rationale'], str):
            raise ValueError('design must be an object; rationale must be a string')
        candidate = json.loads(json.dumps(candidate, allow_nan=False))
        try:
            checked = self.compare(self.task, {'references': []}, candidate['design'])
            evaluation = {'structurally_evaluable': True, 'checks': checked['structural_checks'],
                          'scope': 'Public constraints only; biological performance is not measured.'}
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            evaluation = {'structurally_evaluable': False, 'error': f'{type(exc).__name__}: {exc}',
                          'output_contract': self.task.get('expected_output', {})}
        row = {'candidate': candidate, 'evaluation': evaluation,
               'design_sha256': hashlib.sha256(json.dumps(candidate['design'], sort_keys=True).encode()).hexdigest()}
        self.history.append(row)
        self.store.write(f'candidate-{cid}.json', row)
        self.store.write('ledger.json', self.state())
        self.store.event('public_feedback', **row)
        return {'candidate_id': cid, **evaluation, 'budget': {
            k: v for k, v in self.state().items() if k.startswith('remaining_') or k == 'finalization_due'}}


def llm_config():
    return {'servers': {}, 'biomni_tool_modules': [], 'use_tool_retriever': True,
            'recursion_limit': 100, 'code_timeout': 180,
            'llm': {'model': MODEL, 'base_url': 'http://glm.local/v1',
                    'api_socket': '/proxy/llm.sock', 'api_key_env': 'BIOSYN_PROXY_KEY',
                    'temperature': 0.2, 'max_tokens': BUDGET['max_output_tokens'], 'timeout': CLIENT_TIMEOUT,
                    'extra_body': {'chat_template_kwargs': {'reasoning_effort': 'low', 'clear_thinking': True}}}}


def copy_public_resources(task, dataset, public):
    """Copy only hash-bound files within the dataset's public/resources directory."""
    allowed = (dataset / 'public/resources').resolve()
    receipts = []
    for resource in task['inputs'].get('public_resources', []):
        relative = Path(resource['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[:1] != ('resources',):
            raise ValueError('Public resource path must be within resources/')
        source = (dataset / 'public' / relative).resolve()
        if not source.is_relative_to(allowed) or not source.is_file():
            raise ValueError('Public resource escapes allowed directory or is missing')
        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != resource['sha256']:
            raise ValueError(f'Public resource hash mismatch: {relative}')
        target = public / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle:
            handle.write(content)
        receipts.append({'path': relative.as_posix(), 'sha256': digest, 'bytes': len(content)})
    return receipts


def run_case(root, dataset, destination, row, endpoint, key, compare, budget):
    case = destination / row['case_id']
    case.mkdir(exist_ok=False)
    public, work = case / 'public', case / 'work'
    public.mkdir()
    work.mkdir()
    source = dataset / row['public_path']
    task = json.loads(source.read_text())
    jsonschema.validate(task, json.loads((dataset / 'public_case.schema.json').read_text()))
    resource_receipts = copy_public_resources(task, dataset, public)
    # The older worker's small Task wrapper uses id; the source schema stays unchanged.
    write(public / 'task.json', {**task, 'id': task['case_id']})
    write(public / 'config.json', llm_config())
    store = RunStore(case / 'ledger')
    settings = {name: ServerConfig(command=[str(root / '.venv-biomni/bin/python'), '-m', f'biosyn.mcp.{name}'],
                                   env={'PYTHONPATH': str(root / 'src'), 'OMP_NUM_THREADS': '1'})
                for name in ('dna', 'rna')}
    start = time.monotonic()
    result = {'case_id': task['case_id'], 'track': task['track'], 'model': MODEL,
              'public_task_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'candidate_experimental_outcome': 'not_measured', 'public_resources': resource_receipts}
    with ToolGateway(settings, store, max_calls=budget['mcp_calls']) as gateway:
        write(public / 'catalog.json', gateway.catalog)
        ledger = RetrospectiveLedger(task, store, gateway, compare, budget)
        with tempfile.TemporaryDirectory(prefix='biosyn-retro-') as temp:
            llm_sock, eval_sock = Path(temp) / 'llm.sock', Path(temp) / 'eval.sock'
            with chat_proxy(llm_sock, key, case / 'api-usage.jsonl',
                            max_calls=budget['llm_requests'], max_tokens=budget['max_output_tokens'],
                            max_total_tokens=budget['model_tokens'], model=MODEL,
                            endpoint=endpoint['base_url'].rstrip('/') + '/chat/completions', timeout=PROXY_TIMEOUT) as usage, \
                    ledger_server(eval_sock, ledger):
                ledger.llm_usage = usage
                cmd = command(root, public, work, llm_sock)
                cmd[-1] = 'biosyn.applications.retrospective_worker'
                cmd[1:1] = ['--ro-bind', str(eval_sock), '/proxy/eval.sock']
                write(case / 'execution.json', {'command': cmd, 'budget': budget,
                      'private_reference_access': 'not mounted; graded only by host after episode',
                      'token_limit': 'Next request rejected after cumulative reported usage reaches limit.'})
                with (case / 'agent.log').open('w') as log:
                    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                            env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'})
                    try:
                        result['process_returncode'] = proc.wait(timeout=budget['wall_seconds'])
                        result['timeout'] = False
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        result['process_returncode'] = proc.wait()
                        result['timeout'] = True
                result.update(api_usage=dict(usage), final=ledger.final, candidate_count=len(ledger.history),
                              mcp_calls=gateway.calls)
                episode_path = work / 'output/episode-result.json'
                if episode_path.exists():
                    episode = json.loads(episode_path.read_text())
                    result['recovered_errors'] = episode.get('recovered_errors', [])
                api_log = case / 'api-usage.jsonl'
                calls = [json.loads(line) for line in api_log.read_text().splitlines()] if api_log.exists() else []
                result['successful_llm_calls'] = sum(c.get('http_status') == 200 and c.get('model') == MODEL
                                                     for c in calls)
        if ledger.final:
            selected = json.loads((store.path / 'final.json').read_text())
            design = selected['candidate']['design']
            if hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest() != ledger.final['design_sha256']:
                raise ValueError('Final design hash differs from host ledger')
            write(case / 'submission.json', design)
            private = json.loads((dataset / row['reference_path']).read_text())
            try:
                grade = compare(task, private, design)
                write(case / 'comparison.json', grade)
                result['comparison_status'] = 'computed'
                result['structural_checks'] = grade['structural_checks']
            except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
                result['comparison_status'] = 'invalid_submission'
                result['comparison_error'] = f'{type(exc).__name__}: {exc}'
        else:
            result['comparison_status'] = 'not_finalized'
    result['elapsed_seconds'] = round(time.monotonic() - start, 2)
    write(case / 'result.json', result)
    return result


def workflow_complete(result):
    """Execution completion, not a biological-performance success criterion."""
    return bool(result.get('successful_llm_calls') and result.get('final')
                and result.get('process_returncode') == 0 and not result.get('timeout')
                and result.get('comparison_status') == 'computed')


def endpoint_check(endpoint, key):
    if endpoint['model'] != MODEL:
        raise ValueError('Requested GLM-5.3-Flash; refusing another model')
    with httpx.Client(headers={'Authorization': 'Bearer ' + key}, timeout=600) as client:
        models = client.get(endpoint['base_url'].rstrip('/') + '/models')
        models.raise_for_status()
        if MODEL not in {m['id'] for m in models.json()['data']}:
            raise ValueError('Endpoint does not expose the requested model')
        r = client.post(endpoint['base_url'].rstrip('/') + '/chat/completions', json={
            'model': MODEL, 'messages': [{'role': 'user', 'content': 'Reply with the single word READY.'}],
            'max_tokens': 2048, 'temperature': 0,
            'chat_template_kwargs': {'reasoning_effort': 'low', 'clear_thinking': True}})
        r.raise_for_status()
        reply = r.json()
        if reply.get('model') != MODEL or not reply['choices'][0]['message'].get('content'):
            raise ValueError('GLM completion/model verification failed')
        return reply


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--dataset', type=Path, help='Versioned retrospective dataset directory')
    p.add_argument('--endpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--case-ids', nargs='+')
    p.add_argument('--workers', type=int, choices=range(1, 9), default=2)
    args = p.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(exist_ok=False, parents=True)
    dataset = args.dataset.resolve() if args.dataset else root / 'tasks/application_suite_v3/retrospective'
    manifest = json.loads((dataset / 'manifest.json').read_text())
    rows = manifest['case_index']
    if args.case_ids:
        rows = [r for r in rows if r['case_id'] in args.case_ids]
        if {r['case_id'] for r in rows} != set(args.case_ids):
            raise ValueError('Unknown case id')
    endpoint = json.loads(args.endpoint.read_text())
    key = dotenv_values(endpoint['env_file'])['BIOSYN_GLM_API_KEY']
    write(output / 'inference-test.json', endpoint_check(endpoint, key))
    write(output / 'run-config.json', {'model': MODEL, 'endpoint': endpoint['base_url'],
          'dataset_path': str(dataset), 'dataset_version': manifest['dataset_version'], 'budget': BUDGET, 'workers': args.workers,
          'case_ids': [r['case_id'] for r in rows], 'llm_config': llm_config(),
          'dataset_manifest_sha256': hashlib.sha256((dataset / 'manifest.json').read_bytes()).hexdigest(),
          'scope': 'Development evaluation; formal checks/reference similarity, not wet-lab success.'})
    files = ['src/biosyn/applications/retrospective.py', 'src/biosyn/applications/retrospective_worker.py',
             'src/biosyn/agent/planner.py', 'src/biosyn/benchmarks/proxy.py',
             'Biomni/biomni/agent/a1.py',
             'scripts/evaluate_retrospective_design.py']
    write(output / 'code-hashes.json', {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                        for name in files})
    compare = comparator(root)
    done = threading.Event()
    heartbeat = args.endpoint.parent / 'heartbeat'
    def keepalive():
        while not done.is_set():
            heartbeat.touch()
            done.wait(30)
    thread = threading.Thread(target=keepalive, daemon=True)
    thread.start()
    results = []
    try:
        # Deterministic one-case-per-track pilot before spending on the remaining cases.
        first = {}
        for row in sorted(rows, key=lambda r: r['case_id']):
            first.setdefault(row['track'], row)
        first_ids = {r['case_id'] for r in first.values()}
        stages = [list(first.values()), [r for r in rows if r['case_id'] not in first_ids]]
        for stage in stages:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(run_case, root, dataset, output, row, endpoint, key, compare, BUDGET): row
                           for row in stage}
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        # A failed episode is recorded, never replaced by a mock/alternative model.
                        result = {'case_id': row['case_id'], 'track': row['track'],
                                  'comparison_status': 'execution_error', 'error': f'{type(exc).__name__}: {exc}'}
                        write(output / row['case_id'] / 'result.json', result)
                    results.append(result)
                    write(output / 'summary.json', {'model': MODEL, 'planned': len(rows), 'completed': len(results),
                          'results': sorted(results, key=lambda r: r['case_id'])})
                    print(json.dumps(result), flush=True)
            failed = [r['case_id'] for r in results if not workflow_complete(r)]
            write(output / 'workflow-check.json', {'complete': not failed, 'incomplete_cases': failed,
                'scope': 'Final artifact plus comparison; no claim of biological success.'})
            if failed:
                raise RuntimeError(f'Incomplete pilot workflows: {failed}; remaining cases were not started.')
    finally:
        done.set()
        thread.join(timeout=5)


if __name__ == '__main__':
    main()
