"""Biomni A1 worker for natural-language retrospective design episodes."""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
from langgraph.errors import GraphRecursionError
from openai import APITimeoutError, InternalServerError, RateLimitError

from biosyn.agent.planner import BiomniPlanner
from biosyn.applications.worker import Episode
from biosyn.config import RuntimeConfig


class RetrospectiveEpisode(Episode):
    def __init__(self):
        super().__init__()
        self.gateway = SimpleNamespace(
            catalog=json.loads(Path('/public/catalog.json').read_text()),
            call=lambda name, arguments: self.rpc('tool', {'name': name, 'arguments': arguments}))
        self.state.task.request = self.task['request'] + (
            '\nRead /public/task.json for the complete protein, scaffold, truth table or pathway context. '
            'Design the requested artifact and submit it; no reference answer is provided.')

    def select_with_reserve(self, config):
        """Ask the configured model to select an existing artifact with a small context.

        Never pick a candidate implicitly, and never generate replacement designs here.
        """
        for attempt in (1, 2):
            state = self.summary()
            if state['final'] is not None:
                return
            if (not state['history'] or state['remaining_llm_requests'] < 1
                    or state['remaining_model_tokens'] < 4096 or state['remaining_wall_seconds'] < 10):
                raise RuntimeError('Cannot finalize: no candidate or insufficient reserved budget')
            candidates = [{
                'id': row['id'], 'evaluation': row['evaluation'], 'rationale': row.get('rationale', '')[:500]
            } for row in state['history']]
            messages = [
                {'role': 'system', 'content': 'Select one of the existing submitted designs. '
                    'Prefer structurally evaluable candidates, then compare the actual public feedback '
                    'against the requested objectives. Do not invent measurements or a new design. '
                    'Return only a JSON object with candidate_id and reason. The id must be from the list.'},
                {'role': 'user', 'content': json.dumps({'request': self.task['request'],
                    'objectives': self.task['inputs'].get('objectives', []), 'candidates': candidates},
                    ensure_ascii=False)}]
            self.store.event('reserved_selection_requested', attempt=attempt, messages=messages)
            # Final selection uses the remaining episode time, with no separate fixed request cap.
            timeout = max(1, state['remaining_wall_seconds'] - 5)
            with httpx.Client(transport=httpx.HTTPTransport(uds='/proxy/llm.sock'),
                              base_url='http://glm.local', timeout=timeout) as client:
                response = client.post('/v1/chat/completions', json={
                    'model': config.llm.model, 'messages': messages, 'temperature': 0,
                    'max_tokens': 2048, **config.llm.extra_body})
                response.raise_for_status()
                payload = response.json()
            self.store.event('reserved_selection_response', attempt=attempt, response=payload)
            text = (payload['choices'][0]['message'].get('content') or '').strip()
            text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text).strip()
            try:
                selection = json.loads(text)
                if not isinstance(selection, dict) or selection.get('candidate_id') not in {x['id'] for x in candidates}:
                    raise ValueError('Model did not select a submitted candidate id')
            except (ValueError, TypeError) as exc:
                self.store.event('reserved_selection_invalid', attempt=attempt, error=str(exc))
                continue
            self.rpc('finalize', {'candidate_id': selection['candidate_id'],
                                 'selection_source': 'reserved_llm_selection'})
            return
        raise RuntimeError('Model failed to select an existing candidate within the finalization reserve')

    def run(self):
        config = RuntimeConfig.model_validate(json.loads(Path('/public/config.json').read_text()))
        schema = {'type': 'object', 'required': ['id', 'parent_id', 'design', 'rationale'],
                  'properties': {'id': {'type': 'string'}, 'parent_id': {'type': ['string', 'null']},
                                 'design': {'type': 'object'}, 'rationale': {'type': 'string'}},
                  'additionalProperties': False}
        planner = BiomniPlanner(config, self, submission_schema=schema,
            submission_description='Submit a complete proposed design for public constraint checks. '
                'These checks do not measure expression, half-life, analog circuit behavior or yield.',
            instructions=(
                'You are BioSyn-Agent using Biomni A1. Solve the natural-language biological design request. '
                'Read the full /public/task.json and use its exact required artifact fields. '
                'Work in short executable steps; use Python for calculations instead of solving every detail '
                'in one long response. '
                f'The per-response output cap is {config.llm.max_tokens} tokens, including reasoning. '
                'Construct long sequences with Python rather than typing long nucleotide strings. '
                'Use supplied or verifiably retrieved sequences for established biological parts. '
                'If a required part sequence is unavailable, report the gap explicitly; do not invent '
                'a sequence and label it as a known biological part. '
                'Create complete sequences or directed graphs, not placeholders. You may use Python/BioPython '
                'in the REPL to construct long sequences reliably. Choose your own design and iteration strategy. '
                'Available MCP tools compute DNA statistics, translation and ViennaRNA secondary structure. '
                'Use actual returned results and clearly distinguish predictions from experimental measurements. '
                'The submit tool checks public constraints only; historical references and outcomes are hidden. '
                'Do not attempt to retrieve historical answers. There is no network access except the provided tools. '
                'Submit candidates using biosyn_submit(candidate={id,parent_id,design,rationale}). '
                'Use a unique id; parent_id is null or an earlier candidate id. Inspect feedback and revise as needed. '
                'Submit your first complete candidate before optional extensive optimization, so its artifact is saved. '
                'For circuits node inputs are predecessor node ids; INPUT node names and OUTPUT names must match '
                'the public task. A Boolean graph alone does not establish a DNA implementation. '
                'For pathways include the requested host connection, balanced reactions and yield assumptions '
                'as additional design fields; numerical claims require an explicit calculation. '
                'Tool feedback includes remaining budgets. If finalization_due is true, finish the current '
                'submission and select an existing candidate immediately. Reserve the last three LLM calls for selecting '
                'a submitted design with biosyn_finalize(candidate_id=...). You may finalize an imperfect design '
                'and explain unmet goals. Submitting a candidate does not finalize it. '
                'Do not claim uncomputed biological performance or wet-lab validation.'))
        recovered_errors = []
        try:
            for number in (1, 2):
                planner.run_round(number, stop_when=lambda: self.summary()['final'] is not None,
                                  stop_for_budget=lambda: self.summary()['finalization_due'])
                state = self.summary()
                if state['final'] is not None or state['finalization_due'] or state['history']:
                    break
        except (APITimeoutError, InternalServerError, RateLimitError, GraphRecursionError) as exc:
            event = {'type': type(exc).__name__, 'message': str(exc)[:1000]}
            recovered_errors.append(event)
            self.store.event('design_phase_failed', **event)
            if not self.summary()['history']:
                raise
        if self.summary()['final'] is None:
            self.select_with_reserve(config)
        self.store.write('episode-result.json', {**self.summary(), 'recovered_errors': recovered_errors})


if __name__ == '__main__':
    RetrospectiveEpisode().run()
