"""Explicit artifact contracts and bounded execution shared by comparison methods."""

import os
import signal
import subprocess


CANDIDATE_FIELDS = {"id", "parent_id", "design", "rationale"}


def unpack_design(value, required_fields):
    """Accept a bare design or the documented candidate envelope, never guess wrappers."""
    if not isinstance(value, dict):
        raise ValueError("Expected a design object or complete candidate envelope")
    envelope = set(value) == CANDIDATE_FIELDS
    if envelope:
        if not isinstance(value["id"], str) or not isinstance(value["rationale"], str):
            raise ValueError("Candidate id and rationale must be strings")
        if value["parent_id"] is not None and not isinstance(value["parent_id"], str):
            raise ValueError("Invalid parent_id")
        design = value["design"]
    else:
        design = value
    if not isinstance(design, dict) or not set(required_fields) <= design.keys():
        raise ValueError("Design is missing decision_schema fields: " + ", ".join(required_fields))
    return design, "candidate_envelope" if envelope else "bare_design"


def run_construction(command, cwd, log, timeout):
    """A construction timeout terminates its subprocess group, including shell children."""
    proc = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        raise


def run_design_rounds(episode, planner, max_rounds):
    """A plan-only solution is not a submission; continue within real remaining budgets."""
    for number in range(1, max_rounds + 1):
        state = episode.summary()
        if state["final"] is not None:
            return
        if state["finalization_due"] or state["remaining_evaluations"] == 0:
            break
        planner.run_round(number, stop_when=lambda: episode.summary()["final"] is not None,
                          stop_for_budget=lambda: (
                              episode.summary()["finalization_due"]
                              or episode.summary()["remaining_evaluations"] == 0))
        state = episode.summary()
        if state["final"] is not None or state["history"]:
            return
        episode.store.event("no_artifact_after_round", round=number,
                            remaining_llm_requests=state["remaining_llm_requests"])
    if not episode.summary()["history"]:
        raise RuntimeError("No candidate was submitted before the design budget ended")
