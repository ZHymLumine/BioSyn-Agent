from types import SimpleNamespace
import pytest
from biosyn.agent.state import RunStore
from biosyn.config import RuntimeConfig
from biosyn.applications.retrospective import llm_config


@pytest.mark.parametrize('marker', ['#!BASH', '#!CLI'])
def test_biomni_executes_leading_space_single_line_shell(tmp_path, marker):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from biosyn.agent.planner import BiomniPlanner
    from biosyn.applications.worker import Task
    target = tmp_path / 'executed.txt'
    messages = [f'<execute> {marker} printf executed > {target} </execute>', '<solution>done</solution>']
    runtime = SimpleNamespace(store=RunStore(tmp_path / 'trace'), gateway=SimpleNamespace(catalog={}),
        state=SimpleNamespace(task=Task(id='shell-test', request='Run shell command'), run_id='shell-test'),
        summary=lambda: {}, submit=lambda candidate: candidate)
    config = RuntimeConfig.model_validate(llm_config())
    config.use_tool_retriever = False
    BiomniPlanner(config, runtime, llm=FakeListChatModel(responses=messages)).run_round(1)
    assert target.read_text() == 'executed'


def test_budget_checkpoint_executes_pending_action_before_stopping(tmp_path):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from biosyn.agent.planner import BiomniPlanner
    from biosyn.applications.worker import Task
    target = tmp_path / 'pending.txt'
    model = FakeListChatModel(responses=[f'<execute>open({str(target)!r}, "w").write("saved")</execute>',
                                        '<solution>must not spend this request</solution>'])
    runtime = SimpleNamespace(store=RunStore(tmp_path / 'trace'), gateway=SimpleNamespace(catalog={}),
        state=SimpleNamespace(task=Task(id='reserve-test', request='Save candidate'), run_id='reserve-test'),
        summary=lambda: {}, submit=lambda candidate: candidate)
    config = RuntimeConfig.model_validate(llm_config())
    config.use_tool_retriever = False
    BiomniPlanner(config, runtime, llm=model).run_round(1, stop_for_budget=lambda: model.i == 1)
    assert target.read_text() == 'saved'
    assert model.i == 1


def test_truncated_response_is_not_repaired_and_executed(tmp_path):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from biosyn.agent.planner import BiomniPlanner
    from biosyn.applications.worker import Task
    partial, complete = tmp_path / 'partial.txt', tmp_path / 'complete.txt'
    model = FakeMessagesListChatModel(responses=[
        AIMessage(content=f'<execute>open({str(partial)!r}, "w").write("must not execute")',
                  response_metadata={'finish_reason': 'length'}),
        AIMessage(content=f'<execute>open({str(complete)!r}, "w").write("complete")',
                  response_metadata={'finish_reason': 'stop'}),
        AIMessage(content='<solution>done</solution>', response_metadata={'finish_reason': 'stop'}),
    ])
    runtime = SimpleNamespace(store=RunStore(tmp_path / 'trace'), gateway=SimpleNamespace(catalog={}),
        state=SimpleNamespace(task=Task(id='truncate-test', request='Save a complete artifact'), run_id='truncate-test'),
        summary=lambda: {}, submit=lambda candidate: candidate)
    config = RuntimeConfig.model_validate(llm_config())
    config.use_tool_retriever = False
    BiomniPlanner(config, runtime, llm=model).run_round(1)
    assert not partial.exists()
    assert complete.read_text() == 'complete'
    assert 'reached the output token limit' in (tmp_path / 'trace/conversation-1.json').read_text()
