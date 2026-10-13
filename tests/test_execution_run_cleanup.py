import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from posttrain.execution.contracts import Limits
from posttrain.execution.docker_backend import container_command


def cleanup_module():
    spec = importlib.util.spec_from_file_location('cleanup_execution_run', Path('scripts/cleanup_execution_run.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_scope_is_label_only(monkeypatch):
    scope = 'a' * 32
    monkeypatch.setenv('POSTTRAIN_EXECUTION_RUN', scope)
    command = container_command('sha256:' + 'b' * 64, 'sample', Limits())
    assert command[command.index('--label') + 1] == 'posttrain.execution_run=' + scope
    assert '--env' not in command and '-e' not in command
    monkeypatch.setenv('POSTTRAIN_EXECUTION_RUN', 'invalid scope')
    with pytest.raises(ValueError):
        container_command('sha256:' + 'b' * 64, 'sample', Limits())


def test_cleanup_refuses_other_runs():
    module = cleanup_module()
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[1] == 'ps':
            return SimpleNamespace(returncode=0, stdout='c' * 64 + '\n', stderr='')
        return SimpleNamespace(returncode=0, stdout=json.dumps([{'Config': {'Labels': {'posttrain.execution_run': 'b' * 32}}}]), stderr='')
    result = module.cleanup('a' * 32, run=run)
    assert not result['all_owned_containers_removed']
    assert not any(command[1] == 'rm' for command in calls)


def test_cleanup_records_failed_removal_and_verifies_empty():
    module = cleanup_module()
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[1] == 'ps':
            return SimpleNamespace(returncode=0, stdout='c' * 64 + '\n', stderr='')
        if command[1] == 'inspect':
            return SimpleNamespace(returncode=0, stdout=json.dumps([{'Config': {'Labels': {'posttrain.execution_run': 'a' * 32}}}]), stderr='')
        return SimpleNamespace(returncode=1, stdout='', stderr='daemon unavailable')
    result = module.cleanup('a' * 32, run=run)
    assert not result['all_owned_containers_removed']
    assert result['errors']


def test_empty_cleanup_and_bad_scope():
    module = cleanup_module()
    def run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    assert module.cleanup('a' * 32, run=run)['all_owned_containers_removed']
    with pytest.raises(ValueError):
        module.cleanup('', run=run)


def test_cleanup_only_removes_confirmed_owned_container_then_rechecks():
    module = cleanup_module()
    calls, inventory = [], ['c' * 64]
    def run(command, **kwargs):
        calls.append(command)
        if command[1] == 'ps':
            assert command[-1] == 'label=posttrain.execution_run=' + 'a' * 32
            return SimpleNamespace(returncode=0, stdout='\n'.join(inventory), stderr='')
        if command[1] == 'inspect':
            return SimpleNamespace(returncode=0, stdout=json.dumps([{'Config': {'Labels': {'posttrain.execution_run': 'a' * 32}}}]), stderr='')
        assert command == ['docker', 'rm', '--force', 'c' * 64]
        inventory.clear()
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    result = module.cleanup('a' * 32, run=run)
    assert result['all_owned_containers_removed']
    assert result['removed'] == ['c' * 64]
    assert [command[1] for command in calls] == ['ps', 'inspect', 'rm', 'ps']
