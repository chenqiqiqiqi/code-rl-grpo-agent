from datetime import datetime,timezone
from types import SimpleNamespace
import pytest

from posttrain.training.paid_guard import approved_deadline
from posttrain.training import paid_guard
from posttrain.common import write_json


def test_no_paid_job_without_explicit_matching_authorization():
    start=datetime(2026,9,26,tzinfo=timezone.utc)
    row={'authorized':True,'user_confirmation':'approved recorded request','provider':'autodl',
        'resource_id':'project-instance','gpu_count':1,'hourly_cny':8,'max_gpu_hours':8,
        'gpu_spend_cap_cny':64,'auto_shutdown_authorized':True}
    assert (approved_deadline(row,'project-instance',start,start)-start).total_seconds()==28800
    for change in ({'authorized':False},{'user_confirmation':''},{'gpu_count':4},
        {'hourly_cny':9},{'auto_shutdown_authorized':False},{'resource_id':'another-instance'}):
        with pytest.raises(ValueError):approved_deadline({**row,**change},'project-instance',start,start)


def test_shutdown_attempt_survives_audit_disk_failure(tmp_path, monkeypatch):
    attempted=[]
    def disk_full(*args):raise OSError('simulated disk full')
    def fake_shutdown(command,**kwargs):
        attempted.append(command)
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(paid_guard,'write_json',disk_full)
    monkeypatch.setattr(paid_guard.subprocess,'run',fake_shutdown)
    assert paid_guard.stop(tmp_path,'test_only')==0
    assert attempted==[['/usr/bin/shutdown']]


def test_existing_ready_directory_cannot_launch_paid_work(tmp_path, monkeypatch):
    output=tmp_path/'existing';output.mkdir()
    write_json(output/'watchdog-ready.json',{'pid':7,'deadline':'old'})
    start=datetime.now(timezone.utc).isoformat()
    approval=tmp_path/'approval.json'
    write_json(approval,{'authorized':True,'user_confirmation':'unit test only','provider':'autodl',
        'resource_id':'synthetic-id','gpu_count':1,'hourly_cny':8,'max_gpu_hours':8,
        'gpu_spend_cap_cny':64,'auto_shutdown_authorized':True})
    monkeypatch.setattr(paid_guard.sys,'platform','linux')
    monkeypatch.setenv('AutoDLContainerUUID','synthetic-id')
    monkeypatch.setattr(paid_guard.sys,'argv',['guard','--approval',str(approval),
        '--billing-start',start,'--output',str(output),'--','synthetic-never-executed'])
    spawned=[]
    def fake_popen(*args,**kwargs):
        spawned.append(args)
        return SimpleNamespace(pid=7,returncode=0,poll=lambda:0)
    monkeypatch.setattr(paid_guard.subprocess,'Popen',fake_popen)
    monkeypatch.setattr(paid_guard,'stop',lambda *args:0)
    with pytest.raises(FileExistsError):paid_guard.main()
    assert not spawned


@pytest.mark.parametrize('failure',['spawn','mkdir'])
def test_watchdog_setup_failure_still_requests_shutdown(tmp_path, monkeypatch, failure):
    approval=tmp_path/'approval.json';output=tmp_path/'fresh'
    write_json(approval,{'authorized':True,'user_confirmation':'unit test only','provider':'autodl',
        'resource_id':'synthetic-id','gpu_count':1,'hourly_cny':8,'max_gpu_hours':8,
        'gpu_spend_cap_cny':64,'auto_shutdown_authorized':True})
    monkeypatch.setattr(paid_guard.sys,'platform','linux');monkeypatch.setenv('AutoDLContainerUUID','synthetic-id')
    monkeypatch.setattr(paid_guard.sys,'argv',['guard','--approval',str(approval),
        '--billing-start',datetime.now(timezone.utc).isoformat(),'--output',str(output),'--','test-never-executed'])
    def spawn_failure(*args,**kwargs):raise OSError('simulated process launch failure')
    monkeypatch.setattr(paid_guard.subprocess,'Popen',spawn_failure)
    if failure=='mkdir':
        original_mkdir=type(output).mkdir
        def mkdir_failure(path,*args,**kwargs):
            if path==output:raise OSError('simulated full disk before watchdog')
            return original_mkdir(path,*args,**kwargs)
        monkeypatch.setattr(type(output),'mkdir',mkdir_failure)
    stopped=[];monkeypatch.setattr(paid_guard,'stop',lambda *args:stopped.append(args))
    with pytest.raises(OSError):paid_guard.main()
    assert len(stopped)==1
