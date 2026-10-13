import json
import sys
from dataclasses import asdict
from datetime import datetime,timedelta,timezone

import pytest

from posttrain.common import write_json,file_digest
from posttrain.execution.contracts import Limits
from posttrain.execution.docker_backend import DockerBackend,container_command
from posttrain.execution.sandboxfusion import REQUIRED_PROBES,validate_attestation


def test_container_has_no_credentials_mounts_network_or_privileged_mode():
    cmd=container_command('sha256:'+'a'*64,'safe',Limits())
    assert '--network=none' in cmd and '--read-only' in cmd
    assert '--cap-drop=ALL' in cmd and '--security-opt=no-new-privileges:true' in cmd
    assert all(x not in cmd for x in ['--privileged','--gpus','-v','--volume','--env-file','-e','--mount'])
    with pytest.raises(ValueError):
        container_command('latest','name',Limits())


def test_attestation_rejects_mock_incomplete_expired_and_tampered_evidence(tmp_path):
    path=tmp_path/'attestation.json'
    record={'evidence_level':'mock_contract','endpoint':'local','expires_at':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
            'limits':asdict(Limits()),'probes':{},'image_digest':'sha256:'+'a'*64,'deployment_id':'test'}
    write_json(path,record)
    with pytest.raises(ValueError): validate_attestation(path,'local',Limits())
    record['evidence_level']='isolated_execution'
    write_json(path,record)
    with pytest.raises(ValueError): validate_attestation(path,'local',Limits())
    for name in REQUIRED_PROBES:
        file=tmp_path/f'{name}.json';write_json(file,{'passed':True,'test_fixture':True})
        record['probes'][name]={'passed':True,'evidence_file':file.name,'sha256':file_digest(file)}
    write_json(path,record)
    validate_attestation(path,'local',Limits())
    file.write_text('tampered')
    with pytest.raises(ValueError): validate_attestation(path,'local',Limits())
    record['expires_at']='2020-01-01T00:00:00+00:00';write_json(path,record)
    with pytest.raises(ValueError,match='expired'): validate_attestation(path,'local',Limits())


@pytest.mark.skipif(sys.platform=='linux',reason='Windows host guard')
def test_windows_refuses_candidate_execution_before_even_reading_attestation(tmp_path):
    with pytest.raises(RuntimeError,match='Linux'):
        DockerBackend('sha256:'+'a'*64,Limits(),tmp_path/'nonexistent')
