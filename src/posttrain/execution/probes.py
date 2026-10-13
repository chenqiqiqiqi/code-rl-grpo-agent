from __future__ import annotations

import json
import secrets
import sys
import tempfile
from dataclasses import asdict, replace
from datetime import datetime,timedelta,timezone
from pathlib import Path

from posttrain.common import file_digest,utc_now,write_json
from posttrain.execution.contracts import Limits,Status
from posttrain.execution.docker_backend import execute_container,execute_batch
from posttrain.execution.sandboxfusion import REQUIRED_PROBES


def isolation_check(image: str, output: Path, limits: Limits = Limits(time_s=30)) -> dict:
    """Execute a fixed bounded probe registry. No arbitrary candidate bypass flag."""
    if sys.platform!='linux':
        raise RuntimeError('Linux host with Docker required for live isolation probes')
    output.mkdir(parents=True,exist_ok=False)
    canary=output/'host-canary.txt'
    canary.write_text(secrets.token_hex(32),encoding='utf-8')
    read_probe=f"from pathlib import Path\ntry:\n Path({str(canary.resolve())!r}).read_bytes();print('LEAK')\nexcept OSError:\n print('DENIED')"
    specs={
      'non_root':("import os\nfrom pathlib import Path\ns=Path('/proc/self/status').read_text()\nc=next(x.split()[1] for x in s.splitlines() if x.startswith('CapEff:'))\nprint(os.getuid(),os.getgid(),int(c,16))",lambda r:r.stdout.strip()=='10001 10001 0'),
      'network_denied':("import socket\ns=socket.socket()\ns.settimeout(1)\ntry:\n s.connect(('1.1.1.1',443));print('LEAK')\nexcept OSError:\n print('DENIED')",lambda r:r.stdout.strip()=='DENIED'),
      'host_canary_unreadable':(read_probe,lambda r:r.stdout.strip()=='DENIED'),
      'secrets_absent':("import os\nprint(sorted(os.environ))",lambda r:json.loads(r.stdout.replace("'",'"').strip())==['HOME','LANG','PATH']),
      'answers_absent':("from pathlib import Path\nleaks=[]\nfor name in ['/proc/1/environ','/proc/1/mem','/proc/1/fd/0','/opt/SandboxFusion/sandbox/datasets']:\n try:\n  Path(name).read_bytes();leaks.append(name)\n except OSError: pass\nprint('LEAK' if leaks else 'DENIED')",lambda r:r.stdout.strip()=='DENIED'),
      'read_only_root':("try:\n open('/host-write-probe','w').write('x');print('LEAK')\nexcept OSError:\n print('DENIED')",lambda r:r.stdout.strip()=='DENIED'),
      'memory_limit':("x=bytearray(1024*1024*1024)",lambda r:r.status==Status.MEMORY),
      'output_limit':("import sys\nfor _ in range(4096):\n sys.stdout.write('x'*4096)",lambda r:r.status==Status.OUTPUT and r.truncated),
      'timeout':("while True: pass",lambda r:r.status==Status.TIMEOUT),
      'process_limit':("import os,time\nchildren=[]\ntry:\n for _ in range(64):\n  p=os.fork()\n  if p==0:\n   time.sleep(1);os._exit(0)\n  children.append(p)\n print('UNLIMITED')\nexcept OSError:\n print('LIMITED')\nfinally:\n for p in children:\n  try: os.waitpid(p,0)\n  except ChildProcessError: pass",lambda r:r.stdout.strip()=='LIMITED'),
      'workspace_reset':("from pathlib import Path\np=Path('/tmp/persist-probe')\nprint('LEAK' if p.exists() else 'CLEAN')\np.write_text('x')",lambda r:r.stdout.strip()=='CLEAN'),
      'cpu_limit':("from pathlib import Path\np=Path('/sys/fs/cgroup/cpu.max')\nprint(p.read_text() if p.exists() else 'CGROUP_V1_UNSUPPORTED')",lambda r:r.stdout.strip().split()==[str(limits.cpu_cores*100000),'100000']),
      'child_cleanup':("import os,time\np=os.fork()\nif p==0:\n time.sleep(1);os._exit(0)\nprint('SPAWNED')",lambda r:r.stdout.strip()=='SPAWNED'),
      'unreadable_workspace_cleanup':("import os\nos.mkdir('closed');os.chmod('closed',0)\nprint('DONE')",lambda r:r.status==Status.OK and r.stdout.strip()=='DONE'),
      'candidate_sigkill':("import os,signal\nos.kill(os.getpid(),signal.SIGKILL)",lambda r:r.status==Status.RUNTIME and r.exit_code==-9),
      'python_script_exit':("print('DONE');exit()",lambda r:r.status==Status.OK and r.stdout.strip()=='DONE'),
    }
    probes={}
    for name,(code,check) in specs.items():
        probe_limits=replace(limits,time_s=1) if name=='timeout' else limits
        result,evidence=execute_container(image,code,'',probe_limits,limits.time_s+45)
        try: passed=bool(check(result))
        except (ValueError,TypeError): passed=False
        info=evidence.get('inspect') or {}
        host=info.get('HostConfig',{})
        # Verify actual daemon state as well as command-line intent.
        passed &= (host.get('NetworkMode')=='none' and host.get('ReadonlyRootfs') is True
                   and not info.get('Mounts') and host.get('Privileged') is False
                   and host.get('Memory')==(limits.memory_mb+256)*1024*1024
                   and host.get('PidsLimit')==limits.pids+4
                   and evidence.get('cleanup_return_code')==0)
        if name=='workspace_reset':
            again,second=execute_container(image,code,'',limits,limits.time_s+45)
            passed &= check(again) and second.get('cleanup_return_code')==0
            evidence['second_container']=second
        evidence['probe']=name
        evidence['passed']=bool(passed)
        file=output/f'{name}.json'
        write_json(file,evidence)
        probes[name]={'passed':bool(passed),'evidence_file':file.name,'sha256':file_digest(file)}
    batch_specs={
        'stdin_regular_file':("import os\nprint(os.read(0,os.fstat(0).st_size).decode().strip())",lambda rows:[r.stdout.strip() for r in rows]==['first','second'] and all(r.status==Status.OK for r in rows)),
        'batch_workspace_reset':("from pathlib import Path\np=Path('/tmp/persist-probe')\nprint('LEAK' if p.exists() else 'CLEAN')\np.write_text('x')",lambda rows:all(r.status==Status.OK and r.stdout.strip()=='CLEAN' for r in rows)),
        'batch_child_cleanup':("import os,time\nfrom pathlib import Path\nimport sys\nif sys.stdin.read().strip()=='first':\n p=os.fork()\n if p==0:\n  os.setsid();os.close(0);os.close(1);os.close(2);time.sleep(2);Path('/tmp/escaped').write_text('LEAK');time.sleep(2);os._exit(0)\n print('SPAWNED')\nelse:\n time.sleep(3);print('LEAK' if Path('/tmp/escaped').exists() else 'CLEAN')",lambda rows:rows[0].stdout.strip()=='SPAWNED' and rows[1].stdout.strip()=='CLEAN' and all(r.status==Status.OK for r in rows)),
    }
    for name,(code,check) in batch_specs.items():
        rows,evidence=execute_batch(image,code,['first','second'],limits,limits.time_s*2+45)
        passed=bool(check(rows)) and evidence.get('cleanup_return_code')==0
        evidence.update(probe=name,passed=passed)
        file=output/f'{name}.json';write_json(file,evidence)
        probes[name]={'passed':passed,'evidence_file':file.name,'sha256':file_digest(file)}
    canary.unlink()
    passed=set(probes)==REQUIRED_PROBES and all(p['passed'] for p in probes.values())
    record={'evidence_level':'isolated_execution','endpoint':'docker://'+image,'image_digest':image,
            'deployment_id':'sf469de084-project-batch-v3','created_at':utc_now(),
            'expires_at':(datetime.now(timezone.utc)+timedelta(hours=24)).isoformat(),
            'limits':asdict(limits),'probes':probes,'passed':passed,
            'scope':'Linux Docker boundary smoke verification, not a formal security proof'}
    write_json(output/'attestation.json',record)
    return record
