from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

from posttrain.common import read_jsonl,utc_now,write_json,write_jsonl


def main(argv=None):
    parser=argparse.ArgumentParser(description='Auditable execution-feedback experiment tools')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor')
    fetch=sub.add_parser('fetch-taco')
    fetch.add_argument('--sources',type=Path,default=Path('configs/sources.json'))
    fetch.add_argument('--output',type=Path,default=Path('data/raw/taco-train-00000.parquet'))
    prep=sub.add_parser('prepare-taco')
    prep.add_argument('--input',type=Path,default=Path('data/raw/taco-train-00000.parquet'))
    prep.add_argument('--sources',type=Path,default=Path('configs/sources.json'))
    prep.add_argument('--output',type=Path,required=True)
    prep.add_argument('--limit',type=int,default=100)
    smoke=sub.add_parser('contract-smoke')
    smoke.add_argument('--output',type=Path,required=True)
    isolation=sub.add_parser('isolation-check')
    isolation.add_argument('--image',required=True)
    isolation.add_argument('--output',type=Path,required=True)
    refs=sub.add_parser('validate-references')
    refs.add_argument('--input',type=Path,required=True)
    refs.add_argument('--image',required=True)
    refs.add_argument('--attestation',type=Path,required=True)
    refs.add_argument('--output',type=Path,required=True)
    refs.add_argument('--resume',action='store_true')
    refs.add_argument('--max-references',type=int,default=3)
    args=parser.parse_args(argv)
    if args.command=='doctor':
        result={'checked_at':utc_now(),'python':sys.version,'platform':platform.platform(),
                'docker_cli':shutil.which('docker'),'linux_host':sys.platform=='linux',
                'candidate_execution_available':False,'note':'availability requires live isolation-check attestation'}
    elif args.command=='fetch-taco':
        from posttrain.common import file_digest
        from posttrain.data.download import download_verified
        src=json.loads(args.sources.read_text())['taco']
        url=f"https://huggingface.co/datasets/{src['repo_id']}/resolve/{src['revision']}/{src['stage1_file']}"
        sha=download_verified(url,args.output,src['stage1_sha256'])
        result={'source':src,'path':str(args.output),'sha256':sha,'downloaded_at':utc_now()}
        write_json(Path('manifests/taco_stage1_download.json'),result)
    elif args.command=='prepare-taco':
        from posttrain.data.prepare import prepare_taco
        from posttrain.common import file_digest
        source=json.loads(args.sources.read_text())['taco']
        if file_digest(args.input)!=source['stage1_sha256']:
            raise ValueError('source_checksum_mismatch')
        args.output.mkdir(parents=True,exist_ok=False)
        result=prepare_taco(args.input,source['revision'],args.output,args.limit)
    elif args.command=='contract-smoke':
        from posttrain.smoke import contract_smoke
        result=contract_smoke(args.output)
    elif args.command=='isolation-check':
        from posttrain.execution.probes import isolation_check
        result=isolation_check(args.image,args.output)
        print(json.dumps({'passed':result['passed'],'report':str(args.output/'attestation.json')}))
        return 0 if result['passed'] else 2
    elif args.command=='validate-references':
        from posttrain.data.reference_run import reference_run
        from posttrain.execution.contracts import Limits
        from posttrain.execution.docker_backend import DockerBackend
        backend=DockerBackend(args.image,Limits(time_s=30),args.attestation)
        result=reference_run(args.input,args.output,backend,resume=args.resume,max_references=args.max_references)
    if args.command=='contract-smoke':
        result={k:v for k,v in result.items() if k!='group_statistics'}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
