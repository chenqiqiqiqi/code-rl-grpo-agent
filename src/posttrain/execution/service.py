"""Loopback-only score gateway; use SSH forwarding between GPU and CPU nodes.

Only the CPU gateway loads tests. The disposable candidate container receives code
and test inputs, never expected outputs. Queue failures return invalid scores.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import threading
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import fields,replace
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from posttrain.common import canonical,digest,file_digest,read_jsonl,utc_now,write_json
from posttrain.data.schema import Problem
from posttrain.execution.contracts import CaseResult,JudgeResult,Limits,Status
from posttrain.execution.judge import Judge


class RequestJournal:
    """Durable at-most-once dispatch. Unknown executions require reconciliation.

    Construct once per server process, never once per HTTP request. SQLite is the
    source of truth; per-request JSON files are recoverable exports of results.
    """
    def __init__(self,path,identity):
        self.path=path
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY, value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, request TEXT NOT NULL, '
                       'state TEXT NOT NULL, result TEXT, accepted_at TEXT NOT NULL, completed_at TEXT)')
            db.execute('BEGIN IMMEDIATE')
            previous=db.execute('SELECT value FROM identity WHERE id=1').fetchone()
            if previous and previous[0]!=canonical(identity):raise ValueError('gateway_identity_changed')
            db.execute('INSERT OR IGNORE INTO identity VALUES (1,?)',(canonical(identity),))
            db.execute("UPDATE requests SET state='unknown_after_restart' WHERE state='pending'")

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=30)
        db.execute('PRAGMA synchronous=FULL')
        try:
            with db:yield db
        finally:db.close()

    def claim(self,request):
        with self.connect() as db:
            row=db.execute('INSERT OR IGNORE INTO requests VALUES (?,?,?,?,?,NULL)',
                (request['request_id'],canonical(request),'pending',None,utc_now()))
            return row.rowcount==1

    def lookup(self,request_id):
        with self.connect() as db:
            row=db.execute('SELECT request,state,result,accepted_at,completed_at FROM requests WHERE id=?',
                           (request_id,)).fetchone()
        if row is None:return None
        return {'request':json.loads(row[0]),'state':row[1],'result':json.loads(row[2]) if row[2] else None,
                'accepted_at':row[3],'completed_at':row[4]}

    def complete(self,request_id,result):
        with self.connect() as db:
            changed=db.execute("UPDATE requests SET state='complete',result=?,completed_at=? "
                "WHERE id=? AND state IN ('pending','unknown_after_restart')",
                (canonical(result),utc_now(),request_id))
            if changed.rowcount!=1:raise ValueError('request_absent_or_already_completed')


class ScoringOutcomeUnknown(RuntimeError):
    def __init__(self,receipt):
        self.receipt=receipt
        super().__init__('scoring_outcome_unknown_do_not_redispatch')


def checked_reply(row,request_id,code,tests_hash,executor_version,expected_test_count,scoring_mode='full'):
    if row.get('scoring_mode','full') != scoring_mode:
        raise ValueError('score_mode_mismatch')
    for key,expected in [('request_id',request_id),('code_hash',digest(code)),('tests_hash',tests_hash),
                         ('executor_version',executor_version),('evidence_level','isolated_execution')]:
        if row.get(key)!=expected:raise ValueError('score_identity_mismatch:'+key)
    for key in ('passed','total','retries'):
        if type(row.get(key)) is not int or row[key]<0:raise ValueError('malformed_score_count')
    if row['total']!=expected_test_count:raise ValueError('frozen_test_count_mismatch')
    for key in ('elapsed_s','queue_s'):
        value=row.get(key)
        if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or value<0):
            raise ValueError('malformed_score_timing')
    data={f.name:row[f.name] for f in fields(JudgeResult) if f.name in row}
    data['status']=Status(data['status'])
    data['cases']=tuple(CaseResult(**{**c,'status':Status(c['status'])}) for c in data['cases'])
    result=JudgeResult(**data)
    if type(row.get('valid')) is not bool or row['valid']!=result.valid:
        raise ValueError('inconsistent_remote_score')
    if scoring_mode == 'binary_fail_fast' and row.get('binary_valid') is not result.binary_valid:
        raise ValueError('inconsistent_binary_score')
    return result


class ScoringClient:
    def __init__(self,endpoint,executor_version,problems,timeout_s=1800,scoring_mode='full'):
        if scoring_mode not in ('full','binary_fail_fast'):raise ValueError('invalid_scoring_mode')
        self.scoring_mode=scoring_mode
        parsed=urlsplit(endpoint)
        if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost','::1') or parsed.username or parsed.query:
            raise ValueError('use_loopback_SSH_forward_not_public_unencrypted_service')
        self.endpoint=endpoint.rstrip('/');self.version=executor_version
        self.problems=problems;self.timeout_s=timeout_s

    def score(self,problem_id,code,request_id):
        expected=self.problems[problem_id]
        request=urllib.request.Request(self.endpoint+'/score',data=json.dumps({
            'problem_id':problem_id,'code':code,'request_id':request_id,
            **({'scoring_mode':self.scoring_mode} if self.scoring_mode!='full' else {})}).encode(),
            headers={'Content-Type':'application/json'})
        # Do not use environment proxies for a local SSH tunnel.
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request,timeout=self.timeout_s) as response:
            raw=response.read(expected['test_count']*400000+65537)
        return checked_reply(json.loads(raw),request_id,code,expected['tests_hash'],self.version,expected['test_count'],self.scoring_mode)

    def reconcile(self,problem_id,code,request_id):
        if not re.fullmatch('[A-Za-z0-9_-]{1,100}',request_id):raise ValueError('invalid_request_id')
        expected=self.problems[problem_id]
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(self.endpoint+'/requests/'+request_id,timeout=self.timeout_s) as response:
            receipt=json.loads(response.read(expected['test_count']*400000+65537))
        if receipt['state']=='legacy_complete':
            # Old versions did not persist problem_id. Bind the saved result to
            # the caller's exact code, frozen tests, test count and executor.
            return checked_reply(receipt['result'],request_id,code,expected['tests_hash'],self.version,expected['test_count'],self.scoring_mode)
        if receipt['request'].get('scoring_mode','full') != self.scoring_mode:
            raise ValueError('request_reconciliation_mode_mismatch')
        for key,value in [('request_id',request_id),('problem_id',problem_id),('code_hash',digest(code)),
                          ('tests_hash',expected['tests_hash'])]:
            if receipt['request'].get(key)!=value:raise ValueError('request_reconciliation_identity_mismatch')
        if receipt['state']!='complete':raise ScoringOutcomeUnknown(receipt)
        return checked_reply(receipt['result'],request_id,code,expected['tests_hash'],self.version,expected['test_count'],self.scoring_mode)


def make_server(problems,backend,output,*,port=8080,workers=1,queue_timeout_s=30):
    if backend.evidence_level!='isolated_execution' or workers<1 or queue_timeout_s<=0:
        raise ValueError('invalid_gateway_configuration')
    if not problems or any(p.validation_status!='reference_passed' for p in problems):
        raise ValueError('only_reference_validated_problems_can_be_scored')
    by_id={p.problem_id:p for p in problems}
    if len(by_id)!=len(problems):raise ValueError('duplicate_problem_id')
    output.mkdir(parents=True,exist_ok=True)
    slots=threading.BoundedSemaphore(workers)
    identity={'executor_version':backend.version,'problem_set_sha256':digest([p.to_dict() for p in problems]),
              'workers':workers,'queue_timeout_s':queue_timeout_s,
              'scoring_protocol':'full_or_binary_fail_fast_v1'}
    journal=RequestJournal(output/'requests.sqlite',identity)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send_json(self,status,data):
            raw=json.dumps(data,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw)));self.end_headers()
            self.wfile.write(raw)
        def do_GET(self):
            if self.path=='/identity':self.send_json(200,identity);return
            match=re.fullmatch('/requests/([A-Za-z0-9_-]{1,100})',self.path)
            row=journal.lookup(match.group(1)) if match else None
            if row is None and match:
                legacy=output/(match.group(1)+'.json')
                if legacy.is_file():row={'state':'legacy_complete','result':json.loads(legacy.read_text(encoding='utf-8'))}
            self.send_json(200,row) if row else self.send_json(404,{'error':'not_found'})
        def do_POST(self):
            if self.path!='/score':self.send_json(404,{'error':'not_found'});return
            self.connection.settimeout(15)
            acquired=False
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=2*1024*1024:raise ValueError('invalid_request_size')
                body=json.loads(self.rfile.read(length))
                if set(body) not in ({'problem_id','code','request_id'},
                        {'problem_id','code','request_id','scoring_mode'}):raise ValueError('invalid_request_fields')
                mode=body.get('scoring_mode','full')
                if mode not in ('full','binary_fail_fast'):raise ValueError('invalid_scoring_mode')
                p=by_id[body['problem_id']];code=body['code'];rid=body['request_id']
                if not isinstance(code,str) or not isinstance(rid,str) or not re.fullmatch('[A-Za-z0-9_-]{1,100}',rid):
                    raise ValueError('invalid_code_or_request_id')
                path=output/(rid+'.json')
                # A legacy result file also prevents re-execution after upgrade.
                if path.exists() or not journal.claim({'request_id':rid,'problem_id':p.problem_id,
                    'code_hash':digest(code),'tests_hash':p.tests_hash,'scoring_mode':mode}):
                    self.send_json(409,{'error':'request_id_already_used_reconcile_saved_record'});return
                queued=time.monotonic();acquired=slots.acquire(timeout=queue_timeout_s)
                queue_s=time.monotonic()-queued
                if acquired:
                    result=Judge(backend).score(code,p.tests,Limits(time_s=p.time_limit_s),rid,scoring_mode=mode)
                    result=replace(result,queue_s=queue_s+(result.queue_s or 0),elapsed_s=result.elapsed_s+queue_s)
                else:
                    result=JudgeResult(rid,Status.INFRA,0,len(p.tests),(),queue_s,queue_s,0,0,False,
                        backend.version,backend.evidence_level,digest(code),p.tests_hash,mode)
                record=result.to_dict()
                if not acquired:record['gateway_failure']='queue_timeout'
                journal.complete(rid,record)
                write_json(path,record)
                self.send_json(200,record)
            except (ValueError,KeyError,TypeError) as exc:
                self.send_json(400,{'error':type(exc).__name__})
            finally:
                if acquired:slots.release()
    class BurstHTTPServer(ThreadingHTTPServer):
        # A simultaneous rollout burst can exceed the stdlib default backlog (5)
        # before handler threads have a chance to accept every connection.
        request_queue_size=64

    server=BurstHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads=True
    server.project_identity=identity
    return server


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--extra-input',type=Path,action='append',default=[])
    parser.add_argument('--image',required=True);parser.add_argument('--attestation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--port',type=int,default=8080)
    parser.add_argument('--workers',type=int,default=1);parser.add_argument('--queue-timeout',type=float,default=30)
    args=parser.parse_args()
    from posttrain.execution.docker_backend import DockerBackend
    backend=DockerBackend(args.image,Limits(time_s=30),args.attestation)
    server=make_server([Problem.from_dict(x) for path in [args.input,*args.extra_input] for x in read_jsonl(path)],backend,args.output,
        port=args.port,workers=args.workers,queue_timeout_s=args.queue_timeout)
    write_json(args.output/'gateway_identity.json',server.project_identity)
    print(json.dumps(server.project_identity),flush=True)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
