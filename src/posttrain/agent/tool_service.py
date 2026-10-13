"""Separate, loopback-only public-input execution service. No hidden test files."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import time
import urllib.request
from urllib.parse import urlsplit

from posttrain.common import digest, write_json
from posttrain.execution.contracts import Limits, Status
from posttrain.execution.service import RequestJournal


class ToolInfrastructureError(RuntimeError):
    pass


def checked_request(row):
    if not isinstance(row, dict) or set(row) != {'code', 'stdin', 'request_id'}:
        raise ValueError('public_execution_fields_only')
    if not isinstance(row['request_id'], str) or not re.fullmatch('[a-f0-9]{32}', row['request_id']):
        raise ValueError('invalid_request_id')
    for key, cap in (('code', 131072), ('stdin', 16384)):
        if not isinstance(row[key], str) or len(row[key].encode()) > cap:
            raise ValueError('tool_input_too_large')
    return row


def make_server(backend, output: Path, port=8081, workers=4):
    if backend.evidence_level != 'isolated_execution' or workers < 1:
        raise ValueError('attested_backend_required')
    limits = Limits(time_s=2)
    output.mkdir(parents=True, exist_ok=True)
    identity = {'executor_version': backend.version, 'scope': 'public_input_only',
                'limits': asdict(limits), 'workers': workers}
    journal = RequestJournal(output / 'requests.sqlite', identity)
    slots = threading.BoundedSemaphore(workers)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, row):
            raw = json.dumps(row, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == '/identity':
                self.reply(200, identity)
                return
            match = re.fullmatch('/requests/([a-f0-9]{32})', self.path)
            row = journal.lookup(match[1]) if match else None
            self.reply(200, row) if row else self.reply(404, {'error': 'not_found'})

        def do_POST(self):
            if self.path != '/run':
                self.reply(404, {'error': 'not_found'})
                return
            acquired = False
            self.connection.settimeout(15)
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 1048576:
                    raise ValueError('request_size')
                row = checked_request(json.loads(self.rfile.read(size)))
                rid = row['request_id']
                claim = {'request_id': rid, 'problem_id': 'public_input',
                         'code_hash': digest(row['code']), 'tests_hash': digest(row['stdin'])}
                if not journal.claim(claim):
                    self.reply(409, {'error': 'reconcile_do_not_redispatch'})
                    return
                start = time.monotonic()
                acquired = slots.acquire(timeout=30)
                if not acquired:
                    raise ToolInfrastructureError('queue_timeout')
                case = backend.run(row['code'], row['stdin'], limits, rid, 32)
                if case.status in (Status.INFRA, Status.DATA):
                    raise ToolInfrastructureError('invalid_execution')
                result = {**claim, 'executor_version': backend.version,
                          'evidence_level': backend.evidence_level, 'status': case.status.value,
                          'stdout': case.stdout[:4096], 'stderr': case.stderr[:4096],
                          'exit_code': case.exit_code, 'truncated': case.truncated or len(case.stdout)>4096 or len(case.stderr)>4096,
                          'wall_s': time.monotonic()-start}
                journal.complete(rid, result)
                self.reply(200, result)
            except (ValueError, TypeError, KeyError) as exc:
                self.reply(400, {'error': type(exc).__name__})
            except Exception:
                # Claimed request remains unknown. Never convert infrastructure to an agent error.
                self.reply(503, {'error': 'execution_unknown_reconcile'})
            finally:
                if acquired:
                    slots.release()
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    server.project_identity = identity
    return server


class ToolClient:
    def __init__(self, endpoint, executor_version):
        parsed = urlsplit(endpoint)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1') or parsed.username or parsed.query or parsed.fragment or parsed.path not in ('', '/'):
            raise ValueError('loopback_SSH_tunnel_required')
        self.endpoint, self.version = endpoint.rstrip('/'), executor_version
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def run(self, code, stdin, request_id):
        row = checked_request({'code': code, 'stdin': stdin, 'request_id': request_id})
        req = urllib.request.Request(self.endpoint+'/run', data=json.dumps(row).encode(), headers={'Content-Type':'application/json'})
        try:
            try:
                with self.opener.open(req, timeout=65) as r:
                    result = json.loads(r.read(131073))
            except OSError:
                with self.opener.open(self.endpoint+'/requests/'+request_id, timeout=15) as r:
                    receipt = json.loads(r.read(131073))
                if receipt.get('state') != 'complete':
                    raise ToolInfrastructureError('unknown_tool_outcome')
                result = receipt['result']
            for key, expected in {'request_id': request_id, 'code_hash': digest(code), 'tests_hash': digest(stdin),
                                  'executor_version': self.version, 'evidence_level': 'isolated_execution'}.items():
                if result.get(key) != expected:
                    raise ToolInfrastructureError('tool_identity_mismatch')
            if result.get('status') not in {s.value for s in Status if s not in (Status.INFRA, Status.DATA)}:
                raise ToolInfrastructureError('invalid_tool_status')
            # Only process observations reach the model. No scorer information is available here.
            return {k: result[k] for k in ('status', 'stdout', 'stderr', 'exit_code', 'truncated')}
        except Exception as exc:
            raise ToolInfrastructureError('tool_failed_no_numeric_reward') from exc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--attestation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=8081)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    from posttrain.execution.docker_backend import DockerBackend
    backend = DockerBackend(a.image, Limits(time_s=30), a.attestation)
    server = make_server(backend, a.output, a.port, a.workers)
    write_json(a.output/'identity.json', server.project_identity)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
