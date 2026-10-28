"""Opt-in real DeepSeek acceptance. Uses only a disposable synthetic workspace.

Requires DEEPSEEK_API_KEY in the environment and --run-live (API usage is billed).
Raw transcripts/archives stay under --evidence-root; report.json contains only metrics/checks.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-live', action='store_true', required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--model', default='deepseek-flash')
    args = parser.parse_args()
    key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if not key:
        raise SystemExit('DEEPSEEK_API_KEY is missing')
    root = args.evidence_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    workspace = root / 'workspace'
    workspace.mkdir()
    (workspace / 'receipt.txt').write_text('TINA_REAL_TOOL_RECEIPT_COPPER_5821\n', encoding='utf-8')
    subprocess.run(['git', 'init', '--quiet', str(workspace)], check=True)
    env = {**os.environ, 'TINA_HOME': str(root / 'tina-home'), 'NO_COLOR': '1'}
    # Explicit flags prevent unrelated endpoint/model environment overrides.
    common = [str(args.binary.resolve()), 'chat', '--workspace', str(workspace),
              '--base-url', 'https://api.deepseek.com/v1', '--model', args.model,
              '--api-key-env', 'DEEPSEEK_API_KEY', '--context-positions', '65536',
              '--max-tokens', '2048', '--timeout-seconds', '90', '--no-splash']
    cases = []

    def run(name, prompt=None, *, thinking='off', chat_only=True, archive=None, resume=False, stdin=None):
        archive = root / (archive or ('archive-' + name))
        cmd = common + ['--archive', str(archive), '--thinking', thinking]
        if chat_only:
            cmd += ['--chat-only']
        if resume:
            cmd += ['--resume']
        if prompt is not None:
            cmd += ['--request', prompt]
        started = time.monotonic()
        result = subprocess.run(cmd, input=stdin, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', env=env, timeout=180)
        elapsed = time.monotonic() - started
        output = (result.stdout + '\n' + result.stderr).replace(key, '[redacted]')
        (root / (name + '.txt')).write_text(output, encoding='utf-8')
        assert result.returncode == 0, (name, 'CLI failed; see private evidence log')
        records = [json.loads(f.read_text('utf-8')) for f in archive.rglob('*-turn-*.json')]
        records.sort(key=lambda r: r['body']['sequence'])
        assistants = [r['body'] for r in records if r['body']['role'] == 'assistant']
        assert assistants and not any('[Tina host status:' in r['content'] for r in assistants), (name, 'incomplete generation')
        metric_file = archive / 'last-live-metrics.json'
        metrics = json.loads(metric_file.read_text('utf-8'))
        record = {'case': name, 'status': 'TRANSPORT_COMPLETE', 'wall_seconds': round(elapsed, 3),
                  'last_turn_rounds': metrics['rounds'], 'completion_tokens': metrics['completion_tokens'],
                  'context_source': metrics['context_source']}
        cases.append(record)
        print(json.dumps(record), flush=True)
        return assistants[-1]['content'], archive, output, assistants

    answer, _, output, _ = run('english', 'Reply with exactly TINA_ENGLISH_OK, without quotes or punctuation.')
    assert answer.strip() == 'TINA_ENGLISH_OK'
    assert '[thinking]' not in output, 'off mode unexpectedly produced reasoning'

    answer, _, _, _ = run('chinese', '只回复“中文验收通过”，不要引号，不要其他文字。')
    assert answer.strip() == '中文验收通过'

    answer, archive, _, assistants = run('thinking-tool', 'Use the read tool to read receipt.txt, then return the exact marker from that file. Do not guess or answer before reading it.', thinking='low', chat_only=False)
    assert 'TINA_REAL_TOOL_RECEIPT_COPPER_5821' in answer
    events = [json.loads(f.read_text('utf-8')) for f in archive.rglob('*-agent-*.json')]
    assert any(r['body']['event'].get('type') == 'tool_result' and r['body']['event'].get('status') == 'succeeded' for r in events), 'Missing real successful tool receipt'
    assert all('reasoning_content' in a for a in assistants), 'Reasoning was not persisted'
    assert len(assistants) >= 2, 'Missing model continuation after tool'

    answer, _, _, _ = run('thinking-tool-resume', 'What exact marker did you read from receipt.txt in the previous request? Reply with the marker, using retained history.', thinking='low', chat_only=False, archive='archive-thinking-tool', resume=True)
    assert 'TINA_REAL_TOOL_RECEIPT_COPPER_5821' in answer

    answer, _, _, _ = run('python-code', 'Return only a Python function sum_to(n) that computes 1+2+...+n for nonnegative integer n, using the closed formula and one return statement. No imports, no calls, no annotations, no docstring. No example code.')
    code = answer.strip()
    if code.startswith('```'):
        code = re.sub(r'^```(?:python)?\s*', '', code).rsplit('```', 1)[0].strip()
    tree = ast.parse(code)
    allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
               ast.BinOp, ast.Mult, ast.Add, ast.FloorDiv, ast.Name, ast.Load, ast.Constant)
    assert all(isinstance(node, allowed) for node in ast.walk(tree)), 'Generated code exceeded arithmetic-only acceptance surface'
    assert len(tree.body) == 1 and tree.body[0].name == 'sum_to'
    assert len(tree.body[0].body) == 1 and isinstance(tree.body[0].body[0], ast.Return)
    scope = {'__builtins__': {}}
    exec(compile(tree, '<synthetic-acceptance>', 'exec'), scope)
    assert [scope['sum_to'](n) for n in (0, 1, 10, 100)] == [0, 1, 55, 5050]

    answer, archive, output, _ = run('compact-history', thinking='low', stdin='Remember this project code: IVORY_5821. Reply only ACK_ONE.\nCompute 7*6. Reply only the number.\nReply only READY.\n/compact\nWhat project code did I ask you to remember? Reply only that code.\n/exit\n')
    assert 'IVORY_5821' in answer
    assert any(archive.rglob('*-handoff-*.json')), 'Manual handoff did not create an immutable record'
    assert 'History archived' in output or '已归档' in output

    answer, _, _, _ = run('compact-resume', 'Reply only with the project code that I asked you to remember.', thinking='low', archive='archive-compact-history', resume=True)
    assert 'IVORY_5821' in answer

    for case in cases:
        case['status'] = 'PASS'
    report = {'schema': 'tina.deepseek-live-acceptance.v1', 'model': args.model,
              'endpoint': 'https://api.deepseek.com/v1', 'model_inference': True,
              'synthetic_workspace_only': True, 'cases': cases,
              'status': 'PASS', 'case_count': len(cases),
              'scope': 'Product integration smoke tests; not a capability benchmark or local GPU measurement.'}
    (root / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('PASS: real DeepSeek acceptance', len(cases), 'cases; transcripts remain private.')


if __name__ == '__main__':
    main()
