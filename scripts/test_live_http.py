"""Executable-level live-protocol tests. All model responses here are fixtures, not quality evidence."""
import argparse, concurrent.futures, http.server, json, os, pathlib, subprocess, tempfile, threading


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binary',type=pathlib.Path,default=pathlib.Path('target/debug/tina.exe'));ap.add_argument('--output',type=pathlib.Path);a=ap.parse_args()
    exe=a.binary.resolve();results=[]
    with tempfile.TemporaryDirectory(prefix='tina-http-test-') as tmp:
        root=pathlib.Path(tmp);workspace=root/'workspace';workspace.mkdir();(workspace/'sample.txt').write_text('LIVE_RECEIPT_SENTINEL',encoding='utf-8')
        subprocess.run(['git','init','--quiet',str(workspace)],check=True)
        requests=[]
        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'
            def log_message(self,*_):pass
            def do_POST(self):
                req=json.loads(self.rfile.read(int(self.headers['Content-Length'])));requests.append(req);mode=req['model'];messages=req['messages']
                assert self.path=='/v1/chat/completions'
                assert all('tools' not in m for m in messages)
                if mode=='server-error':
                    body=json.dumps({'error':{'message':'bad key FAKE_PRIVATE_TOKEN'}}).encode();self.send_response(401);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
                tools=[m for m in messages if m['role']=='tool']
                if mode in ('read','write-denied','unknown','length','truncated','invalid-args') and not tools:
                    name='read' if mode in ('read','length','truncated','invalid-args') else ('write_file' if mode=='write-denied' else 'undeclared_action')
                    args={'path':'sample.txt'} if name=='read' else {'path':'must-not-exist.txt','content':'bad','create':True}
                    text=json.dumps(args);text=text if mode!='invalid-args' else '{'
                    pieces=[{'tool_calls':[{'index':0,'id':'call1','type':'function','function':{'name':name,'arguments':text[:4]}}]}, {'tool_calls':[{'index':0,'function':{'arguments':text[4:]}}]}]
                    reason='length' if mode=='length' else 'tool_calls'
                else:
                    if mode=='read':assert tools and 'LIVE_RECEIPT_SENTINEL' in tools[-1]['content']
                    if mode in ('write-denied','unknown'):assert tools and 'denied' in tools[-1]['content'].lower()
                    pieces=[{'reasoning_content':'Checking receipts.'},{'content':'SUCCESS：完成'}];reason='stop'
                frames=[': keepalive\r\n\r\n']
                for i,delta in enumerate(pieces):
                    frames.append('data: '+json.dumps({'choices':[{'index':0,'delta':delta}]},ensure_ascii=False)+('\r\n\r\n' if i==0 else '\n\n'))
                if mode!='truncated':
                    # Deliberately omit usage in handoff test to exercise conservative cold accounting.
                    final={'choices':[{'index':0,'delta':{},'finish_reason':reason}]}
                    if mode!='handoff':final['usage']={'prompt_tokens':1500,'completion_tokens':12}
                    frames.append('data: '+json.dumps(final)+'\n\ndata: [DONE]\n\n')
                body=''.join(frames).encode();self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(body)));self.end_headers()
                # Split inside UTF-8 and JSON to verify transport reconstruction, not just parser units.
                for i in range(0,len(body),7):self.wfile.write(body[i:i+7]);self.wfile.flush()
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();base=f'http://127.0.0.1:{server.server_port}/v1'
        def run(mode,expect=True,input_text=None,extra=(),archive_name=None):
            archive=root/(archive_name or ('archive-'+mode));cmd=[str(exe),'chat','--workspace',str(workspace),'--archive',str(archive),'--base-url',base,'--model',mode,'--thinking','off','--timeout-seconds','8','--no-splash',*extra]
            if input_text is None:cmd+=['--request','Read sample.txt and summarize the result.']
            env={**os.environ,'TINA_HOME':str(root/'config'),'NO_COLOR':'1','TINA_API_KEY':'FAKE_PRIVATE_TOKEN'}
            p=subprocess.run(cmd,input=input_text,capture_output=True,text=True,encoding='utf-8',errors='replace',env=env,timeout=20)
            assert (p.returncode==0)==expect,(mode,p.returncode,p.stdout[-2000:],p.stderr[-1000:])
            assert 'FAKE_PRIVATE_TOKEN' not in p.stdout+p.stderr,(mode,'credential leak')
            assert not (workspace/'must-not-exist.txt').exists()
            results.append({'case':mode,'exit_code':p.returncode,'status':'PASS'});return p
        try:
            for mode in ['read','write-denied','unknown']:assert 'SUCCESS' in run(mode).stdout
            for mode in ['length','truncated','invalid-args','server-error']:run(mode,False)
            run('read',extra=('--resume',),archive_name='archive-read')
            handoff=run('handoff',input_text='a'*4500+'\n'+'b'*4500+'\n'+'c'*4500+'\n/status\n/compact\n/sessions\n/new\n/resume\n/exit\n',extra=('--chat-only','--context-positions','12000','--max-tokens','512'))
            assert handoff.stdout.count('SUCCESS') == 3, handoff.stdout[-3000:]
            assert 'Context handoff' in handoff.stdout and 'exceeds usable input' not in handoff.stdout, handoff.stdout[-3000:]
            # Restore the earlier real receipt in a fresh process; a completed result is never fabricated.
            assert any('LIVE_RECEIPT_SENTINEL' in json.dumps(r,ensure_ascii=False) for r in requests)
            assert any('handoff' in p.name for p in root.rglob('*.json')), 'No persistent handoff record created'
        finally:server.shutdown();server.server_close();thread.join(timeout=3)
    report={'schema':'tina.live-http-fixture-tests.v1','model_inference':False,'cases':results,'count':len(results)}
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
