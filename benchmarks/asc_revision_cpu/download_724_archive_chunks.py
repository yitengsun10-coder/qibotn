"""Read a fixed 724 evidence archive in small SSH chunks after SCP failures.

Key authentication only. This transfers bytes; it does not run benchmarks,
change remote evidence, install software, or confer performance acceptance.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

REMOTE='/root/autodl-tmp/qibotn_chain_resumed_complete_20261009_2254.tar.gz'
PYTHON='/root/autodl-tmp/qibotn_diagnostic_env_20261009/bin/python'
CHUNK=8192


def decode_chunk(encoded,size):
    data=base64.b64decode(encoded.strip(),validate=True)
    if len(data)!=size:raise ValueError('Remote chunk length differs')
    return data


def assemble_chunks(chunks,size,expected):
    data=b''.join(chunks)
    if len(data)!=size or hashlib.sha256(data).hexdigest()!=expected:
        raise ValueError('Assembled transport size/SHA256 differs')
    return data


def download(output,size,expected):
    if output.exists() or output.with_suffix(output.suffix+'.chunks').exists():
        raise FileExistsError('Fresh transfer output required')
    if not 0<size<=256*2**20 or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('Invalid explicit archive size/SHA256')
    folder=output.with_suffix(output.suffix+'.chunks');folder.mkdir(parents=True)
    def fetch(index):
        offset=index*CHUNK;length=min(CHUNK,size-offset);errors=[]
        code=('import base64; f=open('+repr(REMOTE)+',"rb"); '
              'f.seek('+str(offset)+'); print(base64.b64encode(f.read('+str(length)+')).decode("ascii"))')
        command=shlex.quote(PYTHON)+' -c '+shlex.quote(code)
        for attempt in range(3):
            try:
                result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',
                    '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-p','26607',
                    'root@connect.westb.seetacloud.com',command],capture_output=True,timeout=45,check=True)
                raw=decode_chunk(result.stdout,length)
                with (folder/(str(index).zfill(4)+'.bin')).open('xb') as stream:stream.write(raw)
                print('verified transport chunk '+str(index+1)+'/'+str((size+CHUNK-1)//CHUNK),flush=True)
                return raw,{'index':index,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                            'attempts':attempt+1,'prior_errors':errors}
            except (subprocess.SubprocessError,ValueError) as exc:
                errors.append(type(exc).__name__)
                if attempt==2:raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(fetch,range((size+CHUNK-1)//CHUNK)))
    data=assemble_chunks([r[0] for r in results],size,expected)
    with output.open('xb') as stream:stream.write(data)
    receipt={'status':'transport_only','remote_archive':REMOTE,'bytes':size,'sha256':expected,
             'chunks':[r[1] for r in results],'benchmark_restarted':False,'remote_evidence_modified':False}
    with output.with_suffix(output.suffix+'.transfer.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bytes',type=int,required=True)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    result=download(args.output,args.bytes,args.sha256)
    print(json.dumps({k:v for k,v in result.items() if k!='chunks'},indent=2))
