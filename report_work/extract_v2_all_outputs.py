import json
p=json.load(open(r'D:\SIC_Capstone 2026\SIC_Capstone_v2.ipynb',encoding='utf-8'))
for i,c in enumerate(p['cells']):
    for o in c.get('outputs',[]):
        if 'data' in o:
            for k,v in o['data'].items():
                if k in ('text/plain','text/markdown'):
                    s=''.join(v) if isinstance(v,list) else str(v)
                    if any(q in s.lower() for q in ['dice','hd95','mean','validation','swin']): print('CELL',i+1,k,s[:4000])
