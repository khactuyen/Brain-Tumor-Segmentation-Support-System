import json
p=json.load(open(r'D:\SIC_Capstone 2026\SIC_Capstone_v2.ipynb',encoding='utf-8'))
print('cells',len(p['cells']))
for i,c in enumerate(p['cells']):
    t=''.join(c.get('source',[]))
    o=''.join(''.join(x.get('text',[])) for x in c.get('outputs',[]) if isinstance(x,dict))
    if o and any(k in (t+o).lower() for k in ['dice','hd95','mean dice','selected model','split','loaded','dataset']):
        print('CELL',i+1, o[:2000].replace('\n',' | '))
