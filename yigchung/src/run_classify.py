import paths as P
import csv,json,os,sys,time,random,urllib.request,urllib.error
KEYS=[k for k in os.environ.get('GEMINI_KEYS','').split(',') if k]
if not KEYS: sys.exit('Set GEMINI_KEYS (comma-separated) in the environment.')
# argv[1]: 0 = all books, N = stratified sample of 30, or a comma-separated list of book_ids
ARG=sys.argv[1] if len(sys.argv)>1 else '0'
IDS=[x for x in ARG.split(',') if x] if not ARG.isdigit() else None
N=int(ARG) if ARG.isdigit() else 0
OUT=sys.argv[2] if len(sys.argv)>2 else 'classify_all_v5.jsonl'
MODEL=os.environ.get('GEMINI_MODEL','gemini-3.1-flash-lite')
T=open(os.environ.get('PROMPT_FILE',str(P.REPO/'yigchung'/'prompts'/'prompt_classify_v5b.txt')),encoding='utf-8').read()
rows=list(csv.DictReader(open(os.environ.get('CATALOG',str(P.CATALOG)),encoding='utf-8-sig')))
if IDS:
    rows=[r for r in rows if r['book_id'] in IDS]
elif N:
    random.seed(5); by={}
    for r in rows: by.setdefault(r['source'],[]).append(r)
    q={'reviewed_opf':12,'bdrc':12,'nalanda':6}
    rows=[r for s,k in q.items() for r in random.sample(by[s],k)]
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        x=json.loads(l)
        if x.get('http')==200: done.add(x['book_id'])
ki=0;fails=0;retries=0
def call(prompt):
    global ki,retries
    body=json.dumps({'contents':[{'parts':[{'text':prompt}]}],'generationConfig':{'responseMimeType':'application/json','temperature':0}}).encode()
    for attempt in range(9):
        url=f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent'
        req=urllib.request.Request(url,body,{'Content-Type':'application/json','x-goog-api-key':KEYS[ki]})
        try:
            r=json.load(urllib.request.urlopen(req,timeout=90)); return 200,r
        except urllib.error.HTTPError as e:
            retries+=1
            if e.code==429: ki=(ki+1)%len(KEYS); time.sleep(2)
            elif e.code in(401,403):
                bad=KEYS.pop(ki)
                if not KEYS: return e.code,'all keys rejected'
                ki%=len(KEYS)
            elif e.code>=500: time.sleep(3)
            else: return e.code,e.read().decode()[:300]
        except Exception as e:
            retries+=1; time.sleep(3)
    return 0,'gave up'
with open(OUT,'a',encoding='utf-8') as f:
    for r in rows:
        if r['book_id'] in done: continue
        p=T
        for k in ['book_id','source','title_tibetan','title_english','author','verse_like_pct','opening_text','middle_text','ending_text']: p=p.replace('{'+k+'}',(r.get(k) or '(none)'))
        code,res=call(p); rec={'book_id':r['book_id'],'source':r['source'],'http':code}
        if code==200:
            try:
                u=res.get('usageMetadata',{});rec['tokens_in']=u.get('promptTokenCount');rec['tokens_out']=u.get('candidatesTokenCount')
                rec['out']=json.loads(res['candidates'][0]['content']['parts'][0]['text']); fails=0
            except Exception as e: rec['http']=-1;rec['err']=str(e)[:200];fails+=1
        else: rec['err']=str(res)[:300];fails+=1
        f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
        if fails>=12: print('stopping: 12 failures in a row');break
print('retries (429/5xx/network):',retries,'key index at end:',ki)
