"""Combine the 469 short Lotsawa books into ~30k-character documents, within each HF split.
Spans are shifted to the new offsets (inclusive end, HF convention) and verified. Source data is not modified.

Classified system. Usage: python yigchung/src/combine_lotsawa.py     (no arguments)
Rules: only books with source == 'lotsawa'; books are shuffled with seed 7 inside each split (train, validation,
test) and packed until a document would exceed 30,000 characters (separator: a blank line). Splits never mix.
Reads:  <local>/hub/{train,validation,test}.parquet (fetch_hub.py).
Writes: <local>/lotsawa_combined.parquet and lotsawa_combined.jsonl (documents LOTSAWA_COMBINED_<split>_<nnn>),
        data/processed/lotsawa_combined_manifest.csv (which book went where, at what offset).
Prints the span-mismatch count (must be 0)."""
import pandas as pd,json,random,csv,collections
import paths as P
H=str(P.HUB)+'/'
OUT=P.LOTSAWA.parent
OUT.mkdir(parents=True,exist_ok=True)
TARGET=30000; SEP='\n\n'
out=[];man=[];bad=0
for split in ['train','validation','test']:
    d=pd.read_parquet(H+split+'.parquet'); d=d[d.source=='lotsawa']
    books=list(d.itertuples()); random.Random(7).shuffle(books)
    groups=[];cur=[];size=0
    for b in books:
        if cur and size+len(b.text)>TARGET: groups.append(cur);cur=[];size=0
        cur.append(b);size+=len(b.text)+len(SEP)
    if cur: groups.append(cur)
    for gi,g in enumerate(groups,1):
        cid=f'LOTSAWA_COMBINED_{split}_{gi:03d}'; text='';spans=[]
        for b in g:
            if text: text+=SEP
            off=len(text)
            for s in b.spans:
                ns={'id':s['id'],'start':int(s['start'])+off,'end':int(s['end'])+off}
                if text[0:0]+ (text+b.text)[ns['start']:ns['end']+1]!=b.text[s['start']:s['end']+1]: bad+=1
                spans.append(ns)
            text+=b.text
            man.append(dict(combined_id=cid,split=split,book_id=b.book_id,offset=off,chars=len(b.text),n_spans=len(b.spans)))
        # final verification against the finished text
        out.append(dict(book_id=cid,split=split,source='lotsawa_combined',text=text,spans=spans,members=[b.book_id for b in g]))
# verify every span text again on the final documents
orig={}
for split in ['train','validation','test']:
    d=pd.read_parquet(H+split+'.parquet')
    for r in d[d.source=='lotsawa'].itertuples(): orig[r.book_id]=r
for m in man:
    o=orig[m['book_id']]; doc=next(x for x in out if x['book_id']==m['combined_id'])
    for s in o.spans:
        if doc['text'][int(s['start'])+m['offset']:int(s['end'])+m['offset']+1]!=o.text[s['start']:s['end']+1]: bad+=1
pd.DataFrame(out).to_parquet(P.LOTSAWA,index=False)
with open(OUT/'lotsawa_combined.jsonl','w',encoding='utf-8') as f:
    for x in out: f.write(json.dumps(x,ensure_ascii=False)+'\n')
w=csv.DictWriter(open(P.PROCESSED/'lotsawa_combined_manifest.csv','w',encoding='utf-8-sig',newline=''),fieldnames=list(man[0]));w.writeheader();w.writerows(man)
c=collections.Counter(x['split'] for x in out)
print('combined documents:',dict(c),'| books used:',len(man),'| spans:',sum(len(x['spans']) for x in out),'| span mismatches:',bad)
import statistics as st
print('chars per document: median',st.median(len(x['text']) for x in out),'min',min(len(x['text']) for x in out),'max',max(len(x['text']) for x in out))
print('books per document: median',st.median(len(x['members']) for x in out))
pos=[s['start']/len(x['text']) for x in out for s in x['spans']]
print('spans in last 15% of their document:',round(100*sum(p>=0.85 for p in pos)/len(pos),1),'% (was 55% when books were separate)')
