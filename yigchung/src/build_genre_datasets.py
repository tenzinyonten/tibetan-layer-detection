"""Build genre training sets in HF format: without_lotsawa (36 genre books) and with_lotsawa (+26 combined Lotsawa docs).
Label: category (4 genres). Source data is not modified.

Classified system. Usage: python yigchung/src/build_genre_datasets.py     (no arguments; rerun whenever the
catalog, span_corrections.csv or span_additions.csv change). Steps, in order:
  1. keep books whose catalog category is not 'other'; split = catalog genre_split (never the HF split)
  2. where annotator_spans/<book_id>.json exists, use the annotator's spans instead of the HF spans
  3. apply span_corrections.csv (drop, or new edges; edges are trimmed of spaces and newlines)
  4. end a span before an unmatched closing ')' (annotator convention: 605 of 652 edits on 19 books)
  5. add span_additions.csv spans (stops if one overlaps an existing span)
  6. add the extra_books/*.json books that are not in the HF set; corrections and additions apply to them too
  7. head marks (the HM set below) are never inside a span: a mark at a span edge is trimmed (a span that is
     only a mark is dropped); a mark after a text ending (shad, or the letters ག / ཀ) inside a span splits it
     in two; a mark inside a sentence is left alone. Pieces shorter than 3 characters are dropped.
  8. with_lotsawa also gets the combined Lotsawa documents, labelled prayer (an assumption: ~90% of Lotsawa
     books are prayers by a rough title check)
Reads:  <local>/hub/*.parquet, <local>/lotsawa_combined.parquet, the catalog, span_corrections.csv,
        span_additions.csv, <local>/annotator_spans/, <local>/extra_books/.
Writes: <local>/genre_training/{without_lotsawa,with_lotsawa}/{train,validation,test}.parquet (HF schema plus
        category, borderline, label_note; span ends inclusive) and <local>/genre_training/headmark_fixes.csv
        (also kept in data/processed/). Prints the counts and a span-text sanity check (mismatches should be 0)."""
import paths as P
import pandas as pd,csv,collections,os
H=str(P.HUB)+'/'
CAT=P.CATALOG
LOT=P.LOTSAWA
OUT=str(P.GENRE)+'/'
os.makedirs(OUT,exist_ok=True)
cat={r['book_id']:r for r in csv.DictReader(open(CAT,encoding='utf-8-sig')) if r['category']!='other'}
hf=pd.concat([pd.read_parquet(H+s+'.parquet') for s in ['train','validation','test']])
g=hf[hf.book_id.isin(cat)].copy()
g['split']=g.book_id.map(lambda b:cat[b]['genre_split'])
g['category']=g.book_id.map(lambda b:cat[b]['category'])
g['borderline']=g.book_id.map(lambda b:cat[b]['borderline']=='yes')
g['label_note']=''
# use annotator-reviewed spans where a reviewed file exists (annotator_spans/<book_id>.json)
import json,glob as _g
ANN={}
for f in _g.glob(str(P.ANNOTATOR_SPANS/'*.json')):
    j=json.load(open(f,encoding='utf-8'));ANN[j['book_id']]=[{'id':x['id'],'start':x['start'],'end':x['end']} for x in j['spans']]
g['spans']=[ANN.get(r.book_id,r.spans) for r in g.itertuples()]
g.loc[g.book_id.isin(ANN),'label_note']='spans from annotator review (THoesam check)'
print('annotator spans used for:',sorted(ANN))
# apply hand corrections (drop / edit spans) from span_corrections.csv
COR={}
for c in csv.DictReader(open(P.CORRECTIONS,encoding='utf-8-sig')):
    COR[(c['book_id'],int(c['start']),int(c['end']))]=c
applied=collections.Counter()
def fix(row):
    out=[]
    for s in row.spans:
        k=(row.book_id,int(s['start']),int(s['end']));c=COR.get(k)
        if c is None or c['action']=='keep': out.append(s);continue
        if c['action']=='drop': applied['dropped']+=1;continue
        a,e=int(c['new_start']),int(c['new_end'])
        while a<e and row.text[a] in ' \n': a+=1
        while e>a and row.text[e] in ' \n': e-=1
        out.append({'id':s['id'],'start':a,'end':e});applied['edited']+=1
    return out
g['spans']=[fix(r) for r in g.itertuples()]
# annotator convention learned from the 19 reviewed BDRC books (605 of 652 edits): a span ends BEFORE an unmatched closing ')'
def trim_paren(row):
    out=[]
    for s in row.spans:
        a,e=int(s['start']),int(s['end']);x=row.text[a:e+1]
        if x.rstrip().endswith(')') and '(' not in x:
            e=a+len(x.rstrip())-2
            while e>a and row.text[e] in ' \n': e-=1
            applied['trailing_paren_trimmed']+=1
        out.append({'id':s['id'],'start':a,'end':e})
    return out
g['spans']=[trim_paren(r) for r in g.itertuples()]
# hand-added spans (span_additions.csv: book_id,start,end inclusive)
import hashlib as _h
ADD=collections.defaultdict(list)
for c in csv.DictReader(open(P.ADDITIONS,encoding='utf-8-sig')):
    ADD[c['book_id']].append({'id':_h.md5(f"{c['book_id']}_{c['start']}_{c['end']}".encode()).hexdigest(),'start':int(c['start']),'end':int(c['end'])})
def add_spans(row):
    out=list(row.spans)
    for n in ADD.get(row.book_id,[]):
        assert not any(s['start']<=n['end'] and s['end']>=n['start'] for s in out),('added span overlaps',row.book_id,n)
        out.append(n);applied['added']+=1
    return sorted(out,key=lambda s:int(s['start']))
g['spans']=[add_spans(r) for r in g.itertuples()]
print('corrections applied:',dict(applied),'of',len(COR),'listed')
lot=pd.read_parquet(LOT)
lot=lot.assign(category='prayer',borderline=False,
               label_note='combined Lotsawa document; labelled prayer because ~90% of Lotsawa books are prayers (rough title check)')
# extra annotator-reviewed books that are not in the HF dataset (extra_books/<book_id>.json)
EX=[json.load(open(f,encoding='utf-8')) for f in sorted(_g.glob(str(P.EXTRA_BOOKS/'*.json')))]
extra=pd.DataFrame(EX) if EX else None
if extra is not None:   # hand corrections / additions also apply to the extra books
    extra['spans']=[fix(r) for r in extra.itertuples()]
    extra['spans']=[add_spans(r) for r in extra.itertuples()]
if EX: print('extra books added:',[(e['book_id'],e['split'],e['category'],len(e['spans'])) for e in EX])
cols=['book_id','split','source','text','spans','category','borderline','label_note']   # first 5 = same order as BDRC/yigchung_detection_v2
base=pd.concat([g[cols],extra[cols]],ignore_index=True) if extra is not None else g[cols]
# head marks (༄༅ ༈ ༁ྃ ...) open the NEXT text, so they are never part of a Yigchung span (user rule, 2026-11-12):
#  - mark at the end or start of a span: trim it (plus the shad/space around it); a span that is only a mark: drop
#  - mark inside a span right after a text ending (shad, or ག/ཀ where the shad is omitted): split into two spans, mark removed
#  - mark inside a sentence (no ending before it): leave the span as it is
import re,hashlib
HM='༁༄༅༆༇༈༉༊࿐࿑࿓࿔'
RUN=re.compile(f'[{HM}][{HM}ྃ༔།༎༑ ་\n]*')
hm_log=[]
def headmarks(df,name):
    new=[]
    for r in df.itertuples():
        out=[]
        for s in r.spans:
            a,e=int(s['start']),int(s['end']);x=r.text[a:e+1]
            if not any(c in HM for c in x): out.append(s);continue
            cuts=[0]
            for m in RUN.finditer(x):
                prev=x[:m.start()].rstrip(' \n')
                at_edge=m.start()==0 or not x[m.end():].strip()
                if at_edge or (prev and prev[-1] in '།༎༑༔གཀ'): cuts+=[m.start(),m.end()]
            cuts.append(len(x))
            pieces=[]
            for i in range(0,len(cuts),2):
                p,q=a+cuts[i],a+cuts[i+1]-1
                while p<=q and r.text[p] in ' \n་': p+=1
                while q>=p and r.text[q] in ' \n': q-=1
                if q-p+1>=3 and r.text[p:q+1].strip('།༎༑༔ \n་'): pieces.append((p,q))
            for j,(p,q) in enumerate(pieces):
                sid=s['id'] if j==0 else hashlib.md5((s['id']+f'_{j}').encode()).hexdigest()
                out.append({'id':sid,'start':p,'end':q})
            act='dropped (mark only)' if not pieces else 'unchanged (mark mid-sentence)' if pieces==[(a,e)] else f'split into {len(pieces)}' if len(pieces)>1 else 'trimmed'
            hm_log.append({'set':name,'book_id':r.book_id,'old_start':a,'old_end':e,'action':act,
                           'new_spans':' '.join(f'{p}-{q}' for p,q in pieces),'old_text':x.replace('\n','⏎')[:300]})
        new.append(out)
    df=df.copy();df['spans']=new;return df
base=headmarks(base,'genre');lot=headmarks(lot,'lotsawa')
pd.DataFrame(hm_log).to_csv(OUT+'headmark_fixes.csv',index=False,encoding='utf-8-sig')
print('head-mark rule:',dict(collections.Counter(h['action'] for h in hm_log)))
import pyarrow as pa,pyarrow.parquet as pq
# same schema as BDRC/yigchung_detection_v2 (spans: struct<id string, start int32, end int32>), extra columns at the end
SCHEMA=pa.schema([('book_id',pa.string()),('split',pa.string()),('source',pa.string()),('text',pa.string()),
    ('spans',pa.list_(pa.struct([('id',pa.string()),('start',pa.int32()),('end',pa.int32())]))),
    ('category',pa.string()),('borderline',pa.bool_()),('label_note',pa.string())])
sets={'without_lotsawa':base,'with_lotsawa':pd.concat([base,lot[cols]],ignore_index=True)}
for name,d in sets.items():
    os.makedirs(OUT+name,exist_ok=True)
    print(f'\n== {name} ==')
    for sp in ['train','validation','test']:
        part=d[d.split==sp].reset_index(drop=True)
        part['spans']=[[{'id':str(x['id']),'start':int(x['start']),'end':int(x['end'])} for x in sorted(v,key=lambda x:int(x['start']))] for v in part.spans]
        pq.write_table(pa.Table.from_pandas(part[cols],schema=SCHEMA,preserve_index=False),f'{OUT}{name}/{sp}.parquet')
        print(f"{sp:10} docs {len(part):3} | spans {sum(len(s) for s in part.spans):5} | 4-class {dict(collections.Counter(part.category))}")
# sanity: every span still matches the HF text
bad=0
for r in g.itertuples():
    if r.book_id in ANN: continue
    src=hf[hf.book_id==r.book_id].iloc[0]; srcs={x['id']:x for x in src.spans}
    for a in r.spans:
        if a['id'] not in srcs: continue   # hand-added span
        b=srcs[a['id']]
        if (r.book_id,int(b['start']),int(b['end'])) in COR: continue   # corrected spans are expected to differ
        if r.text[a['start']:a['end']+1]!=src.text[b['start']:b['end']+1] and r.text[a['start']:a['end']+1]+')'!=src.text[b['start']:b['end']+1].rstrip(): bad+=1
print('\nspan check on genre books, mismatches:',bad)
