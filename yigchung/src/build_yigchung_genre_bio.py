#!/usr/bin/env python3
"""Build the Yigchung BIO DatasetDict for mmBERT from the genre training parquet files
(same windowing/labeling as sabche/src/build_sabche_dataset.py).
Spans in the parquet have an INCLUSIVE end; label_tokens wants [start, end), so end+1 is passed.
Usage: python build_yigchung_genre_bio.py [NAME] [PARQUET_DIR] [SPLITS]
  NAME         with_lotsawa (default) or without_lotsawa
  PARQUET_DIR  folder holding <split>.parquet (default <local>/genre_training/NAME)
  SPLITS       comma list (default train,validation,test)
Fixed settings: tokenizer jhu-clsp/mmBERT-base; windows of 8192 tokens (8190 text tokens plus CLS and SEP);
stride 5120; labels O=0, B-YIGCHUNG=1, I-YIGCHUNG=2; neighbouring spans must not overlap (asserted).
Classified system; its output is what is uploaded as BDRC/Bo-Yigchung-Classified-Dataset.
Reads:  the parquet files from build_yigchung_genre_datasets.py and the catalog (for a missing category column).
Writes: <local>/genre_training/bio_NAME/ (a DatasetDict) and bio_NAME_stats.json (windows, docs, label counts,
        inverse-frequency class weights, spans without a B token)."""
import sys,json
import paths as P
from collections import Counter
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(P.REPO/'common'))
from build_tsawa_dataset import IGNORE_LABEL,label_tokens,pack_window,sliding_windows,special_token_ids
from datasets import Dataset,DatasetDict,Features,Sequence,Value
from transformers import AutoTokenizer
NAME=sys.argv[1] if len(sys.argv)>1 else 'with_lotsawa'
# optional: argv[2] = folder holding <split>.parquet (default genre_training/<NAME>), argv[3] = comma list of splits
G=P.GENRE
SRC=Path(sys.argv[2]) if len(sys.argv)>2 else G/NAME
SPLITS=sys.argv[3].split(',') if len(sys.argv)>3 else ['train','validation','test']
OUT=G/f'bio_{NAME}';MAXLEN,STRIDE=8192,5120
import csv as _csv
CAT={r['book_id']:r['category'] for r in _csv.DictReader(open(P.CATALOG,encoding='utf-8-sig'))}
LABEL2ID={'O':0,'B-YIGCHUNG':1,'I-YIGCHUNG':2};ID2LABEL={v:k for k,v in LABEL2ID.items()}
tok=AutoTokenizer.from_pretrained('jhu-clsp/mmBERT-base',use_fast=True);tok.model_max_length=int(1e12)
cls_id,sep_id,pad_id=special_token_ids(tok)
out={};counts={};stats_docs={};lost=Counter()
for split in SPLITS:
    d=pd.read_parquet(SRC/f'{split}.parquet')
    if 'category' not in d.columns: d['category']=[CAT.get(b,'lotsawa' if s=='lotsawa' else 'other') for b,s in zip(d.book_id,d.source)];out[split]=[];counts[split]=Counter()
    stats_docs[split]={'docs':len(d),'spans':int(sum(len(s) for s in d.spans)),'by_category':dict(Counter(d.category))}
    for r in d.itertuples():
        sp=sorted((int(s['start']),int(s['end'])+1,s['id'],'YIGCHUNG') for s in r.spans)
        for a,b in zip(sp,sp[1:]):
            assert b[0]>=a[1],(r.book_id,a,b)
        enc=tok(r.text,add_special_tokens=False,return_offsets_mapping=True)
        offs=[tuple(o) for o in enc['offset_mapping']]
        labels=label_tokens(offs,sp,LABEL2ID)
        lost['spans']+=len(sp);lost['spans_without_B_token']+=len(sp)-sum(x==1 for x in labels)
        cov=100.0*sum(e-s for s,e,*_ in sp)/max(len(r.text),1)
        for wi,(ws,we) in enumerate(sliding_windows(len(enc['input_ids']),MAXLEN-2,STRIDE)):
            ex=pack_window(enc['input_ids'],labels,offs,ws,we,cls_id,sep_id,pad_id,MAXLEN)
            ex.update({'pecha_id':r.book_id,'source_batch':r.source,'category':r.category,'window_index':wi,
                       'n_tokens_doc':len(enc['input_ids']),'coverage_pct':round(cov,4)})
            counts[split].update(x for x in ex['labels'] if x!=IGNORE_LABEL);out[split].append(ex)
    print(split,'docs',len(d),'windows',len(out[split]),flush=True)
feats=Features({'input_ids':Sequence(Value('int32')),'attention_mask':Sequence(Value('int8')),'labels':Sequence(Value('int32')),
    'token_start':Value('int32'),'token_end':Value('int32'),'char_start':Value('int32'),'char_end':Value('int32'),
    'pecha_id':Value('string'),'source_batch':Value('string'),'category':Value('string'),'window_index':Value('int32'),
    'n_tokens_doc':Value('int32'),'coverage_pct':Value('float64')})
DatasetDict({s:Dataset.from_list(v,features=feats) for s,v in out.items()}).save_to_disk(str(OUT))
tr=counts.get('train',Counter());tot=sum(tr.values())
stats={'label2id':LABEL2ID,'max_length':MAXLEN,'stride':STRIDE,'tokenizer':'jhu-clsp/mmBERT-base','source':str(SRC),
       'windows':{s:len(v) for s,v in out.items()},'docs':stats_docs,
       'label_counts':{s:{ID2LABEL[k]:v for k,v in sorted(c.items())} for s,c in counts.items()},
       'class_weights_inverse_freq_train':({ID2LABEL[c]:round(tot/(3*tr[c]),4) for c in sorted(tr)} if tr else None),**lost}
(G/f'bio_{NAME}_stats.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2));print('saved',OUT)
