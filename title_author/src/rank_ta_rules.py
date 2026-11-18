#!/usr/bin/env python3
"""Fit author weights on TRAIN with a per-work softmax (picks one candidate per work). Writes ta_rule_weights.json."""
import sys, json, numpy as np
sys.path.insert(0,"/Users/tenzinyonten/layer_detection/title_author/src")
import ta_rule_extract as T
def groups(split,n,seed):
    G=[]
    for r in T.load(split,n,seed).itertuples():
        head,tail=T.region_text(r); g_reg,g=T.gold_span(r,"author")
        cs=[]
        for region,text in [("head",head)]+([("tail",tail)] if tail else []):
            for f,i,name in T.author_candidates_v2(text,region,r.whole_work):
                cs.append((f,int(region==g_reg and T.iou((i,i+len(name)),g)>=0.5)))
        G.append(cs)
    return G
tr=groups("train",6000,1); va=groups("validation",0,0)
names=sorted(tr[0][0][0] if tr[0] else next(c for g in tr for c in g)[0])
def mat(G):
    out=[]
    for cs in G:
        if cs: out.append((np.array([[f[k] for k in names] for f,_ in cs]),np.array([y for _,y in cs])))
        else: out.append(None)
    return out
TR,VA=mat(tr),mat(va)
print("oracle train",np.mean([1 if g is not None and g[1].any() else 0 for g in TR]),"val",np.mean([1 if g is not None and g[1].any() else 0 for g in VA]))
# conditional logit, full-batch Adam
Xall=np.vstack([g[0] for g in TR if g is not None]); mu=Xall.mean(0); sd=Xall.std(0)+1e-6
mu[names.index("bias")]=0; sd[names.index("bias")]=1
w=np.zeros(len(names)); m=np.zeros_like(w); v=np.zeros_like(w)
fit=[(((X-mu)/sd),y) for X,y in [g for g in TR if g is not None and g[1].any()]]
for it in range(1,301):
    grad=np.zeros_like(w); 
    for X,y in fit:
        s=X@w; s-=s.max(); p=np.exp(s); p/=p.sum()
        grad+=X.T@(p-y/y.sum())
    grad=grad/len(fit)+1e-3*w
    m=.9*m+.1*grad; v=.999*v+.001*grad**2
    w-=0.05*(m/(1-.9**it))/(np.sqrt(v/(1-.999**it))+1e-8)
def acc(G):
    ok=0
    for g in G:
        if g is None: continue
        X,y=g; ok+=y[int(np.argmax(((X-mu)/sd)@w))]
    return ok/len(G)
print("top1 train",acc(TR),"val",acc(VA))
W={k:float(c/s) for k,c,s in zip(names,w,sd)}
W["bias"]=float(w[names.index("bias")]-sum(c*m_/s for c,m_,s,k in zip(w,mu,sd,names) if k!="bias"))
json.dump({k:round(x,4) for k,x in W.items()},open("/Users/tenzinyonten/layer_detection/title_author/src/ta_rule_weights.json","w"),indent=1)
