"""Frozen scGPT forward, fixed-token occlusion and replay benchmark."""
import time, json
import numpy as np
import torch
from common import *

class Forward:
    def __init__(self):
        self.k,self.p,self.manifest=dataset()
        self.prep,self.cfg,self.x,self.genes,self.meta,px,pg,pm=raw_data()
        self.model,self.vocab,self.args,self.audit=self.prep.atlas.load_dense(self.cfg)
        self.model.eval().requires_grad_(False)
        self.c=self.prep.cohort(self.x,self.genes,self.meta,self.cfg,self.vocab,self.manifest['classes'])
        self.sae,self.sae_path=load_sae()
        self.gene_col={g:i for i,g in enumerate(self.genes)}
        self.vocab_to_col={self.vocab[g]:i for i,g in enumerate(self.genes) if g in self.vocab}
        self.candidates=json.loads((OUT/'candidates.json').read_text())
        self.baselines=json.loads((OUT/'baseline_directions.json').read_text())
        self.pca=load_npz(HERE/'pca.npz')

    @torch.inference_mode()
    def embedding(self,b):
        h=self.model.encoder(b['genes'])+self.model.value_encoder(b['values'])
        for layer in self.model.transformer_encoder.layers:
            h=layer(h,src_key_padding_mask=b['padding'])
        v=~b['padding']; h=(h*v.unsqueeze(-1)).sum(1)/v.sum(1,keepdim=True)
        assert torch.isfinite(h).all()
        return h

    @torch.inference_mode()
    def scores(self,b):
        h=self.embedding(b); z=self.sae.encode(h)[0]
        out=[z[:,c['feature']] for c in self.candidates]
        for v in self.baselines:
            if v['kind']=='embedding': a=h[:,v['dimension']]
            else:
                mu=torch.as_tensor(self.pca['mean'],dtype=h.dtype)
                d=torch.as_tensor(self.pca['directions'][:,v['dimension']],dtype=h.dtype)
                a=(h-mu)@d
            out.append(v['sign']*a)
        a=torch.stack(out,dim=1).numpy(); assert np.isfinite(a).all(); return a

    def full_bins(self,row):
        raw=self.x[row]; cols=raw.indices; counts=raw.data.astype(np.float64)
        total=counts.sum(); keep=self.c.vocab_ids[cols]>=0; cols,counts=cols[keep],counts[keep]
        norm=np.log1p(counts/total*1e4)
        edges=np.quantile(norm,np.linspace(0,1,self.cfg['n_bins']-1))
        left=np.digitize(norm,edges); right=np.digitize(norm,edges,right=True)
        rng=np.random.default_rng(np.random.SeedSequence([self.cfg['seed'],row,11]))
        ties=rng.random(len(self.genes)); bins=np.ceil(ties[cols]*(right-left)+left).astype(np.float32)
        out=np.zeros(len(self.genes),np.float32); out[cols]=bins
        return out

    def batch(self, base, values):
        n=len(values)
        return {'genes':base['genes'].repeat(n,1),'values':torch.as_tensor(values,dtype=torch.float32),
                'padding':base['padding'].repeat(n,1)}

def main():
    f=Forward(); ix=f.manifest['splits'][0]['indices']['train']; rows=f.k['rows'][ix[:8]]
    b=f.c.tokenize(rows); start=time.time(); h=f.embedding(b).numpy(); elapsed=time.time()-start
    err=float(np.max(np.abs(h-f.k['embedding'][ix[:8]]))); assert err<3e-5
    for i,row in enumerate(rows):
        valid=~b['padding'][i].numpy(); cols=[f.vocab_to_col[int(v)] for v in b['genes'][i][valid]]
        assert np.array_equal(f.full_bins(int(row))[cols],b['values'][i][valid].numpy())
    base=f.c.tokenize(rows[:1]); v=base['values'].numpy().repeat(8,axis=0)
    for i in range(8): v[i,i]=0
    start=time.time(); s=f.scores(f.batch(base,v)); elapsed2=time.time()-start
    no_op=f.scores(base); replay=f.scores(f.batch(base,base['values'].numpy()))
    assert np.max(np.abs(no_op-replay))<1e-6
    result={'status':'PASS','checkpoint_load':f.audit,'embedding_replay_max_error':err,
      'full_bins_matches_tokenizer':True,'no_op_max_error':float(np.max(np.abs(no_op-replay))),
      'seconds_8_original_cells':elapsed,'seconds_8_perturbed_cells':elapsed2,
      'seconds_per_forward':elapsed2/8,'estimated_full_12288_forwards_minutes':12288*elapsed2/8/60,
      'cpu_threads':torch.get_num_threads(),'mps_available':torch.backends.mps.is_available(),
      'intervention':'fixed-token value substitution in binned model input; model sensitivity, not raw-count causal intervention'}
    js(OUT/'forward_preflight.json',result); print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
