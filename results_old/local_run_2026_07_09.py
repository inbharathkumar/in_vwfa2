# -*- coding: utf-8 -*-
# Local / headless reproducer for Stage 2 sections 2.2-2.5 (cov4.8_v1, 2026-07-09).
# Set ROOT once (folder containing dataframe/ and results/), then run:  python3 local_run_2026_07_09.py
# Regenerates results/cov_stage2_2026_07_09/ (PNGs, CSVs, labels, pkls) from the data files.
# Parcel distance uses scipy Dijkstra graph-distance (fast, no tvb-gdist needed); the Colab code
# path is unchanged and still consumes your original gdist-based df_distance.
import warnings; warnings.filterwarnings('ignore')
import os, pickle, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import gaussian_kde
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from sklearn.mixture import GaussianMixture
from nilearn.plotting import plot_surf_stat_map, plot_surf_contours
from PIL import Image

ROOT = '/sessions/optimistic-eager-ptolemy/mnt/fable'                 # <-- EDIT THIS
WINNERS = f'{ROOT}/results/sample_colab_results/iteration_winners.pkl' # or your own winners pkl
OUT = f'{ROOT}/results/cov_stage2_2026_07_09'; os.makedirs(OUT, exist_ok=True)
TARGET, TVALS = 'WordvsPER', 'scaled_tval'
ROI_COLS = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#42d4f4','#f032e6','#bfef45','#fabed4','#469990','#dcbeff','#9A6324']
AMP = {1:'aVWFA',2:'cVWFA',3:'pVWFA'}; AMPC=['#e6194b','#3cb44b','#4363d8']
norm01 = lambda a: (np.asarray(a,float)-np.min(a))/(np.max(a)-np.min(a)+1e-10)

# ---------- load + Stage-1/2.1 inputs ----------
df = pd.read_parquet(f'{ROOT}/dataframe/df_tmaps.parquet')
GEOM = pd.read_pickle(f'{ROOT}/dataframe/geometry.pkl'); IW = pd.read_pickle(WINNERS)
SUBJECTS = list(IW.keys())
df['y_bin'] = pd.cut(df['y_pos'], bins=10)
g = df.groupby(['subj','sess','y_bin'], observed=False)[TARGET]
cq = lambda x: x[x>1.65].quantile(0.5) if (x>1.65).any() else 0
df['local_thresh'] = g.transform(cq); df['local_max'] = g.transform('max')
sv = (df[TARGET]-df['local_thresh'])/(df['local_max']-df['local_thresh'])
df[TVALS] = np.where(df[TARGET]>=df['local_thresh'], sv, 0.0)

v2a = dict(zip(*[df.drop_duplicates('vrtx_id')[c] for c in ['vrtx_id','annot_name']]))
comp = []
for s,ws in IW.items():
    for w in ws:
        rl=[v2a[v] for v in w['combo_union_vertices'] if v in v2a]
        for lab,c in Counter(rl).items():
            comp.append({'subj':s,'roi_name':f"R{w['round']}_{w['combo_name']}",'parcellation':lab,'count':c,'percentage':c/len(rl)*100})
DF_COMP = pd.DataFrame(comp)

TP=['G_and_S_occipital_inf','G_temporal_inf','G_oc-temp_lat-fusifor','S_collat_transv_ant','S_collat_transv_post','S_temporal_inf','S_occipital_ant','S_oc-temp_lat']
def edge_graph(v,f):
    e=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[0,2]]]); w=np.linalg.norm(v[e[:,0]]-v[e[:,1]],axis=1); n=len(v)
    return csr_matrix((np.r_[w,w],(np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(n,n))
drows=[]
for s,ws in IW.items():
    vg=GEOM[s]['pial'].astype(np.float64); fg=GEOM[s]['faces'].astype(np.int32); G=edge_graph(vg,fg)
    an=[n.decode() for n in GEOM[s]['annot_names']]; al=GEOM[s]['annot_labels']; pc={}
    for t in TP:
        if t in an:
            pv=np.where(al==an.index(t))[0]
            if len(pv)>0: cc=vg[pv]; pc[t]=int(pv[np.argmin(np.linalg.norm(cc-cc.mean(0),axis=1))])
    tn=list(pc); tg=np.array([pc[t] for t in tn]); srcs=[]; sk=[]
    for w in ws:
        rv=w['combo_union_vertices']; rc=vg[rv]; srcs.append(int(rv[np.argmin(np.linalg.norm(rc-rc.mean(0),axis=1))])); sk.append(f"R{w['round']}_{w['combo_name']}")
    D=dijkstra(G,indices=srcs,directed=False)
    for i,rk in enumerate(sk):
        for tnm,tv in zip(tn,tg): drows.append({'subj':s,'roi_name':rk,'target_parcel':tnm,'min_geodesic_dist':float(D[i,tv])})
DF_DIST=pd.DataFrame(drows)

rm=[]
for s,ws in IW.items():
    sw=df[df.subj==s].groupby('vrtx_id')[TVALS].mean(); pial=GEOM[s]['pial']
    for w in ws:
        rk=f"R{w['round']}_{w['combo_name']}"; verts=np.array(w['combo_union_vertices'])
        cs=DF_COMP[(DF_COMP['subj']==s)&(DF_COMP['roi_name']==rk)]; dom=cs.loc[cs.percentage.idxmax(),'parcellation']
        ds=DF_DIST[(DF_DIST['subj']==s)&(DF_DIST['roi_name']==rk)&(DF_DIST['target_parcel']==dom)]
        rx=pial[verts]; cy=float(rx[np.argmin(np.linalg.norm(rx-rx.mean(0),axis=1)),1])
        rm.append({'subj':s,'round':w['round'],'roi_key':rk,'roi':w['combo_name'],'verts':verts,'centroid_y':cy,
                   'roi_dsc':w['dice'],'roi_pow':float(sw.reindex(verts).mean()),'roi_gdist':float(ds.min_geodesic_dist.iloc[0])})
DM=pd.DataFrame(rm)
for s in SUBJECTS:
    m=DM.subj==s
    for c in ['roi_dsc','roi_pow']:
        v=DM.loc[m,c].values; DM.loc[m,c+'_n']=(v-v.min())/(v.max()-v.min()+1e-10)
    dv=DM.loc[m,'roi_gdist'].values; DM.loc[m,'roi_gdist_n']=1-(dv-dv.min())/(dv.max()-dv.min()+1e-10)
    DM.loc[m,'stg2_aparc_score']=(DM.loc[m,'roi_dsc_n']+DM.loc[m,'roi_pow_n']+DM.loc[m,'roi_gdist_n'])/3
print('inputs ready:',len(DM),'candidate ROIs')

# ---------- render helpers (cropped ventral-OTC submesh) ----------
_SUB={}
def submesh(s):
    if s in _SUB: return _SUB[s]
    co=GEOM[s]['inflated'].astype(np.float64); fa=GEOM[s]['faces']; cu=GEOM[s]['curv']
    sel=np.zeros(co.shape[0],bool); sel[df.query('subj==@s').vrtx_id.unique()]=True
    for _ in range(4): sel[np.unique(fa[sel[fa].any(1)])]=True
    ni=np.where(sel)[0]; im=-np.ones(co.shape[0],int); im[ni]=np.arange(len(ni))
    _SUB[s]=(ni,co[ni],im[fa[sel[fa].all(1)]],cu[ni]); return _SUB[s]
def render(s,vals,title,path,roimap=None,levels=None,colors=None,labels=None,cmap='YlOrRd',thr=0.5,vmax=None,vmin=None):
    ni,sc,sf,scv=submesh(s); vmax=vmax if vmax is not None else float(np.nanmax(vals))
    fig=plot_surf_stat_map((sc,sf),vals[ni],hemi='left',view='ventral',bg_map=scv,cmap=cmap,threshold=thr,
                           vmax=vmax,vmin=vmin,bg_on_data=True,title=title,engine='matplotlib',colorbar=True)
    if roimap is not None:
        plot_surf_contours((sc,sf),roimap[ni],levels=levels,colors=colors,figure=fig,legend=False)
        if labels:
            from matplotlib.lines import Line2D
            fig.axes[0].legend(handles=[Line2D([0],[0],color=colors[i],lw=3,label=labels[i]) for i in range(len(labels))],
                               loc='lower left',fontsize=7,framealpha=0.85)
    fig.savefig(path,dpi=130,bbox_inches='tight'); plt.close(fig)
def montage(paths,path,cols=3):
    ims=[Image.open(p) for p in paths]; w=max(i.width for i in ims); h=max(i.height for i in ims)
    rows=int(np.ceil(len(ims)/cols)); M=Image.new('RGB',(w*cols,h*rows),'white')
    for k,im in enumerate(ims): M.paste(im,((k%cols)*w,(k//cols)*h))
    M.save(path)
def save_label(path,verts,coords):
    with open(path,'w') as f:
        f.write(f"# label\n{len(verts)}\n")
        for v in verts: f.write(f"{v} {coords[v,0]:.4f} {coords[v,1]:.4f} {coords[v,2]:.4f} 1\n")
def fit_gmm(x):
    x=x.reshape(-1,1); best=None
    for k in range(1,min(4,max(1,len(x)//15))+1):
        gm=GaussianMixture(k,covariance_type='full',random_state=0).fit(x); b=gm.bic(x)
        if best is None or b<best[1]: best=(gm,b,k)
    return best[0],best[2]
def adj(s):
    f=GEOM[s]['faces']; n=GEOM[s]['inflated'].shape[0]; e=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[0,2]]])
    return csr_matrix((np.ones(len(e)*2),(np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(n,n))
anat_tab=lambda s: df.query('subj==@s').drop_duplicates('vrtx_id').set_index('vrtx_id')
print('helpers ready')

# ================= 2.2 CORTICAL SURFACE =================
D=f'{OUT}/2.2_cortical_surface'; os.makedirs(f'{D}/labels',exist_ok=True)
MEAS=[('anat_depth','Sulcal Depth (lh.sulc)'),('anat_ct','Thickness (mm)'),('anat_curv','Curvature')]
arows=[]; vtyp={}
for s in SUBJECTS:
    at=anat_tab(s)
    for w in IW[s]:
        v=np.array(w['combo_union_vertices']); dep=at.reindex(v)['anat_depth'].values
        gm,k=fit_gmm(dep); typ=norm01(gm.score_samples(dep.reshape(-1,1)))
        vtyp[(s,w['combo_name'])]=dict(zip(v,typ)); mu=gm.means_.ravel()
        spread=np.median(np.abs(dep-mu[gm.predict(dep.reshape(-1,1))]))
        cy=DM[(DM['subj']==s)&(DM['round']==w['round'])].centroid_y.iloc[0]
        arows.append({'subj':s,'roi':w['combo_name'],'round':w['round'],'n':len(v),'centroid_y':cy,
                      'depth_mean':dep.mean(),'depth_sd':dep.std(),'gmm_k':k,'anat_spread':spread})
ANAT=pd.DataFrame(arows)
for s in SUBJECTS:
    m=ANAT.subj==s; ANAT.loc[m,'roi_depth_score']=1-norm01(ANAT.loc[m,'anat_spread'].values)
ANAT.round(4).to_csv(f'{D}/cluster_table_anat.csv',index=False)
S22={}
for s in SUBJECTS:
    sub=ANAT[ANAT.subj==s].nlargest(3,'roi_depth_score').sort_values('centroid_y',ascending=False)
    pial=GEOM[s]['pial']; S22[s]=[]; os.makedirs(f'{D}/labels/{s}',exist_ok=True)
    for i,(_,r) in enumerate(sub.iterrows(),1):
        w=[x for x in IW[s] if x['combo_name']==r['roi']][0]; v=np.array(w['combo_union_vertices'])
        save_label(f'{D}/labels/{s}/stage2_2_roi_{i}.label',v,pial)
        S22[s].append({'idx':i,'roi':r['roi'],'verts':v,'centroid_y':r['centroid_y'],'score':r['roi_depth_score']})
pickle.dump(S22,open(f'{D}/stage2_2_rois.pkl','wb')); pickle.dump(vtyp,open(f'{D}/vertex_typicality.pkl','wb'))
def order_ap(s): return sorted(IW[s],key=lambda w:-ANAT[(ANAT['subj']==s)&(ANAT['roi']==w['combo_name'])].centroid_y.iloc[0])
for s in SUBJECTS:
    at=anat_tab(s); ws=order_ap(s)
    fig,ax=plt.subplots(1,3,figsize=(13,0.55*len(ws)+1.6))
    for pi,(col,lab) in enumerate(MEAS):
        a=ax[pi]; allv=np.concatenate([at.reindex(w['combo_union_vertices'])[col].values for w in ws]); xs=np.linspace(allv.min(),allv.max(),200)
        for i,w in enumerate(ws[::-1]):
            y0=i*1.0; v=at.reindex(w['combo_union_vertices'])[col].values; c=ROI_COLS[len(ws)-1-i]
            if v.std()>1e-6:
                d=gaussian_kde(v)(xs); d=d/d.max()*1.7; a.fill_between(xs,y0,y0+d,color=c,alpha=0.55,lw=0.8,edgecolor='k')
                gm,_=fit_gmm(v);
                for pk in gm.means_.ravel(): a.plot([pk,pk],[y0,y0+0.4],'k:',lw=0.9)
            a.plot(v,np.full(len(v),y0-0.08)+np.random.uniform(-0.03,0.03,len(v)),'|',color=c,ms=4,alpha=0.5)
            if pi==0: a.text(allv.min(),y0+0.15,w['combo_name'],fontsize=7,fontweight='bold')
        a.set_yticks([]); a.set_xlabel(lab,fontsize=9); a.set_title(lab.split(' (')[0],fontsize=10,fontweight='bold'); a.spines[['left','right','top']].set_visible(False)
    fig.suptitle(f'{s} - Anatomical distributions per candidate ROI (ant->pos, dotted=GMM peaks)',fontsize=11,fontweight='bold')
    fig.tight_layout(); fig.savefig(f'{D}/ridgeline_{s}.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(13,4))
    for pi,(col,lab) in enumerate(MEAS):
        a=ax[pi]; data=[at.reindex(w['combo_union_vertices'])[col].values for w in ws]; p=a.violinplot(data,showmedians=True)
        for j,b in enumerate(p['bodies']): b.set_facecolor(ROI_COLS[j]); b.set_alpha(0.5)
        a.set_xticks(range(1,len(ws)+1)); a.set_xticklabels([w['combo_name'].split('Tmax')[0].replace('Rnd','R') for w in ws],rotation=60,fontsize=6)
        a.set_ylabel(lab,fontsize=9); a.set_title(lab.split(' (')[0],fontsize=10,fontweight='bold')
    fig.suptitle(f'{s} - Anatomical distributions (violin fallback)',fontsize=11,fontweight='bold'); fig.tight_layout(); fig.savefig(f'{D}/violin_{s}.png',dpi=120,bbox_inches='tight'); plt.close(fig)
    raw=df.query('subj==@s').groupby('vrtx_id')[TARGET].mean(); sm=np.zeros(GEOM[s]['inflated'].shape[0]); sm[raw.index.values]=raw.values
    rmap=np.zeros_like(sm,int); labs=[]
    for r in S22[s]: rmap[r['verts']]=r['idx']; labs.append(f"{AMP[r['idx']]} (n={len(r['verts'])},Y={r['centroid_y']:.0f})")
    render(s,sm,f'{s} - Stage 2.2 ROIs (Cortical Surface)',f'{D}/rois_2_2_{s}.png',rmap,[1,2,3],AMPC,labs,vmax=float(raw.max()))
montage([f'{D}/rois_2_2_{s}.png' for s in SUBJECTS],f'{D}/rois_2_2_ALL.png')
print('2.2 done')

# ================= 2.3 NON-TARGET =================
D=f'{OUT}/2.3_nontarget'; os.makedirs(f'{D}/labels',exist_ok=True); NT=['FacesvsNull','LimbsvsNull']; PCTL=90
nrows=[]; vsc={}; thr={}
for s in SUBJECTS:
    gg=df.query('subj==@s').groupby('vrtx_id')[NT].mean()
    ft=np.percentile(gg.FacesvsNull[gg.FacesvsNull>0],PCTL); lt=np.percentile(gg.LimbsvsNull[gg.LimbsvsNull>0],PCTL); thr[s]=(ft,lt)
    rel=np.maximum(gg.FacesvsNull.values/ft,gg.LimbsvsNull.values/lt); sc=np.where(rel>1,0.0,1-np.clip(rel,0,1)); vsc[s]=dict(zip(gg.index.values,sc))
    for w in IW[s]:
        v=np.array(w['combo_union_vertices']); scv=np.array([vsc[s][x] for x in v]); cy=DM[(DM['subj']==s)&(DM['round']==w['round'])].centroid_y.iloc[0]
        nrows.append({'subj':s,'roi':w['combo_name'],'round':w['round'],'n':len(v),'centroid_y':cy,
                      'faces_p90':round(ft,2),'limbs_p90':round(lt,2),'pct_clean':round(100*np.mean(scv>0),1),'roi_nontarget_score':round(scv.mean(),4)})
NTS=pd.DataFrame(nrows); NTS.to_csv(f'{D}/cluster_table_nontarget.csv',index=False); S23={}
for s in SUBJECTS:
    sub=NTS[NTS.subj==s].nlargest(3,'roi_nontarget_score').sort_values('centroid_y',ascending=False); pial=GEOM[s]['pial']; S23[s]=[]; os.makedirs(f'{D}/labels/{s}',exist_ok=True)
    for i,(_,r) in enumerate(sub.iterrows(),1):
        w=[x for x in IW[s] if x['combo_name']==r['roi']][0]; v=np.array(w['combo_union_vertices']); save_label(f'{D}/labels/{s}/stage2_3_roi_{i}.label',v,pial)
        S23[s].append({'idx':i,'roi':r['roi'],'verts':v,'centroid_y':r['centroid_y'],'score':r['roi_nontarget_score']})
pickle.dump({'stage2_3':S23,'vscore':vsc},open(f'{D}/stage2_3_rois.pkl','wb'))
for s in SUBJECTS:
    grp=df.query('subj==@s').groupby('vrtx_id')[[TARGET]+NT].mean(); rv=np.unique(np.concatenate([w['combo_union_vertices'] for w in IW[s]])); ir=grp.index.isin(rv); ft,lt=thr[s]
    fig,ax=plt.subplots(1,2,figsize=(11,5))
    for a,nc,th in zip(ax,NT,[ft,lt]):
        a.scatter(grp[TARGET][~ir],grp[nc][~ir],s=5,c='#ccc',alpha=0.4,label='other bigLOTS'); a.scatter(grp[TARGET][ir],grp[nc][ir],s=11,c='#e6194b',alpha=0.7,label='candidate ROI')
        lim=[min(grp[TARGET].min(),grp[nc].min()),max(grp[TARGET].max(),grp[nc].max())]; a.plot(lim,lim,'k--',lw=0.8,alpha=0.6); a.axhline(th,color='green',ls=':',lw=1.2,label=f'p{PCTL}={th:.1f}')
        a.set_xlabel(f'{TARGET} t'); a.set_ylabel(f'{nc} t'); a.set_title(nc,fontsize=10); a.legend(fontsize=7)
    fig.suptitle(f'{s} - Word vs Non-target selectivity',fontsize=11,fontweight='bold'); fig.tight_layout(); fig.savefig(f'{D}/selectivity_{s}.png',dpi=120,bbox_inches='tight'); plt.close(fig)
    raw=df.query('subj==@s').groupby('vrtx_id')[TARGET].mean(); sm=np.zeros(GEOM[s]['inflated'].shape[0]); sm[raw.index.values]=raw.values; rmap=np.zeros_like(sm,int); labs=[]
    for r in S23[s]: rmap[r['verts']]=r['idx']; labs.append(f"{AMP[r['idx']]} (n={len(r['verts'])},clean={r['score']:.2f})")
    render(s,sm,f'{s} - Stage 2.3 ROIs (Non-target clean)',f'{D}/rois_2_3_{s}.png',rmap,[1,2,3],AMPC,labs,vmax=float(raw.max()))
montage([f'{D}/rois_2_3_{s}.png' for s in SUBJECTS],f'{D}/rois_2_3_ALL.png'); print('2.3 done')

# ================= 2.4 SESSION COUNT =================
D=f'{OUT}/2.4_session_count'; os.makedirs(f'{D}/labels',exist_ok=True); trows=[]; TRIM={}; SCM={}
for s in SUBJECTS:
    A=adj(s); nse=df.query('subj==@s').sess.nunique(); th=nse/2; scmap=np.zeros(GEOM[s]['inflated'].shape[0],int); pial=GEOM[s]['pial']; TRIM[s]=[]; os.makedirs(f'{D}/labels/{s}',exist_ok=True)
    for w in IW[s]:
        cnt=Counter()
        for _,cl in w['clusters'].items():
            for v in cl: cnt[v]+=1
        u=np.array(w['combo_union_vertices']); cv=np.array([cnt[v] for v in u]); scmap[u]=np.maximum(scmap[u],cv); keep=u[cv>th]
        if len(keep)>0:
            nc,lb=connected_components(A[keep][:,keep],directed=False); big=keep[lb==np.bincount(lb).argmax()]
        else: nc,big=0,keep
        TRIM[s].append({'roi':w['combo_name'],'round':w['round'],'orig':u,'trimmed':big}); save_label(f'{D}/labels/{s}/trim_{w["combo_name"]}.label',big,pial)
        trows.append({'subj':s,'roi':w['combo_name'],'n_sessions':nse,'n_orig':len(u),'n_kept':int((cv>th).sum()),'n_trim_contiguous':len(big),'n_fragments':int(nc),'pct_retained':round(100*len(big)/len(u),1)})
    SCM[s]=scmap
pd.DataFrame(trows).to_csv(f'{D}/cluster_table_sessioncount.csv',index=False); pickle.dump(TRIM,open(f'{D}/trimmed_rois.pkl','wb'))
for s in SUBJECTS:
    nse=df.query('subj==@s').sess.nunique(); scmap=SCM[s].astype(float); rmap=np.zeros_like(scmap,int); lv=[]; co=[]
    for i,r in enumerate(TRIM[s],1):
        if len(r['trimmed'])>0: rmap[r['trimmed']]=i; lv.append(i); co.append(ROI_COLS[(i-1)%len(ROI_COLS)])
    render(s,scmap,f'{s} - Vertex session-count (jet) + trimmed cores (>50% of {nse} sessions)',f'{D}/sesscount_{s}.png',rmap,lv,co,None,cmap='jet',thr=0.5,vmax=float(nse),vmin=0)
montage([f'{D}/sesscount_{s}.png' for s in SUBJECTS],f'{D}/sesscount_ALL.png'); print('2.4 done')

# ================= 2.5 COMBINED =================
D=f'{OUT}/2.5_combined'; os.makedirs(f'{D}/labels',exist_ok=True); WA,WD,WN=1.0,1.0,1.0
CB=DM[['subj','roi','round','stg2_aparc_score']].merge(ANAT[['subj','roi','roi_depth_score']],on=['subj','roi']).merge(NTS[['subj','roi','roi_nontarget_score']],on=['subj','roi'])
tl={(s,r['roi']):r['trimmed'] for s in SUBJECTS for r in TRIM[s]}; CB['n_trim']=[len(tl[(s,r)]) for s,r in zip(CB.subj,CB.roi)]; cyt=[]
for s,r in zip(CB.subj,CB.roi):
    v=tl[(s,r)]; pial=GEOM[s]['pial']
    cyt.append(float(pial[v][np.argmin(np.linalg.norm(pial[v]-pial[v].mean(0),axis=1)),1]) if len(v)>0 else np.nan)
CB['centroid_y_trim']=cyt
for s in SUBJECTS:
    m=CB.subj==s; CB.loc[m,'stg2_combined_score']=(WA*norm01(CB.loc[m,'stg2_aparc_score'])+WD*norm01(CB.loc[m,'roi_depth_score'])+WN*norm01(CB.loc[m,'roi_nontarget_score']))/(WA+WD+WN)
CB.round(4).to_csv(f'{D}/cluster_table_combined_ALL.csv',index=False); FIN={}; fr=[]
for s in SUBJECTS:
    sub=CB[(CB.subj==s)&(CB.n_trim>=20)].nlargest(3,'stg2_combined_score').sort_values('centroid_y_trim',ascending=False); pial=GEOM[s]['pial']; FIN[s]=[]; os.makedirs(f'{D}/labels/{s}',exist_ok=True)
    for i,(_,r) in enumerate(sub.iterrows(),1):
        v=tl[(s,r['roi'])]; save_label(f'{D}/labels/{s}/stage2_final_roi_{i}.label',v,pial)
        FIN[s].append({'idx':i,'name':AMP[i],'roi':r['roi'],'verts':v,'centroid_y':r['centroid_y_trim'],'combined':r['stg2_combined_score']})
        fr.append({'subj':s,'final':AMP[i],'roi':r['roi'],'n':len(v),'Y':round(r['centroid_y_trim'],1),'aparc':round(r['stg2_aparc_score'],3),'anat':round(r['roi_depth_score'],3),'nontarget':round(r['roi_nontarget_score'],3),'combined':round(r['stg2_combined_score'],3)})
pickle.dump(FIN,open(f'{D}/stage2_final_rois.pkl','wb')); pd.DataFrame(fr).to_csv(f'{D}/final_selection.csv',index=False)
for s in SUBJECTS:
    raw=df.query('subj==@s').groupby('vrtx_id')[TARGET].mean(); sm=np.zeros(GEOM[s]['inflated'].shape[0]); sm[raw.index.values]=raw.values; rmap=np.zeros_like(sm,int); labs=[]
    for r in FIN[s]: rmap[r['verts']]=r['idx']; labs.append(f"{r['name']} (n={len(r['verts'])},Y={r['centroid_y']:.0f},s={r['combined']:.2f})")
    render(s,sm,f'{s} - Stage 2 FINAL ROIs (combined, session-trimmed)',f'{D}/rois_final_{s}.png',rmap,[1,2,3],AMPC,labs,vmax=float(raw.max()))
montage([f'{D}/rois_final_{s}.png' for s in SUBJECTS],f'{D}/rois_final_ALL.png')
montage([f'{OUT}/2.2_cortical_surface/rois_2_2_sub-01.png',f'{OUT}/2.3_nontarget/rois_2_3_sub-01.png',f'{D}/rois_final_sub-01.png'],f'{D}/factor_comparison_sub-01.png',cols=3)
print('2.5 done'); print(pd.DataFrame(fr).to_string()); print('\nALL SECTIONS COMPLETE ->',OUT)
