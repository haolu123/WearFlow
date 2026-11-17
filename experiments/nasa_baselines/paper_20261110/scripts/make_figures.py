"""Publication figures from frozen prediction tables; never trains or selects models."""

import json,hashlib

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

PAPER=Path(__file__).resolve().parents[1];SUITE=PAPER.parent;OUT=PAPER/'figures';OUT.mkdir(exist_ok=True)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})

LABELS={'fusion_A':'Fusion A','fusion_B':'Fusion B','fusion_C':'Fusion C','fusion_D':'Fusion D','CT100':'Conditions + time','sensor12_100':'Sensor monitor','Kim_TF_raw_ensemble':'Kim: raw Transformer ensemble','Kim_TF_cubic_mean_NASA':'Kim: cubic extrapolation','Sun_PIGPR':'Sun PI-GPR','Zhang_PINN_top12':'Zhang PINN'}

COLORS={'fusion_A':'#0072B2','fusion_B':'#56B4E9','fusion_C':'#009E73','fusion_D':'#D55E00'}

LABELS.update({'gpr':'GPR','svr':'SVR','sun_pigpr_causal_nasa':'Sun et al. PI-GPR','chen2025_pidggcn_causal_calculus_corrected':'Chen PIDGGCN (corrected)','chen2025_pidggcn_causal_paper_printed':'Chen PIDGGCN (printed)',

 'sun2020_lstm_module':'Sun et al. (2020) LSTM','persistence_sensor100':'Persistence','linear2_sensor100':'Two-point linear','linear2_nonnegative_sensor100':'Two-point linear, nonnegative','Kim_TF_raw_ensemble':'Kim et al. (2026) Transformer','Kim_TF_cubic_mean_NASA':'Kim et al. (2026) cubic'})

COLORS.update({'Kim_TF_cubic_mean_NASA':'#333333','persistence_sensor100':'#CC79A7','linear2_sensor100':'#8C6D31','linear2_nonnegative_sensor100':'#B8A26B','sun2020_lstm_module':'#7C3D9D'})

AUDIT={'figure_cases_fixed_before_plot':[3,11,4,10,14],'model_selection_from_test':False,'files':{},'checks':[]}

def label(m):return LABELS.get(m,m.replace('_',' '))

def save(fig,name):

    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=300,bbox_inches='tight')

    plt.close(fig)

def read(path):

    AUDIT['files'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return pd.read_csv(path)

def macro(d,keys):

    q=d.copy();q['ae']=(q.pred_mm-q.target_mm).abs()

    # Mean seeds within each case, then cases; never treat members/seeds as cases.

    return q.groupby(keys+['case','seed']).ae.mean().groupby(keys+['case']).mean().groupby(keys).mean()

def box(ax,xy,w,h,text,color='#eaf1f7'):

    ax.add_patch(FancyBboxPatch(xy,w,h,boxstyle='round,pad=.012',facecolor=color,edgecolor='#667788',lw=1));ax.text(xy[0]+w/2,xy[1]+h/2,text,ha='center',va='center',fontsize=10)

def arrow(ax,a,b,text=None,style='-'):

    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.3,color='#374151',linestyle=style))

    if text:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.018,text,ha='center',fontsize=9)



def framework():

    fig,ax=plt.subplots(figsize=(14,8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')

    ax.text(.02,.96,'(a) Inference: causal propagation and observation correction',fontsize=14,weight='bold')

    box(ax,(.02,.65),.15,.13,'Previous corrected\nwear state');box(ax,(.235,.65),.19,.13,'Neural ODE\n$dVB/dt=r_\\theta(VB,c)\\geq0$\nActual interval $\\Delta t_i$')

    box(ax,(.49,.65),.16,.13,'Correction\n$(1-\\alpha)\\hat{VB}_i^-+\\alpha z_i$')

    box(ax,(.715,.65),.25,.13,'Monotone projection\n$\\hat{VB}_i=\\max(\\hat{VB}_{i-1},\\tilde{VB}_i)$')

    arrow(ax,(.17,.715),(.235,.715));arrow(ax,(.425,.715),(.49,.715),'Prior');arrow(ax,(.65,.715),(.715,.715))

    box(ax,(.44,.83),.26,.065,'Current sensor + conditions + time\nMonitor observation $z_i$', '#e7f3e8');arrow(ax,(.57,.83),(.57,.78))

    ax.text(.02,.57,'Initialize once from the sensor monitor. Invalid sensor: skip correction. No true-VB reset.',fontsize=11)

    ax.text(.02,.51,'Future query: propagate the corrected state over the requested interval; no future sensor is used.',fontsize=11)

    ax.text(.02,.42,'(b) Single-shot training / adaptation: not recurrent fusion training',fontsize=14,weight='bold')

    box(ax,(.03,.18),.19,.13,'Sensor + conditions + time\nMonitor estimate $z_i$','#e7f3e8')

    box(ax,(.33,.18),.22,.13,'ODE integration\n$z_i \\rightarrow \\hat{VB}_{i\\to j}$')

    box(ax,(.67,.18),.29,.13,'Future-state supervision\n$L_{future}=(\\hat{VB}_{i\\to j}-VB_j)^2$','#fff0df')

    arrow(ax,(.22,.245),(.33,.245),'$\\Delta t_{ij}$');arrow(ax,(.55,.245),(.67,.245))

    ax.text(.125,.10,'Current-state supervision $L_{current}$',ha='center',fontsize=10)

    ax.text(.63,.10,'Freeze or update ODE according to the A–D protocol.',ha='center',fontsize=10)

    ax.text(.02,.02,'Correction weight and monotone projection are inference operations; panel (b) does not unroll panel (a).',fontsize=11)

    save(fig,'fig01_framework')



def performance(current,future):

    cm=macro(current,['method']).sort_values();fm=macro(future,['method','horizon'])

    fig,(ax,bx)=plt.subplots(1,2,figsize=(15,6),gridspec_kw={'width_ratios':[1.2,1]})

    ax.barh(np.arange(len(cm)),cm.values,color=[COLORS.get(m,'#8795a1') for m in cm.index]);ax.set_yticks(np.arange(len(cm)),[label(m) for m in cm.index]);ax.invert_yaxis();ax.set_xlabel('Case-macro MAE (mm)');ax.set_title('(a) Current VB: common target records')

    for n,v in enumerate(cm):ax.text(v+.001,n,f'{v:.3f}',va='center',fontsize=8)

    ax.set_xlim(0,cm.max()*1.18)

    for m in fm.index.get_level_values('method').unique():

        s=fm.loc[m];bx.plot(s.index,s.values,'o-' if m.startswith('fusion_') else 's--',label=label(m),color=COLORS.get(m),lw=2 if m.startswith('fusion_') else 1.2,markersize=4)

    bx.set(xlabel='Horizon (subsequent labeled record)',ylabel='Case-macro MAE (mm; log scale)',xticks=[1,2,3]);bx.set_yscale('log');bx.set_title('(b) Future VB: horizon-specific common targets');bx.legend(fontsize=8,loc='upper left');bx.grid(alpha=.2)

    counts=current.groupby(['method','seed']).size().unique().tolist();fc=future.groupby(['method','seed','horizon']).size().reset_index(name='n')

    coverage={int(h):sorted(g.n.unique().tolist()) for h,g in fc.groupby('horizon')}

    fig.text(.5,.015,f'Current targets per method/seed: {counts}; future targets by h: {coverage}.\nRaw-waveform Kim baseline and top-12 feature models have different input/capacity budgets.',ha='center',fontsize=9)

    fig.tight_layout(rect=[0,.09,1,1]);save(fig,'fig02_common_target_performance')

    AUDIT['performance_current_targets_per_seed']=counts;AUDIT['performance_future_targets_per_seed']=coverage

    pd.concat([cm.rename('mae').reset_index().assign(task='current'),fm.rename('mae').reset_index().assign(task='future')]).to_csv(OUT/'fig02_values.csv',index=False)



def cases(current,future):

    chosen=[3,11,4,10,14];methods=['fusion_A','fusion_D'];fig,axs=plt.subplots(5,3,figsize=(15,15))

    delta_rows=[]

    for row,case in enumerate(chosen):

        c=current[current.case==case];f=future[(future.case==case)&(future.horizon==1)];g=future[(future.case==case)&(future.horizon==3)]

        truth=c.groupby('run').target_mm.first();axs[row,0].plot(truth.index,truth.values,'ko-',ms=3,lw=1.4,label='Measured VB')

        for m in methods:

            z=c[c.method==m].groupby('run').pred_mm.agg(['mean','std']);axs[row,0].plot(z.index,z['mean'],'o-',color=COLORS[m],label=label(m),ms=3);axs[row,0].fill_between(z.index,z['mean']-z['std'].fillna(0),z['mean']+z['std'].fillna(0),color=COLORS[m],alpha=.13)

            q=f[f.method==m].copy();q['pred_increment']=q.pred_mm-q.origin_pred_mm;q['true_increment']=q.target_mm-q.origin_target_mm

            z=q.groupby('origin_run')[['pred_increment','true_increment']].mean();axs[row,1].plot(z.index,z.pred_increment,'o-',color=COLORS[m],label=label(m),ms=3)

            if m==methods[0]:axs[row,1].plot(z.index,z.true_increment,'ko--',ms=3,label='Measured increment')

            q['increment_error']=q.pred_increment-q.true_increment;delta_rows.append(q)

            q=g[g.method==m].copy();q['error']=q.pred_mm-q.target_mm;z=q.groupby('origin_run').error.mean();axs[row,2].plot(z.index,z,'o-',color=COLORS[m],label=label(m),ms=3)

        axs[row,2].axhline(0,color='black',lw=.7);axs[row,1].axhline(0,color='gray',lw=.6)

        for col in range(3):axs[row,col].set_xlabel('Original cut/run index' if col==0 else 'Forecast origin run');axs[row,col].grid(alpha=.15)

        axs[row,0].set_ylabel(f'Case {case}\nVB (mm)');axs[row,1].set_ylabel('$\\Delta VB$ over h1 (mm)');axs[row,2].set_ylabel('h3 signed error (mm)')

    axs[0,0].set_title('(a) Current corrected wear');axs[0,1].set_title('(b) Forecast increment, h1');axs[0,2].set_title('(c) Future error, h3')

    for ax in axs[0]:ax.legend(fontsize=8)

    fig.text(.5,.008,'Cases fixed before inspecting the new tables: matched low-feed pairs 3/11 and 4/10, plus case14.\nShading: ±1 SD across five seeds (not a confidence interval). Increment = future prediction − origin corrected estimate.',ha='center',fontsize=10)

    fig.tight_layout(rect=[0,.045,1,1]);save(fig,'fig03_fixed_case_trajectories');pd.concat(delta_rows).to_csv(OUT/'fig03_h1_increment_records.csv',index=False)

    AUDIT['checks'].append('Figure3 delta is future prediction minus origin estimate; true delta uses two measured labels; no oracle rate claim')



def mechanism():

    d=read(SUITE/'internal/results/summary.csv');d=d[d.region=='overall']

    arms=['current_only','local_only','integral_only','both'];names=['Current only','Local consistency','Integral supervision','Both'];colors=['#777777','#CC79A7','#0072B2','#D55E00']

    fig,(ax,bx)=plt.subplots(1,2,figsize=(11,4.8))

    values=[d[(d.arm==a)&(d.task=='current')]['mean'].item() for a in arms];ax.bar(names,values,color=colors);ax.tick_params(axis='x',rotation=15);ax.set(ylabel='Case-macro MAE (mm)',title='(a) Current VB, cold-start mechanism study')

    for n,v in enumerate(values):ax.text(n,v+.002,f'{v:.3f}',ha='center',fontsize=9)

    for a,name,col in zip(arms,names,colors):

        tasks=[('persistence_h' if a=='current_only' else 'h')+str(h) for h in [1,2,3]]

        v=[d[(d.arm==a)&(d.task==t)]['mean'].item() for t in tasks];bx.plot([1,2,3],v,'o-',color=col,label=name+(' (persistence)' if a=='current_only' else ''))

    bx.set(xlabel='Horizon',xticks=[1,2,3],ylabel='Case-macro MAE (mm)',title='(b) Future VB, separate target protocol');bx.legend(fontsize=8);bx.grid(alpha=.2)

    fig.text(.5,.015,'Random shared initialization; 100 full-batch epochs; auxiliary weights fixed at 1.\nLocal consistency is a time partial derivative holding sensor/condition fixed, not the total trajectory derivative.',ha='center',fontsize=9)

    fig.tight_layout(rect=[0,.10,1,1]);save(fig,'fig04_cold_start_mechanism')

    AUDIT['checks'].append('Figure4 sourced only from independent cold-start study; current-only future is persistence, not untrained random ODE')



def main():

    framework();mechanism()

    current=read(PAPER/'tables/current_predictions.csv');future=read(PAPER/'tables/future_predictions.csv')

    current=current[current.in_common].copy();future=future[future.in_common].copy()

    assert set(['fusion_A','fusion_D']).issubset(current.method.unique())

    assert not current.duplicated(['method','fold','seed','case','run']).any()

    assert not future.duplicated(['method','fold','seed','case','origin_run','target_run','horizon']).any()

    assert np.isfinite(current[['pred_mm','target_mm']]).all().all();assert np.isfinite(future[['pred_mm','target_mm','origin_pred_mm','origin_target_mm']]).all().all()

    performance(current,future);cases(current,future)

    plotted=pd.read_csv(OUT/'fig02_values.csv');summary=read(PAPER/'tables/summary.csv');errors=[]

    for r in plotted.itertuples():

        scope='current_common128' if r.task=='current' else 'all_methods_common';h=0 if r.task=='current' else int(r.horizon)

        row=summary[(summary.method==r.method)&(summary.task==r.task)&(summary.scope==scope)&(summary.horizon==h)&(summary.region=='overall')]

        assert len(row)==1;errors.append(abs(r.mae-row.mae_mm.iloc[0]))

    assert max(errors)<1e-12

    increments=pd.read_csv(OUT/'fig03_h1_increment_records.csv')

    delta_error=(increments.pred_increment-increments.true_increment)-((increments.pred_mm-increments.target_mm)-(increments.origin_pred_mm-increments.origin_target_mm))

    assert np.max(abs(delta_error))<1e-12 and (increments.origin_kind=='fused_state').all()

    AUDIT.update(summary_comparison_rows=len(errors),summary_max_error=max(errors),delta_identity_max_error=float(np.max(abs(delta_error))),delta_origin_kind='fused_state',output_formats=['png300dpi','pdf','svg'],optional_oracle_figure=False)

    (PAPER/'figures/audit.json').write_text(json.dumps(AUDIT,indent=2))

if __name__=='__main__':main()

