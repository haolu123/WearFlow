"""Display inner validation and outer diagnostic curves without selecting on test."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run_selection_comparison import ROOT
from .run_gap_selected import OUT,POOLS

def main():
    dest=ROOT/'epoch_curves';dest.mkdir(exist_ok=True)
    paths={'gap':OUT,'correlation':ROOT/'correlation','hybrid':ROOT/'hybrid'}
    styles={'ct':('black','--'),'gap12':('#df7b20','-'),'gap18':('#df7b20',':'),
            'correlation12':('#2276bb','-'),'correlation18':('#2276bb',':'),
            'hybrid12':('#2d9861','-'),'hybrid18':('#2d9861',':')}
    curves={};rows=[];epochs=np.arange(1,501)
    for fold,pool in POOLS.items():
        for rule,path in paths.items():
            selected=json.loads((path/'selected_epochs.json').read_text())
            for method in ['ct','ct_top12','ct_top18']:
                if method=='ct' and rule!='gap':continue
                label='ct' if method=='ct' else rule+method[-2:]
                inner=np.array([pd.read_csv(path/fold/f'inner_{c}'/'seed0'/method/'history.csv').validation_mae.values for c in pool])
                outer=np.array([pd.read_csv(path/fold/'final'/f'seed{s}'/method/'history.csv').validation_mae.values for s in range(5)])
                chosen=selected[fold+'_'+method]
                assert inner.shape==(7,500) and outer.shape==(5,500)
                assert chosen==int(inner.mean(0).argmin()+1)
                curves[fold,label]=(inner,outer,chosen)
                for i,e in enumerate(epochs):
                    rows.append(dict(fold=fold,method=label,epoch=int(e),inner_mean=float(inner[:,i].mean()),outer_mean=float(outer[:,i].mean()),outer_min=float(outer[:,i].min()),outer_max=float(outer[:,i].max()),selected_epoch=chosen))
    pd.DataFrame(rows).to_csv(dest/'curves.csv',index=False)
    def draw(filename,labels,zoom=False):
        fig,axs=plt.subplots(2,2,figsize=(16,10),layout='constrained')
        for col,fold in enumerate(POOLS):
            for label in labels:
                inner,outer,chosen=curves[fold,label];color,ls=styles[label]
                for row,values in enumerate([inner,outer]):
                    mean=values.mean(0);axs[row,col].plot(epochs,mean,color=color,ls=ls,lw=1.6,label=label)
                    axs[row,col].scatter([chosen],[mean[chosen-1]],s=35,color=color,marker='o' if label=='ct' or label.endswith('12') else 's',zorder=5)
                    if len(labels)<=3 and row==1:axs[row,col].fill_between(epochs,values.min(0),values.max(0),color=color,alpha=.10)
            axs[0,col].set_title(f'{fold}: inner validation (7 holdout cases, seed0)')
            axs[1,col].set_title(f'{fold}: outer test (5 seeds, case-macro MAE)')
        for ax in axs.flat:
            ax.set_xlabel('Epoch');ax.set_ylabel('MAE (mm)');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
            if zoom:ax.set_xlim(20,500);ax.set_ylim(0,.25)
            else:ax.set_xlim(1,500);ax.set_ylim(bottom=0)
        fig.suptitle('Markers: epoch selected by INNER validation only. Outer curves are diagnostic.\n'+('Zoom: epoch20-500, MAE0-0.25; values outside axes omitted.' if zoom else 'Full epoch and MAE range.'),fontsize=13)
        fig.savefig(dest/filename,dpi=140);plt.close(fig)
    labels=list(styles);draw('all_full.png',labels);draw('all_zoom.png',labels,True)
    for rule in paths:draw(rule+'.png',['ct',rule+'12',rule+'18'])
    chosen=pd.DataFrame([dict(fold=f,method=m,selected_epoch=v[2],inner_at_selected=v[0].mean(0)[v[2]-1],outer_at_selected=v[1].mean(0)[v[2]-1],outer_at500=v[1].mean(0)[-1]) for (f,m),v in curves.items()])
    chosen.to_csv(dest/'selected_epoch_points.csv',index=False)
    page='''<!doctype html><meta charset="utf-8"><title>内部/外部 MAE 随 epoch 曲线</title><style>body{font:16px system-ui;margin:30px;line-height:1.7}img{width:100%;max-width:1700px}table{border-collapse:collapse}th,td{padding:7px;border-bottom:1px solid #ddd}</style>
<h1>内部验证、外部测试 MAE 随 epoch 变化</h1>
<p>每张图：左列fold A，右列fold B；上排内部验证，下排外部测试。黑虚线是工况＋时间；蓝色是相关性排序，橙色是差异/变化排序，绿色是混合规则。实线为12维，点线为18维。所有sensor模型都包含工况＋时间。</p>
<p>内部曲线：7次留一case验证的MAE等权平均，seed0。外部曲线：每个模型先计算测试case等权MAE，再平均5个种子；不是平均预测后算误差。单规则图的外部阴影是5种子的最小到最大范围，不是置信区间。标记点是仅由内部验证确定的epoch，不是外部最优点。上下两排来自不同训练划分，不能直接解释为同一个模型的训练/测试间隙。</p>
<p><a href="curves.csv">逐epoch数值</a> · <a href="selected_epoch_points.csv">选中轮数与对应误差</a> · <a href="../index.html">返回实验结果</a></p>
<h2>全部规则：局部放大</h2><p>为便于比较后期差异，横轴20–500，纵轴0–0.25mm；范围外数据未显示，完整曲线在下方。</p><a href="all_zoom.png"><img src="all_zoom.png"></a>
<h2>全部规则：完整范围</h2><a href="all_full.png"><img src="all_full.png"></a>'''
    for rule,title in [('correlation','相关性排序'),('gap','差异／变化排序'),('hybrid','相关性门槛＋差异／变化')]:
        page+=f'<h2>{title}：完整范围及外部种子波动</h2><a href="{rule}.png"><img loading="lazy" src="{rule}.png"></a>'
    page+='<h2>内部选中轮数</h2>'+chosen.round(5).to_html(index=False)
    (dest/'index.html').write_text(page,encoding='utf-8')
    (dest/'audit.json').write_text(json.dumps(dict(curve_rows=len(rows),curves=14,epochs=500,plots=5,inner_selected_epochs_verified=True)),encoding='utf-8')
    print(chosen.to_string(index=False))

if __name__=='__main__':main()
