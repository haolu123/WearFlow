"""Independent aggregation/link audit of the frozen paper artifacts."""
from pathlib import Path
import hashlib, json, re
import numpy as np
import pandas as pd
P = Path(__file__).resolve().parents[1]
summary = pd.read_csv(P/'tables/summary.csv')
checks = []
for task, scope, counts in [('current','current_common128',{0:128}),('future','all_methods_common',{1:100,2:86,3:73})]:
    frame = pd.read_csv(P/f'tables/{task}_predictions.csv')
    frame = frame[frame.in_common.astype(str).str.lower().eq('true')].copy()
    frame['error'] = (frame.pred_mm-frame.target_mm).abs()
    for (method,h), part in frame.groupby(['method','horizon']):
        counts_found = part.groupby('seed').size().unique()
        assert set(counts_found)=={counts[int(h)]}, (method,h,counts_found)
        estimate = part.groupby(['fold','case','seed']).error.mean().groupby(['fold','case']).mean().mean()
        row=summary[(summary.method==method)&(summary.task==task)&(summary.scope==scope)&(summary.horizon==h)&(summary.region=='overall')]
        assert len(row)==1, (method,h,scope)
        assert abs(estimate-row.iloc[0].mae_mm)<1e-12, (method,h,estimate)
        checks.append({'task':task,'method':method,'horizon':int(h),'targets_per_seed':int(counts_found[0]),'independent_mae':float(estimate)})
    for h, hp in frame.groupby('horizon'):
        pools=[]
        for _, mp in hp.groupby(['method','seed']):
            keys=['fold','case','run'] if task=='current' else ['fold','case','origin_run','target_run']
            pools.append(set(map(tuple,mp[keys].to_numpy())))
        assert all(pool==pools[0] for pool in pools), (task,h,'unequal target keys')
manuscript=(P/'MANUSCRIPT.md').read_text(encoding='utf-8')
for match in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',manuscript):
    assert (P/match).is_file(), match
assert len(re.findall(r'!\[Figure ',manuscript))==4
refs=set(re.findall(r'- \*\*\[([^\]]+)\]',manuscript))
citations=set(re.findall(r'(?<!\*)\[([A-Za-z][A-Za-z0-9]+)\](?!\()',manuscript))
assert citations <= refs, citations-refs
assert not re.search(r'\bTODO\b|\bTBD\b|Zhang et al\. \(2020\)',manuscript)
files=[p for p in P.rglob('*') if p.is_file() and p.suffix in {'.md','.py','.csv','.json','.png','.pdf','.svg'} and p.name not in {'qa_manifest.json','QA_REPORT.md'}]
out={'status':'passed','training_started':False,'independently_recomputed_rows':len(checks),'aggregations':checks,'citations':sorted(refs),'figure_count':4,'sha256':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(P/'qa_manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
(P/'QA_REPORT.md').write_text(f'''# Final artifact audit

- Status: passed.
- Independently recomputed {len(checks)} overall common-target MAEs from prediction records, using within-case seed averaging and equal case weights; agreement within 1e-12.
- Verified identical per-method/per-seed target keys: current 128, future h1/h2/h3 100/86/73.
- Checked four embedded figures, local image paths, citation keys and absence of drafting placeholders.
- Source-level protocol and reproduction reviews: TABLE_PROTOCOL.md and evidence/METHODS_CODE_EVIDENCE.md; original-pool reproduction audit: tables/audit.json.
- Figure numerical and visual review: FIGURE_AUDIT.md and figures/audit.json. Root also inspected the framework and case-trajectory raster figures.
- No training was started. Predictions and source experiment files were not modified.
- This artifact audit does not convert the exploratory study into an untouched confirmatory test or validate the uniqueness of instantaneous wear rates.

See qa_manifest.json for exact artifact hashes and independently recomputed values.
''',encoding='utf-8')
print(json.dumps({'status':'passed','rows':len(checks),'figures':4}))
