"""Assemble the evidence-backed paper; never starts training."""
from pathlib import Path
import re

P = Path(__file__).resolve().parents[1]
read = lambda name: (P / name).read_text(encoding='utf-8-sig')
intro, discussion = read('INTRODUCTION_DISCUSSION_DRAFT.md').split('# Discussion', 1)
methods = '## Problem formulation' + read('METHODS_DRAFT.md').split('## Problem formulation', 1)[1]
methods = re.sub(r'\*\*Editorial unit note:\*\*[^\n]+\n\n', '', methods)
methods = methods.replace('For each case, deployment starts', 'In the following deployment equations, m, p and q denote standardized wear, and Phi is the standardized-state integrator; predictions are converted back to millimetres for scoring. For each case, deployment starts')
captions = re.findall(r'\*\*Suggested English caption\.\*\* (.*)', read('FIGURE_CAPTIONS.md'))
figures = ['fig01_framework','fig02_common_target_performance','fig03_fixed_case_trajectories','fig04_cold_start_mechanism']
def figure(i):
    return '\n\n![Figure %d](figures/%s.png)\n\n**Figure %d.** %s\n\n' % (i+1, figures[i], i+1, captions[i])
methods = methods.replace('## Training and controlled adaptation', figure(0) + '## Training and controlled adaptation')
results = read('RESULTS_DRAFT.md')
results = results.replace('## Original complete-system results', figure(1) + '## Original complete-system results')
results = results.replace('## Mechanism ablation with matched random initialization', figure(2) + '## Mechanism ablation with matched random initialization')
results = results.replace('## Measured-initial-state diagnostic', figure(3) + '## Measured-initial-state diagnostic')
refs = '\n'.join(line for line in read('REFERENCES.md').splitlines() if line.startswith('- **['))
# Keep the audited reproduction disclosures with references in this review draft.
abstract = '''# Learning condition-dependent tool-wear evolution through integral supervision of a physical-state Neural ODE

*Evidence-backed manuscript draft, 10 November 2026. Author details and journal formatting are not assigned.*

## Abstract

Sensor-based tool-wear monitoring and future-state forecasting require both an informative current-state estimate and a transition law that explains accumulated wear. Local derivative consistency alone does not directly constrain the wear accumulated over a finite machining interval. We couple a sensor-based monitor with a condition-dependent Neural ODE whose scalar state is flank wear and whose nonnegative rate is supervised through integrated future-state predictions. Single-shot joint adaptation updates the monitor and dynamics; causal deployment subsequently combines propagation with current observations and monotone correction. Experiments on the NASA milling dataset use case-level outer splits, training-only feature selection, and five final seeds. Complete-system comparisons are distinguished from component predictions and evaluated on explicitly matched targets. On the common current-wear pool, joint adaptation achieves a case-macro MAE of 0.08981 mm, compared with 0.09466 mm for Gaussian process regression; their paired case-bootstrap difference includes zero. On common future targets, joint adaptation achieves 0.09083, 0.09729, and 0.10043 mm at the first three subsequent labeled records. At the third horizon, its error is 24.4% lower than an equal-adaptation-budget current-supervision control. A separate cold-start ablation favors integral supervision over the tested local derivative constraint for future prediction, while not improving current estimation. Case analyses expose both reduced long-horizon error and persistent initial-state and cross-case limitations. These exploratory results support finite-interval evolution supervision as a forecasting mechanism, without establishing unique instantaneous-rate recovery or universal monitoring superiority.

**Keywords:** tool wear; Neural ODE; integral supervision; condition-dependent dynamics; sensor monitoring; manufacturing prognostics.

'''
conclusion = '''# Conclusion

This study couples sensor-based current-wear estimation with a nonnegative condition-dependent rate model that actually propagates the physical wear state. Future-state supervision constrains the accumulated effect of that rate over measured time intervals. The complete-system evaluation and the separate mechanism ablation support improved longer-horizon prediction under the tested protocols. They do not show that joint training always improves current-state estimation: continuing current-only supervision remains competitive, and the current-wear advantage over GPR is uncertain at the case level. The observed benefits therefore concern supervised finite-interval evolution and its interaction with state estimation, rather than recovery of a unique physical wear law. Independent confirmation across additional cases, operating conditions and time intervals remains necessary before deployment claims.

'''
text = abstract + intro + '\n# Methods\n\n' + methods + '\n' + results + '\n# Discussion' + discussion + '\n' + conclusion + '# References and reproduction disclosures\n\n' + refs + '\n'
(P/'MANUSCRIPT.md').write_text(text, encoding='utf-8')
print(f'Wrote MANUSCRIPT.md: {len(text.split())} whitespace-delimited words')
