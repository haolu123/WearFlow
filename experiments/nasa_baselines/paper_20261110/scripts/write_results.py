"""Generate manuscript Results from frozen table artifacts; no training."""

from pathlib import Path

import pandas as pd



P = Path(__file__).resolve().parents[1]

S = pd.read_csv(P/'tables/summary.csv')

LABELS = {

 'fusion_A':'A: frozen propagation–correction baseline',

 'fusion_B':'B: additional current-only supervision',

 'fusion_C':'C: frozen ODE, adapted monitor',

 'fusion_D':'D: jointly adapted monitor and ODE',

 'CT100':'Conditions–time MLP', 'sensor12_100':'Raw current-feature MLP',

 'gpr':'GPR','svr':'SVR','Zhang_PINN_top12':'Zhang PINN (NASA adaptation)',

 'sun_pigpr_causal_nasa':'Sun PI-GPR (causal adaptation)',

 'chen2025_pidggcn_causal_paper_printed':'Chen PIDGGCN (printed derivative)',

 'chen2025_pidggcn_causal_calculus_corrected':'Chen PIDGGCN (corrected derivative)',

 'Kim_TF_raw_ensemble':'Kim Transformer ensemble',

 'Kim_TF_cubic_mean_NASA':'Kim cubic future-VB extension',

 'sun2020_lstm_module':'Sun et al. (2020) LSTM module',

 'persistence_sensor100':'Persistence', 'linear2_sensor100':'Two-point linear extrapolation',

 'linear2_nonnegative_sensor100':'Nonnegative-slope linear extrapolation'}



def md(df):

 return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(f'{x:.5f}' if isinstance(x,float) else str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])



def current_table():

 q=S[(S.task=='current')&(S.scope=='current_common128')]

 w=q.pivot(index='method',columns='region',values='mae_mm')

 order=['fusion_A','fusion_B','fusion_C','fusion_D','CT100','sensor12_100','gpr','svr','Zhang_PINN_top12','sun_pigpr_causal_nasa','chen2025_pidggcn_causal_paper_printed','chen2025_pidggcn_causal_calculus_corrected','Kim_TF_raw_ensemble']

 w=w.loc[order,['overall','fold_a','fold_b','fold_a_in','fold_a_out']].rename(index=LABELS)

 w.columns=['Overall','Fold A','Fold B','A: VB ≤ 0.74','A: VB > 0.74'];w.index.name='Method';return md(w.reset_index())



def future_table():

 q=S[(S.task=='future')&(S.scope=='all_methods_common')&(S.region=='overall')]

 w=q.pivot(index='method',columns='horizon',values='mae_mm')

 order=['fusion_A','fusion_B','fusion_C','fusion_D','persistence_sensor100','linear2_sensor100','linear2_nonnegative_sensor100','sun2020_lstm_module','Kim_TF_cubic_mean_NASA']

 w=w.loc[order].rename(index=LABELS);w.columns=['h1','h2','h3'];w.index.name='Method';return md(w.reset_index())



def original_table():

 rows=[]

 for m in ['fusion_A','fusion_B','fusion_C','fusion_D']:

  q=S[(S.method==m)&(S.scope=='original_fusion')&(S.region=='overall')]

  row={'Variant':LABELS[m],'Current VB':float(q[q.task=='current'].mae_mm.iloc[0])}

  for h in [1,2,3]:row[f'h{h}']=float(q[(q.task=='future')&(q.horizon==h)].mae_mm.iloc[0])

  rows.append(row)

 return md(pd.DataFrame(rows))



def main():

 text='''# Results



## Evaluation coverage and separation of system outputs



The main comparison evaluates the complete propagation–correction system, not the raw monitoring network. Current-state results use the same 128 labeled targets per initialization seed for every method. These are the original 138 labeled targets with ten labeled initialization records excluded; a case without a labeled first record does not lose its first later label. The forecasts from all compared methods share 100, 86, and 73 origin–target pairs per seed at h1, h2, and h3, respectively, covering 14, 13, and 12 cases. The intersection is determined by causal history availability and finite predictions, not by error magnitude. No method is evaluated on future sensor data.



The original complete-system forecasts cover 114/100/86 pairs per seed. Those results are retained in Table 3 below, while Table 2 uses the external-method intersection. Changing the target pool changes the case-averaged error even when the underlying predictions are identical. Raw-monitor results on all 138 labels and single-shot predictions are component analyses and are not substituted for the complete-system results.



## Current flank-wear estimation



Table 1 reports case-macro MAE in millimetres on the common current-state pool. The proposed jointly adapted system D obtains 0.08981 mm, compared with 0.09176 mm for the frozen complete baseline A and 0.09618 mm for the raw current-feature monitor evaluated on the same 128 targets. The current-only adaptation B has the lowest mean among the complete variants at 0.08850 mm. Thus joint adaptation does not give the best current estimate, despite using future-state supervision.



**Table 1. Current-state MAE on 128 common targets per seed.** The range strata use the measured target VB solely for retrospective analysis; the 0.74-mm boundary is not an input to prediction. Counts and case coverage for each stratum are retained in the table data. Different information inputs and training costs are reported in Methods and the budget table.



'''+current_table()+'''



The deterministic GPR baseline obtains 0.09466 mm on the same pool. D is numerically lower by 0.00485 mm, but the exploratory case-paired 95% interval is [−0.02703, 0.01596] mm. This does not establish a stable current-state advantage over GPR. Relative to the adapted Zhang PINN and the Kim raw-signal ensemble, D has mean differences of −0.02148 and −0.05250 mm, with intervals [−0.04213, −0.00420] and [−0.07771, −0.02904] mm, respectively. These are comparisons of implemented systems with different information budgets, not attribution to integral supervision alone.



## Future-state prediction on common targets



**Table 2. Future-state MAE on all-method common origin–target pairs.** There are 100/86/73 pairs per seed at h1/h2/h3. A horizon is a later labeled record, not a fixed duration; case coverage also changes across horizons. Kim future VB denotes polynomial-curve evaluation, not its native KDE threshold-time output.



'''+future_table()+'''



D obtains 0.09083, 0.09729, and 0.10043 mm at h1, h2, and h3. Its means are below those of the recurrent forecasting module and both linear extrapolation variants at all three horizons. The case-paired differences between D and the recurrent module are −0.04311, −0.04446, and −0.06180 mm; the corresponding exploratory intervals are [−0.06959, −0.01466], [−0.09098, −0.00141], and [−0.12010, −0.01087] mm. The Kim curve extension has much larger long-horizon error, discussed below, but this finding is specific to the adopted early-origin forecasting protocol and original cut coordinate.



Within the complete variants, B gives the lowest h1 mean. D gives the lowest h2 and h3 means. Compared with B, D reduces h3 MAE by 0.03243 mm, or approximately 24.4%, with interval [−0.06093, −0.00700] mm. In contrast, the D–B intervals at h1 and h2 include zero. The D–C h3 difference is −0.01535 mm with interval [−0.03448, 0.00009] mm. This comparison does not clearly separate the additional benefit of updating the rate network from adapting the monitor through a frozen rate network.



## Original complete-system results and adaptation budget



**Table 3. Original complete-system ABCD results, without restricting to the external-method forecast intersection.** Current targets remain 128 per seed; future coverage is 114/100/86 pairs. These values preserve the original experiment and must not be combined with Table 2 baseline values as if their target pools were identical.



'''+original_table()+'''



All variants use the same pretrained monitoring and rate networks as their initial point. B, C, and D receive the same additional 100 optimization epochs, with different update paths. The original complete-system results show the same qualitative pattern as the common-target comparison: current-only adaptation is best for current wear and h1, whereas joint adaptation has the lowest h2 and h3 mean. The original h3 difference between D and C has an interval crossing zero, so the lowest mean is not presented as proof that both networks must always be updated.



## Case trajectories and accumulated wear



The case analysis includes the previously identified condition-matched pairs 3/11 and 4/10, and the larger-wear case 14. These cases were specified for inspection before generating the paper figures; they were not selected by searching for the largest new improvement. The plots show measured wear, causal current estimates, and forecasts from labeled origins, with seed-averaged trajectories for readability. The numerical MAEs are computed from individual-seed errors before averaging and therefore need not equal the error of a plotted mean trajectory.



Case 11 illustrates the distinction between current monitoring and evolution prediction. On the common pools, D has a slightly higher current MAE than A (0.06390 versus 0.05596 mm), while its h3 MAE is substantially lower (0.06377 versus 0.13713 mm). Case 3 provides a counterexample to uniform improvement: its current MAE increases from 0.04811 mm in A to 0.06399 mm in D, and the h1 error also increases. In case 4, GPR has a lower current MAE than D (0.05201 versus 0.09812 mm), even though D forecasts better than the recurrent module. A monitoring ranking therefore does not directly determine a forecasting ranking.



Case 14 remains difficult. D reduces its common-pool h3 MAE relative to B from 0.36598 to 0.22293 mm, but the remaining error is large. In the polynomial baseline, a short estimated history can produce rapidly increasing cubic extrapolation; the archived member-level example from cut 3 to cut 8 predicts wear far above the measured target. The failure affects multiple ensemble members rather than only one outlier. A monotonic fitted trend is not sufficient to guarantee a useful extrapolation magnitude.



The interval plots compare the predicted increment, forecast minus its estimated origin state, with the measured target-minus-origin increment. The signed forecast error is decomposed into initial-state error and increment error on identical keys. These quantities diagnose whether apparent forecast gains coincide with better cumulative evolution or with compensation between errors. They do not independently identify the true instantaneous rate, and the estimated-origin increment is not an oracle-rate measurement.



## Mechanism ablation with matched random initialization



The cold-start experiment separates future-state supervision from the extra pretrained optimization budget. Each arm starts from the same monitoring and rate initialization within a split and seed and is trained for 100 epochs. It compares current-only supervision, local derivative consistency, integral future-state supervision, and their combination. This is a separate protocol with 138 current labels and 124/110/96 future pairs per seed; it does not reuse the Table 2 intersection or its fusion outputs.



**Table 4. Cold-start mechanism ablation, case-macro MAE in millimetres.** Current-only future prediction uses persistence because its rate network is not trained. All other future columns use the learned rate integral. All coefficients are fixed, not selected on the test results.



| Supervision | Current VB | h1 | h2 | h3 |

|---|---:|---:|---:|---:|

| Current only | 0.09329 | 0.11370 | 0.18148 | 0.25086 |

| Current + local derivative | 0.09561 | 0.10877 | 0.15565 | 0.20211 |

| Current + integral evolution | 0.10006 | 0.08832 | 0.08915 | 0.10921 |

| Current + local + integral | 0.10252 | 0.09280 | 0.11182 | 0.14817 |



Integral-only supervision has lower future errors than local-only supervision by 0.02045, 0.06650, and 0.09290 mm. The respective exploratory intervals are [−0.03651, −0.00785], [−0.09554, −0.03970], and [−0.13060, −0.05686] mm. It does not improve current-state MAE. Adding the local loss to the integral loss is also not beneficial in this fixed configuration. This is evidence for the value of future-state supervision in the implemented propagation model, not evidence that every derivative-based physical constraint is ineffective.



## Measured-initial-state diagnostic and computation



With measured VB at the forecast origin, A, B, and C use the same unchanged rate network and have identical oracle propagation errors. At h3, D reduces the original oracle MAE from 0.12341 to 0.10846 mm. This diagnostic supports an improvement in propagation independent of the monitor's origin error, but it is evaluated on the original oracle pool, not Table 2's common subset. It is not an operational forecast or a guaranteed lower bound on estimated-state prediction error.



All formal results use their specified final epoch; recorded test curves do not select an epoch. The Kim study contains 100 trained members, with cumulative training, evaluation, and checkpoint-writing time of 41.82 hours and a file-timestamp wall-clock span of 42.21 hours. Its all-epoch derived reports required 13.69 minutes. The two PIDGGCN derivative variants required about 2 hours 21 minutes for 48 runs. The ODE's 500-epoch pretraining and the inherited validation models used to choose fusion coefficients are additional costs, not hidden within the 100-epoch adaptation budget. Timing measurements have different scopes and are not a controlled hardware-speed comparison.



Training archives and report reconstruction were audited separately. The Kim archive audit checked 10,100 saved prediction archives and recomputed recorded monitoring metrics; the derived-report audit checked 1,000 report markers, 3,000 CSV files, input hashes, state targets, interval metadata, and ensemble means. These checks validate data handling and reproducibility of the saved outputs; they do not eliminate the adaptation and statistical limitations of the study.

'''

 (P/'RESULTS_DRAFT.md').write_text(text,encoding='utf-8')

 print('Results generated from frozen summaries.')



if __name__=='__main__':main()

