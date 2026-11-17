> Archived appendix-writing evidence note. Filenames mentioned below refer to the separate appendix source-data package; not all are distributed in this code-only repository. See ../REPRODUCING.md for executable commands.

# Appendix B — 实现与对照配置



## B.1 NASA 特征、归一化和划分



两个电流通道各25维：9个时域统计＋16个PSD特征，共50个候选，**两通道、时域／频域合计选12**，不是每类12个。每个训练case内计算有标签有效记录上 |Spearman(feature,VB)|，各训练case等权平均；常量或不足3点记0；并列按feature名称排序。外层7训练case，内层留1验证、6训练；验证／测试标签不参加筛选。



时域9项：mean，population std，RMS，min，max，peak-to-peak，mean(z³) skewness，mean(z⁴) kurtosis，mean(x²) energy。注意NASA kurtosis不是excess，energy是均方幅值而非总能量，且等于RMS²。



PSD：fs=250 Hz；Welch Hann，nperseg=1024、noverlap=512、nfft=1024、constant detrend、density、mean averaging。频带[0,5)、[5,15)、[15,30)、[30,60)、[60,125] Hz，各输出绝对功率和总PSD功率比例；另6项为AC power（分段去均值后所有PSD频点积分，未手工剔除DC频点）、centroid、bandwidth、normalized entropy、flatness、peak frequency。完整名称与精确公式以 `NASA_feature_dictionary.csv` 和 `NASA_execution_source/scripts/inspect_psd_features.py` 为准。centroid／bandwidth与非DC entropy／flatness的权重定义不同，不能笼统写所有指标均排除DC。



选中维度在有效训练记录上求1%／99%分位并clip，再拟合mean/std。允许无VB标签的有效训练sensor参与这个无监督scaler步骤，排名则需VB标签。DOC/feed标准化并拼接2个material指示变量；累计加工时间除以训练h3正时间间隔中位数T。VB用有标签训练集mean/std变为z。网络rate是在z与τ=t/T单位，物理速率为 (sigma_VB/T)r。无效记录不进入sensor scaler。



`NASA_split_parameters/` 的16份JSON是从实际checkpoint导出的完整配置和scaler；不要仅复用foldA的12维名单给foldB／inner。文件中的stored_checkpoint_epoch=500只表示读取参数档案的末轮checkpoint，warm A/D实际monitor采用同运行的epoch100权重，见ABCD planned_tasks。



## B.2 网络与预算



|实验|monitor|rate|训练及部署|

|---|---|---|---|

|NASA monitor|17→64→32→1，SiLU隐藏激活|—|AdamW lr.001、wd.0001；固定epoch100用于论文warm基线|

|NASA独立ODE|—|5→32→32→1，SiLU，softplus速率|独立500轮、训练实测VB初值；精确优化配置见ODE checkpoint metadata|

|Warm A|冻结monitor100|冻结ODE500|无额外适配；部署固定alpha＋截断|

|Warm B|再100轮当前监督|保持原ODE|AdamW lr.0001、wd.0001|

|Warm C|再100轮当前＋未来监督|冻结，但允许未来梯度穿过ODE到monitor|当前和未来MSE权重均1|

|Warm D|再100轮当前＋未来监督|与monitor一起更新|同C损失和预算；不是递推展开训练|

|Cold四臂|共同随机初始化|共同随机初始化|100轮full-batch AdamW lr.001、wd.0001；当前／local／future存在时均权重1|

|Stiehl Local/Integral|13→64→32→1，SiLU|1→32→32→1，SiLU＋softplus|随机初始化100轮，AdamW lr.001、wd.0001；训练损失按刀具等权|



积分均固定20步RK4，在每个请求时间区间均匀分20子步；区间保持工况固定，直接autograd穿过积分，不用adjoint。只预测未来时不使用未来sensor。局部约束是固定sensor及condition的时间偏导，不是沿sensor轨迹全导数。



NASA 2fold、每fold7次inner seed0＋5个final seed（0、1、2、3、4）；warm本次ABCD final B/C/D合计30次适配，不应误写为该批也训练了14inner适配。alpha复用前阶段内层选择并固定。cold四臂合计96次；Stiehl九个LOTO，每个fixed inner与5final seeds、四个网络臂，共216次；另18次GPR拟合。权重取固定末轮，不从测试选择epoch。所有CT后备及ODE预训练成本均须披露。



## B.3 外部／标准对照（建议一张表）



|方法|输入和任务|配置／选择|必须披露的适配|

|---|---|---|---|

|GPR|共同17维，当前VB|Constant×RBF/Matérn1.5/2.5＋White；LML，inner选kernel；A Matérn1.5、B RBF|确定性方法，不虚构5次独立训练；完整参数见standard/results|

|SVR|共同17维，当前VB|RBF；C .1/1/10/100，epsilon .01/.1，gamma scale/.01；inner选|A C10 eps.01 gamma.01；B C1 eps.01 gamma.01|

|Sun2020 LSTM forecasting module|过去2个monitor估计VB→未来3步|LSTM64/32编码、Dense32、重复3、LSTM32/16解码、Dense1；Adam.001、logcosh、batch16、100轮|仅forecasting module，不是完整ResNet／校正系统；训练用训练真实历史VB，部署历史值来自monitor；不是Sun2024 PI-GPR|

|Zhang PINN|共同17维，当前VB|solution32编码与sin predictor，dynamicalF 4×50；alpha1 beta.75；100轮|NASA输入适配；保留作者导数实现契约；没有复现ARIMA future模块，未来记N/A|

|Sun2024 PI-GPR|12候选中训练Pearson选HI，工况／时间为GP附加协变量|因果Gaussian历史平滑7点 sigma.5；训练OLS物理均值＋Matérn3/2残差；3375网格内层选择；两fold均sigma_f=.15,sigma_n=.1,l=1.5|原文居中平滑改因果；共享训练物理均值，不用测试工具标签拟合；物理均值并非wear-rate field|

|Chen2025 PIDGGCN|17维与同case历史prefix；当前VB|feature graph、gated64/64、head64/32/1、dropout.5；RMSprop.001，100轮；printed/corrected两版固定|图可见范围和时间偏导显式因果化；物理先验拟合训练轨迹；原文导数印刷式与解析导数不同，两版并列不测试择优|

|Kim Transformer+cubic+KDE|六通道raw窗口，当前估计与历史外推|window300/hop50、embedding20、4head、5layer、FF2048；FP32 batch128；每ensemble10成员×100轮，2fold×5ensemble|不同输入／容量／成本；原输出是阈值时间KDE众数，未来VB是单调无截距三次曲线求值均值扩展；原run/cut坐标，不重置origin，≥3历史点且rank足够|

|Linear history extrapolation|两个历史monitor估计＋实际dt|raw slope与nonnegative slope两版，无新训练|任务扩展，过去估计值不是测试真实VB；较短历史不具覆盖|



详细公式、文献页码、未说明的实现选择和推断信息范围见各 `comparators` 文件。所有适配都应在论文注明；不能把缺少作者细节的自行决策写成完全逐项原样复现。Kim失败成员／NaN覆盖、负值／极端外推没有按结果删除；共同池按有限预测的交集定义。



## B.4 Stiehl独立协议



每个Fx/Fy/Fz通道17维：mean/std/RMS/ptp/median/IQR/skew/**excess kurtosis**/crest，加6个带功率[0,250)、[250,500)、[500,1000)、[1000,2000)、[2000,5000)、[5000,10000)Hz、centroid、entropy。用信号中央60%、Hann Welch2048／重叠1024；实际时间戳推采样率约25kHz，量化修正见feature_audit_timestamp_quantization_v1.json，不误用NASA250Hz。



九刀LOTO；内层预指定同机床刀具作为验证，剩7刀拟合，final用其余8刀；这不是大网格nested搜索。top12训练内选择，1/99clip＋zscore；时间除1000，磨损zscore。没有工具ID输入；此数据集该协议rate仅依赖VB，不人为虚构变化工况。每个split的名单／分数／clip／mean/std／wear scaler在 `Stiehl/splits`；51维字典和18划分名单另提供CSV。



标签按严格previous-run因果对齐；原标签大量为数据集提供的插值结果且无可靠原测量mask，不得统称每个值都是独立直接测量。完整定义和代码见data_protocol.py。

