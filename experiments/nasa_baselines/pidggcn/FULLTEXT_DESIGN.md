# PIDGGCN 全文核查及 NASA 因果适配设计

来源：用户提供 `<repository>/1-s2.0-S1474034625006834-main.pdf`，18页，DOI 10.1016/j.aei.2025.103790。页码均为PDF印刷页码。旧blocked记录对应此前未获得全文阶段；现在已消除全文阻塞，但未获得作者源码，仍存在以下实施歧义。

## 原方法任务及证据

原文是当前tool wear monitoring，不是无未来sensor的多步未来状态预测。p10 §5.1明确物理模型仅训练使用，推断不构建或应用物理模型。其损失有真实当前磨损监督和导数一致性，不含把学习速率数值积分到未来标签的监督。原实验HMoTP三把工具同工况、CFRP数据；NASA跨工况是迁移。

## 公式—实现对应

| 页/公式 | 原文 | 实现/明确决策 |
|---|---|---|
| p3 Eq1 | f(t)=t exp(A+Bt+Ct²) | `fit_prior`，真实mm标签拟合；统一u=t/T保持数值稳定 |
| p4 Eq4、5；p9 Eq32 | 训练Pearson排名、top10%、训练minmax | NASA沿用已冻结训练内Spearman top12电流及clip_zscore；不是原特征前处理逐项复刻 |
| p4 §2.3；p5 Eq16 | feature nodes；S=XnormᵀXnorm | X形状[已见记录数,特征数]，按列L2归一化；S为[特征,特征]，第二层为64×64 |
| p5 Eq15–20 | τ=mean(S)+0.5std(S)，A=softmax(ReLU(S−τ)) | `dynamic_adjacency`，population std，最后一维softmax；零logit有正权，不虚构hard稀疏边 |
| p6 Eq21–25；p5 Eq10–13 | update/reset/candidate门控、时间递推 | `DynamicGatedLayer`：message=A@当前向量；candidate=Wc(message)+Uc(r⊙h)，自行实现而非调用不等价PyTorch GRU |
| p6 Eq26 | 层输入H及终端hidden符号 | 每层每case零初始化，输出全序列；不把全序列末状态作为下一层早期初值 |
| p6 Eq27–28、p7 Eq30–31 | Ld+Lp+Lk，虽文字写four terms | `physics_loss`三个均方项；节点在case内均值、case等权；原L2平方和与文字MSE不一致，显式采用均值 |
| p6 半衰减段、p7 Fig6 | 5000轮，前2500权重1，每100轮按两倍关系减半 | `HalfDecay`提供原版和显式比例压缩版；拟正式100轮使用warmup50、52起每2轮检查，训练loss决定、不用验证/测试 |
| p11 Table2 HMoTP | DGGCN64/64，linear64/32/1，dropout.5，RMSprop.001 | `PIDGGCN`相同宽度；dropout具体位置原文未指定，当前放两层输出及head隐藏激活后；head ReLU为实现假设 |

## 原文矛盾及不能冒充的细节

1. **图节点表述冲突**：p3综述说每cutting sample为node；p4技术定义明说sensor features为nodes，Eq16 XᵀX、Fig2标签f支持feature图。本实现以技术定义为准，不能称已恢复作者确切tensor布局。
2. **物理导数误差**：Eq1真正微分为 `(1+Bt+2Ct²)exp(A+Bt+Ct²)`。p7 Eq31及Fig5都印 `(1+Bt+Ct²)exp(...)`。已视觉核对。`paper_printed`和`calculus_corrected`必须分开同配置报告，不按测试择优。
3. **时间输入未写清**：文章定义∂ŷ/∂t但没说明t如何进入网络，也没指定AD还是相邻差分。NASA显式保留timestamp输入，AD求Jacobian对角∂ŷ_i/∂u_i，不能用grad(sum(ŷ))误把其他节点贡献合入。此偏导固定其他输入，不能等同沿整个sensor轨迹的总导数。
4. **推断节点可见性不明**：文章明确训练不使用测试工具、排名和归一化训练内计算，但未明说测试图用完整工具轨迹还是当前prefix。不能断言作者泄漏。本适配每时刻只以同case当前及过去有效sensor列轨迹构图，包括阈值统计；无其他测试case，无未来sensor。
5. Eq21–23中的i−1输入索引与当前监测输出未给首步处理，Eq24排版不完整；采用Fig3/4标准门控与当前图聚合。跨层h_N传参含义不明确，采用每层zero initial以保持因果。
6. RMSprop的alpha/eps/momentum、batch、dropout位置、head激活、物理拟合优化器与边界没有完整说明。代码使用PyTorch RMSprop默认alpha=.99/eps=1e-8/momentum0、case等权full batch、PyTorch默认初始化，无权重衰减。

## NASA预设协议（正式训练尚未启动）

- 使用现有16split和138外层标签，top12/clip_zscore完全复用，不重排、不重拟合sensor scaler。X17是12电流+4工况编码+1时间。模型需要时序prefix用于图及门控，不能强行当作L1 MLP。
- 每case按真实time排序；坏sensor不进入图、hidden或feature统计。监督仅train_fit_mask；验证/测试无效当前sensor仍须用同split/seed冻结CT后备，保留评价池。当前smoke覆盖训练部分，完整fallback/evaluation适配待正式runner实现与审计。
- 物理prior只在各split训练标签中按原始 `(DOC,feed,material)` 分组拟合，模拟原文同工况；真实VB mm，u=t/T，T取冻结训练time_scale。若组不足4个t>0且VB>0标签或不足3个不同正时间、拟合失败或系数触界，则回退该split全训练pool。绝不为测试case拟合prior。
- 拟合采用非线性最小二乘，log-linear初值仅用正时间/正VB，优化原mm残差；训练case等权，A/B/C预设界[-30,30]。记录系数、残差、样本数、case、fallback原因；界限和稀少样本规则是NASA数值稳定假设。
- 网络预测y_z；导数和prior统一为z/u：`reference_rate_u / wear_std`。这使物理loss权重依赖选定时间/磨损标准化，与原文未说明单位的loss有差异，明确披露。
- 两个导数版本各2fold×(7inner seed0+5final seed)=24次，各100epoch，共48次；版本固定全部报告，不做外层选择。每轮保存train/validation/test原始预测及三项loss、权重和选择所需指标。
- 物理参考仅训练正则，**不附加不存在于论文的future head或ODE**。未来VB比较暂标不适用；不能把未来时间输入且伪造未来sensor作为原方法未来预测。

## 当前实现验证及耗时

`smoke.py`仅跑fold_a/inner_1三轮，无正式训练。prefix与未来sensor扰动误差均0；own-time偏导对照有限差分最大误差约1.24e-12；corrected导数与解析微分误差1.11e-16；模型重载完全一致。

feature图的零列在二阶导时会使普通L2 norm产生NaN，已使用`sqrt(sum(x²)+1e-12)`及方差sqrt epsilon稳定化；这属于数值实现，不改变softmax语义。完整序列图受未来sensor扰动的对照为非零，仅证明当前离线实现需因果适配，不证明原作者实际推断泄漏。

最新三轮训练约3.39秒，即1.13秒/epoch，CPU单线程float64。每版本24×100约45分钟，双版本约90分钟，未包含正式逐epoch预测保存及审计；建议预留约2小时。三轮总loss46.24→509.52→71.50，梯度有限但物理项波动大，smoke只确认代码可执行与梯度正确，不构成收敛/有效性证据。

运行：`D:/anaconda3/envs/CondWear-ODE/python.exe pidggcn/smoke.py`。`smoke_results.json`含完整数值和拟合信息；`fulltext/formulas-05.png`至`07.png`为已视觉核对页面。
