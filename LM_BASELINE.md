# LM Baseline：从成语任务转向通用大模型

> 本文记录项目核心目标变更后的一条干净主线：**做一个单纯的大模型——自己训练、
> 自己推理、自己适配**。与具体上层业务无关，也不依赖检索外挂。
> 模型能力来自参数本身的语言建模，不是字典查找或业务规则。

## 1. 为什么要转向

此前的成语释义路线（eval2/3/4）用数据证明了三件事：

- eval2 的 `0.1859` 是 holdout 污染下的记忆分，撤回；
- eval3 词级留白上，加长训练 / BPE / 100M 扩容三路线全部 `NO_CHANGE`
  （`0.0676` → `0.0669` → `0.0835`，均未达阈值）；
- 成语释义本质是"查字典"式记忆任务，检索层虽能 `exact_match=1.0`，但那是
  **换工具而非模型变强**（`EVAL4.md`）。

根因不是超参或容量，是**任务性质错配**：把"记住 150 个成语的释义"当成了
"语言能力"。既然核心目标是一个通用的单纯大模型，就该用真正的语言建模任务
（预测下一个 token）来训练和测量，而不是记忆型问答。

**转向决定**：保留全部历史模块作为归档（可复现、默认不启用），新增独立的
通用数据与评测主线，训练核心零改动。

## 2. 新增通用模块

训练核心（`config` / `trainer` / `model` / `data` / `tokenizer` / `inference` /
`export` / `loader`）消费 uint16 token 流，本就与成语无关，**一行未改**。
只新增两块：

### `autotrain/corpus/` —— 通用语料管线

- **源**：中文维基百科 20231101 快照，走 HuggingFace datasets-server `/rows`
  API（纯文本，无需 pyarrow/datasets）。默认 pin `offset=0, n_rows=400`，
  400 篇 / 8.78 MB，源 sha256 `c51924882562…`。CC BY-SA 4.0。
- **切分**：按**文档索引**做种子置换切 holdout（`seed=42, fraction=0.05`），
  而不是 token offset——训练游标 wrap-around 永远碰不到 holdout。这正是
  当初 eval2 踩坑的反面。
- **产物**：`train_tokens.bin`（306 万 token）、`holdout_tokens.bin`（11.95 万
  token）、`tokenizer.json`（char，8192）、`manifest.json`。`corpus_id` 封印
  （排除时间戳），`verify` 逐文件校验 checksum/尺寸/token 数。

### `autotrain/eval_lm/` —— 通用语言建模评测

- 在冻结 holdout 上算 **token 级交叉熵 / 困惑度**，非重叠窗口，eval 模式；
- 确定性 greedy 续写（KV cache，EOS/长度截断）作为定性诊断，**不计入分数**；
- `eval_lm_hash` 绑定 corpus_id/holdout sha/tokenizer sha/seq_len/batch/seed；
  `compare_reports` 拒绝跨 hash、跨族比较——困惑度的进步不能被拿来冒充
  eval3 答对率或 eval4 检索命中率的进步。
- 与 `autotrain/eval/` 的分工：后者是**合成受控实验**（Markov + i.i.d. 负
  对照，验证"确实学到转移结构"）；前者是**真实自然语言困惑度**（维基
  holdout，衡量实际语言建模）。互不可替代、不可跨族比较。

## 3. 基线结果（本次实测）

模型：4 层 / 256 维 / 4 头，514 万参数（非嵌入 302 万），char tokenizer
8192。语料：train 306 万 token。训练：seq_len 128，global_batch 32768，
400 步，1310 万 token，fp32，约 264 秒（49.7k tok/s）。

| 指标 | step0（随机初始化） | step400（训练后） | 变化 |
|------|---:|---:|---|
| val loss | 9.0448 | 5.8898 | −3.16 |
| holdout 交叉熵 (nats/token) | 9.0769 | 5.7180 | −3.36 |
| **holdout 困惑度** | **8750.8** | **304.3** | **↓ 28.7 倍** |

- step0 的 CE `9.08 ≈ ln(8192)=9.01`，正是均匀分布的理论值——证明测量
  起点正确，没有"背词频"作弊空间。
- 严格同 `eval_lm_hash`（`2b70b6f6793432ab`）比较，判定 **IMPROVED**。
  这是模型**本身**学到的语言结构，不是检索、不是记忆答案。
- 续写诊断（`"自然语言处理是"` → `"中国国，中国国国国…"`）：仍在重复早期
  退化阶段，符合 5M 模型 + 400 步的规模——诚实反映基线水平，不做夸大。
  降低困惑度与产出流畅文本是不同阶段的能力，后者需要更大规模/更长训练。

## 4. 与历史尺子不可比（明确声明）

本基线**不可**与 eval2/3/4 的任何数字比较：

| 尺子 | 测什么 | 单位 |
|------|--------|------|
| eval2/3 | 成语释义答案质量（char-F1 等） | 答对率 |
| eval4 | 检索命中释义 | exact_match |
| **eval_lm** | 真实文本 next-token 建模 | 困惑度 |

eval_lm 的 `incomparable_with` 显式列出前三者，`eval_lm_hash` 把它们锁在
不同命名空间，`compare_reports` 会直接拒绝跨族比较。

## 5. 复现

```bash
# 1) 下载 + 构建语料（网络仅 fetch 需要；已缓存则幂等）
python -m autotrain.corpus fetch
python -m autotrain.corpus build
python -m autotrain.corpus verify

# 2) 训练前基线
python -m autotrain.export export-initial \
  --config runs/lm-baseline/config.yaml --out runs/lm-baseline/bundle_step0

# 3) 训练
python -m autotrain.trainer --config runs/lm-baseline/config.yaml \
  --data runs/corpus/lm/train_tokens.bin --out runs/lm-baseline

# 4) 评测 + 对比 + 续写
python runs/lm-baseline/score.py
```

`corpus_id=81bd7bb3646e57be`，源 sha256 `c51924882562…`，报告与对比在
`runs/lm-baseline/reports/`。语料二进制 / bundle / checkpoint 不入库，仅
脚本、配置、报告、日志入库。

## 6. 扩语料 + 加长训练（`runs/lm-scaled/`）

基线之后的第一条路线被执行：只扩大 **train** 流，冻结评测尺子一动不动。

### 数据与不变量

- 新增公开源 `wiki_zh_ext`：同一个中文维基 20231101 快照的 rows **400..1199**
  （800 篇 / 12.7 MB / sha256 `10fd960a…`），与基线 rows 0..399 **零重叠**；
  基线的 20 篇 holdout 来自 rows 0..399，因此新文本绝不进入评测留白。
- 扩展流 = 原 `train_tokens.bin` **字节前缀** + 新文档 token：
  3,059,577 → 4,656,004 appended → **7,715,581 token**。脚本校验
  `prefix_identical_to_base=True`，保证旧 checkpoint 的绝对 data cursor 不失效。
- 复用冻结 char tokenizer，不重建语料、不改 manifest：
  `corpus_id=81bd7bb3646e57be`、holdout sha `65947b95…`、tokenizer sha
  `8ee35713…` 全部保持原值，`eval_lm_hash` 仍是 **`2b70b6f6793432ab`**。
- 冻结词表的 OOV 按固定词表常规落入 `<unk>`，实测 **5,070 / 4,656,004
  = 0.1089%**，已封进 `expansion.json`，没有语料漂移。

### 训练与踩坑

配置与基线同架构（4 层 / 256 维 / 4 头 / 8192 词表），`max_steps=1200`；
`loss_spike_sigma` 从 3 提到 10，避免正常 loss 波动误杀。最终采用**从 step0
干净重训**，而不是从 step400 续训，原因是续训链被两个问题污染：

1. `WarmupCosineScheduler.state_dict()` 只保存 `step_num`，不保存
   `lr_peak/max_steps/warmup_steps`；延长预算后 cosine 重算，末态 LR 从
   `1e-4` reheat 到 `7.44e-4`。已补 `--lr-peak-override` 与 rollback 时
   同步降低 scheduler peak，并加测试覆盖。
2. 被污染的 checkpoint 出现 `step=500` 但 `scheduler_state.step_num=130`，
   rollback 后重读同一难 batch，形成确定性 spike 死循环。放弃续训链后干净重训。

另修了 `score.py` 里按字符串排序 checkpoint 的错误：`ckpt_step_900.pt` 会排在
`ckpt_step_1200.pt` 后面，导致第一次误评 step900；现已按文件名数字排序。
`source.py` 也补了 429/5xx 指数退避，解决 HF datasets-server 偶发 502。

干净重训结果：1200 步 / 39,321,600 token / final val loss **4.9186** /
786 秒 / 50.0k tok/s。

### 严格 A/B

| 指标 | step400 基线 | step1200 扩语料长训 | 变化 |
|------|---:|---:|---|
| holdout 交叉熵 (nats/token) | 5.7180 | 4.6711 | −1.0469 |
| **holdout 困惑度** | **304.29** | **106.82** | **↓ 65.0%（2.85×）** |

`comparable=true`，`eval_lm_hash=2b70b6f6793432ab` 两侧一致，判定
**IMPROVED**。这是同一尺子上的真实 perplexity 下降，不是检索、不是记忆问答。

续写诊断（`"自然语言处理是"`）：基线 `"中国国，中国国国国…"` → 扩展后
`"一些人类的人类的人类…"`。重复从字级退化变成词级重复，说明语言结构
学到了一些，但离流畅生成仍远；这是诊断，不计入分数。

## 7. 后续质量路线（已执行，详见 `LM_QUALITY.md`）

本节原先列出的三条方向已经在同一把 `eval_lm_hash=2b70b6f6793432ab` 上
全部做完，结果不再只是建议：

1. **更多 disjoint 语料 + 更多步数**：rows 1200..1999 再扩 800 篇，
   训练流 771 万 → 1,158 万 token，2400 步，ppl **106.82 → 56.62**
   （↓47.0%，`IMPROVED`）——当前最有效路线。
2. **扩模型容量**：4L/256 → 8L/512，5.15M → 29.17M 参数，
   ppl **106.82 → 87.61**（↓18.0%，`IMPROVED`）；有效但单位算力收益
   低于继续加数据/步数。
3. **同 hash BPE**：用 char-ID-aligned BPE 保持冻结 char holdout 不重分词，
   hash 不变；ppl **106.82 → 195.43**（↑83.0%，实为退化，比较器按协议
   报 `NO_CHANGE`）。在当前 char-level 困惑度语义下不进入主线。

## 8. ext3 加长训练（完成，当前最佳）

继续追加 `wiki_zh_ext3`（rows 2000..2799，800 篇），训练流
11,578,205 → 15,609,856 token，以 ext2 流为字节前缀，四块文档 id 两两
零重叠，冻结 tokenizer / holdout / eval_lm_hash 全部不动。架构 4L/256/4H，
预算 4800 步实际在 **step 4350 early stop**（val loss 停滞）。

同一把尺子上按实际落盘评三个点：

| checkpoint | holdout 困惑度 | vs step2400 (56.62) |
|---:|---:|---|
| step3600 | 38.27 | `IMPROVED` |
| step4000 | 35.76 | `IMPROVED` |
| **step4350** | **34.91** | **↓38.3%，`IMPROVED`** |

step3600→4350 仍 `IMPROVED`（38.27→34.91），**未 plateau、未回归**，但
相邻段降幅从 2.51 收窄到 0.85，边际递减。当前最佳更新为
**step4350 / ppl 34.91 / CE 3.5529**。完整曲线与 plateau 检查见
`runs/lm-scaled3/reports/lm_scaled3_curve.json`，详见 `LM_QUALITY.md` 第 7 节。

> 训练内部 `final_val_loss=4.5991` 高于 ext2 的 4.0003，但两者 val 集切自
> 各自训练流的尾部、并非同一批数据，跨实验不可比；判据只认冻结 holdout
> 困惑度。别拿错尺子下结论 —— 这是本系列反复强调的纪律。
>
> **early-stop 判据已复核并修复（2026-11-24，见 `EARLY_STOP_REVIEW.md`）**：
> 除跨实验不可比外，更根本的问题是 `_validate` 的 4 个 val 窗口几乎完全
> 重叠、有效样本仅 4,100 token，噪声导致 step 4350 过早早停（模型未收敛，
> 冻结困惑度同期仍在降）。已改为 16 个等距互不重叠窗口铺满 31 万 token
> 尾部（约 76×）；`min_delta`/patience 本轮刻意不动以保持单变量归因。
> `runs/lm-fixedval` 干净重训到 4800 步验证中，结果待回填。

继续推进的优先级（更新）：数据已到 1561 万 token、步数已进边际递减区，
下一轮更值得动的是**数据多样性 / 更大模型 / 解码策略**，而不是把同一配方
再拉长；early-stop 判据已按 `EARLY_STOP_REVIEW.md` 修复，正在
`runs/lm-fixedval` 干净重训到 4800 步验证，不要直接调预算。
解码策略（采样 / top-k / repetition penalty）属推理自优化，可改善重复退化
但不改变 perplexity。

关键纪律不变：任何改动都用**同一语料、同一 holdout、同一 eval_lm_hash** 严格
A/B，绝不用检索或记忆型问答冒充语言能力提升。线上推理服务仍未替换。
