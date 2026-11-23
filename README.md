# wanxiang-autotrain

万象（Wanxiang）—— **从零训练、自有、通用的大语言模型框架**。

> **项目目标：做一个单纯的大模型。** 三大支柱：**模型自己训练、推理自己优化、
> Agent 自己适配**。本仓库是可复现的训练基础设施 + 测量仪器；模型能力来自
> 参数与语言建模本身，不是字典查找、检索外挂或业务规则。

## 当前主线：通用 LM 管线 ✅

> 通用语料（`autotrain/corpus`）下载公开中文维基快照，确定性切分文档级
> holdout，构建 char/BPE tokenizer，产出 uint16 token 流——训练核心零改动。
> 通用评测（`autotrain/eval_lm`）在冻结 holdout 上测 token 级交叉熵 / 困惑度，
> 并做确定性续写。`eval_lm_hash` 把报告绑定语料/holdout/tokenizer/配置，
> 严格拒绝跨尺子比较。基线见 `runs/lm-baseline/`（ppl 8750.8 → 304.3），
> 扩语料长训见 `runs/lm-scaled/`（→ 106.8）与 `runs/lm-scaled2/`（→ 56.62）；
> 三条质量路线（更多数据/步数、扩容量、同 hash BPE）见 `LM_QUALITY.md`。
> **ext3 加长训练**见 `runs/lm-scaled3/`：预算 4800 步在 step4350 early stop，
> 当前最佳 **step4350 / ppl 34.91**（同一 `eval_lm_hash=2b70b6f6793432ab`）。
> **early-stop 判据复核**见 `EARLY_STOP_REVIEW.md`（step4350 属过早早停，
> 根因是 val 采样窗口重叠、有效样本仅 4.1K token）；修复已实施（16 个分散
> val 窗口），`runs/lm-fixedval` 为干净重训验证 run。总纪律见 `LM_BASELINE.md`。
>
> **测量仪器分工**：`autotrain/eval/` 是合成受控实验（Markov + i.i.d. 负对照，
> 验证"确实学到转移结构"）；`autotrain/eval_lm/` 是真实自然语言困惑度（维基
> holdout，衡量实际语言建模能力）。两者互不可替代，也不可跨族比较。

### 历史实验（已归档，非主线）

> 早期的成语释义路线与检索增强实验（`m2` / `sft` / `eval2` / `eval3` /
> `eval4` / `retrieval`）作为代码归档保留，可复现但不再是项目方向。结论：
> 成语释义是不可泛化的记忆型查字典任务，三条生成路线（加长训练 / BPE /
> 100M 扩容）均 `NO_CHANGE`；检索层能做到精确命中，但属于「换工具」而非
> 「模型变强」。这些模块默认不参与通用 LM 主线。全量测试 200+ passed。

## 核心组件

| 组件 | 位置 | 作用 |
|------|------|------|
| 配置系统 | `autotrain/config.py` | YAML 配置 + 确定性哈希守卫；`max_steps` 作为预算旋钮不参与哈希，延长预算可合法续训，其余字段变更拒绝续训 |
| 基座模型 | `autotrain/trainer/model.py` | GPT 式 decoder-only，RMSNorm + SwiGLU + 权重绑定 |
| 数据加载 | `autotrain/trainer/data.py` | uint16 token 流 + 游标续训，train/val 尾部分割（分散、互不重叠的 val 窗口） |
| checkpoint | `autotrain/trainer/checkpoint.py` | 原子落盘，含模型/优化器/调度器/数据游标/RNG 状态；恢复时校验配置哈希 |
| 监控 | `autotrain/trainer/monitor.py` | 3σ loss spike 检测（预热期不误报）+ val loss 停滞 early-stop |
| 训练循环 | `autotrain/trainer/runner.py` | warmup + cosine 调度、梯度累积、spike 自动回滚降 LR |
| 实验存储 | `autotrain/store/experiment_store.py` | SQLite 实验记录 + 指标时序 |
| 导出契约 | `autotrain/export/contract.py` | bundle 文件集、config schema、逐文件 checksum、逐张量 sha256、tied alias |
| 导出 / 加载 | `autotrain/export/` | `.pt` → bundle（临时目录 + 校验 + 原子改名）；bundle → 模型，不依赖训练循环 |
| 冻结评测集 | `evals/*.bin` + `manifest.json` | 200K tokens 结构化 Markov 集 + 200K tokens i.i.d. 均匀负对照 |
| 客观指标 | `autotrain/eval/metrics.py` | 交叉熵、bits/token、perplexity、结构捕获率、greedy 转移命中率；无 LLM-as-a-judge |
| 通用 LM 评测 | `autotrain/eval_lm/` | 冻结 holdout 上的 token 级困惑度 + 确定性续写 |

**为什么用 Markov 语料**：其 unigram 分布仍是均匀的，理论最优交叉熵是
`ln(8192)=9.0109`，而真实一阶转移分布的最优值约 `2.30` nats。低于 `ln(V)`
只可能来自「学到转移结构」，不可能来自「背词频」。i.i.d. 均匀集是负对照：
如果模型在那里也显得变好，说明测量坏了。

**只读边界**：训练代码不写 bundle，评测器不写 `evals/`；任何阈值、窗口、
种子或语料字节的变动都会改变 `evalspec_hash`，使新旧报告直接判为
`INCOMPARABLE`，从而杜绝「自己给自己挪球门」。

#### Bundle 结构

```
<bundle>/
  config.json        架构、结构、tokenizer 元数据、dtype
  model.safetensors  权重（省略 tied alias `lm_head.weight`）
  tokenizer.json     最小 token-id 规格
  MANIFEST.json      来源、step、逐张量清单、逐文件 checksum
```

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # 必须 editable：evals/ 是仓库级冻结资产，不进 wheel
pytest tests/ -q
```

`runs/` 下的语料、tokenizer、checkpoint 与 bundle 是本机生成物，不随仓库分发；
报告与 provenance（`.json` / `.py` / `.yaml` / `.log`）随仓库入库，便于复现实验。

## 通用 LM 复现

```bash
# 1. 训练（uint16 token 流，配置见 runs/lm-*/config.yaml）
python -m autotrain.trainer \
  --config runs/lm-baseline/config.yaml \
  --data <train_tokens.bin> --out <run-dir>

# 2. 导出 bundle
python -m autotrain.export export \
  --checkpoint <run-dir>/checkpoints/ckpt_step_<N>.pt \
  --config runs/lm-baseline/config.yaml --out <bundle-dir>

# 3. 在同一把冻结尺子上评困惑度
python -m autotrain.eval_lm score \
  --bundle <bundle-dir> --corpus <frozen-corpus-dir> \
  --out <report.json> --seq-len 128 --batch-size 16

# 4. 严格 A/B（跨 eval_lm_hash 直接拒绝）
python -m autotrain.eval_lm compare \
  --baseline <report-a.json> --candidate <report-b.json>
```

完整命令与逐 run 的 provenance 见 `runs/lm-*/` 下的脚本与 `reports/`。

## 严格 A/B 纪律

- **冻结尺子**：`eval_lm_hash` 绑定 corpus id、holdout sha256、tokenizer sha256、
  seq_len、batch_size、seed；跨 hash 一律拒绝比较。
- **train-only 扩展**：新增语料只追加训练流，holdout / tokenizer / manifest /
  eval corpus 字节不动；扩展后的 train 流必须以旧流为字节前缀。
- **冻结 holdout 不参与训练内部 early stop**：否则会把最终评测尺子变成训练信号。
- **数字排序**：checkpoint 按文件名中的数字 step 排序。

各 run 的曲线、A/B 结论与诊断报告见 `LM_BASELINE.md`、`LM_QUALITY.md`、
`EARLY_STOP_REVIEW.md` 与 `runs/*/reports/`。

## 路线图

| 里程碑 | 内容 | 状态 |
|--------|------|------|
| M1 | Trainer Runner + 断点续训 | ✅ |
| M1.5 | 只读评测器 + 自描述导出契约 | ✅ |
| M2 | 数据管线（采集/清洗/配比/版本化） | 待开始 |
| M3 | Orchestrator + 一键流水线 | 待开始 |
| M4 | 超参搜索（代理模型扫描 + μP 外推） | 待开始 |

## 语料许可

训练与评测使用的中文维基百科快照遵循 **CC BY-SA 4.0**
（https://creativecommons.org/licenses/by-sa/4.0/）。历史归档实验使用的
公开成语数据来自 `chinese-xinhua`（MIT）。所有报告中的数值均为本机复现结果，
不承诺跨设备 / 跨快照复现出相同数值。
