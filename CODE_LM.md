# 代码模型（Code LM）——本地可运行的代码补全 + 对话

> 目标：在通用 LM 底座之后，用**本地源码**训练一个代码模型，纯本地 CPU 可部署，
> 同时具备「续写代码」与「对话式补全」两种形态。
>
> 状态：**已在 Windows RTX 5080 上完成 4800 步预训练 + SFT（step-5600），两把冻结尺子均验证**。
> 代码 holdout ppl：8637.55 → 40.60 → 33.83 → **11.06**（step-4800）→ **10.75**（step-5600）。
> SFT answer CE：9.067 → 3.761 → 3.495 → 2.508 → **2.338**；top-1：0.07% → **55.9%**。
> **但 greedy 生成仍普遍重复退化，客观指标改善 ≠ 生成质量达标**，见第 6 节。

---

## 0. 三条不可放松的边界

1. **脱敏**：语料来自本机一个专有源码工程（路径与项目名不记录）。源码、
   jsonl、token 流、SFT 样本、manifest
   全部只落在 `runs/`（`.gitignore` 忽略），**绝不提交、绝不上传**。所有 manifest
   只记匿名统计与内容 hash，不记路径、文件名、作者、仓库名。
   [collect.py](<autotrain/code_corpus/collect.py>) 的每条记录只有
   `text / lang / doc_sha256 / lines / chars`。
2. **保留代码结构**：通用语料的 `clean_text()` 会把 `\n`/`\t` 压成空格，
   那是代码的语法。本管线用 `clean_code()` 保留换行与缩进，只统一行尾、
   去控制字符、折叠连续空行。
3. **冻结尺子**：holdout 按**文件**切分（不是 token offset），训练游标 wrap
   也碰不到评测代码；`corpus_id` / `eval_lm_hash` / `sft_hash` 全部封印。

---

## 1. 已完成（Windows 侧）

| 项 | 结果 |
|---|---|
| 语料采集 | 97,513 个候选源码文件 → 确定性抽样 8,777 个文档 / 64 MiB，按内容 hash 去重 |
| 语言分布 | `cs` 3322 / `cpp` 1726 / `h` 1505 / `lua` 1198 / `shader` 390 / `c` `cginc` `hlsl` `ts` `py` 等 |
| 冻结语料 | `runs/code-corpus/`：`corpus_id=7d2c1051a78b28f5`，**23,725,640 train / 1,204,087 holdout tokens** |
| tokenizer | BPE 8192，5830 个代码子词，压缩率 **2.93 字符/token**，`public`/`void`/`GameObject`/`transform`/`Vector3`/`UnityEngine`/`namespace` 均为单 token，decode 无损 |
| 尺子 A（预训练） | `eval_lm_hash=e11e685d29058360`；step-0 baseline ppl 8637.55 |
| 尺子 B（SFT） | `runs/code-sft/`：52,520 train / 2,744 holdout 样本，`corpus_id=1b30355353ad94cd`，`template_hash=d89c4bdb…`，`sft_hash=ca0791379eeaf147`；step-0 answer CE 9.067 / top-1 0.074% |
| 跨尺子隔离 | `eval_lm compare` 明确拒绝 `2b70b6f6793432ab`(wiki) vs `e11e685d29058360`(code) |
| 确定性 | 重跑采集 + 构建得到相同 jsonl hash、`corpus_id`、tokenizer sha、train sha |
| 本地推理 | CPU `generate` 冒烟通过；chat template + BPE tokenizer 校验通过 |
| 全量测试 | `210 passed, 1 skipped`（新模块未改动任何既有测试） |
| 补全模型 step-1200 | 1200 步 / 100 秒；ppl 8637.55 → **40.60**（`IMPROVED`），`final_val_loss=3.7691` |
| 对话模型 step-2000 | 800 步 SFT / 58 秒；ppl **33.83**、answer CE **3.495**、top-1 **38.5%** |
| 续训到 4800 | 3600 步 / 296 秒；`final_val_loss 3.7691 → 2.4847`；ppl 40.60 → **11.06**，answer top-1 → **54.0%** |
| SFT v2 step-5600 | 800 步 / 63 秒；ppl **10.75**、answer CE **2.338**、top-1 **55.9%** |
| greedy 生成 | **所有阶段均重复退化**（`override override void OnOnOn...`、`// the focus to the focus...`）；`--do-sample --top-k` 未稳定解决。指标改善 ≠ 生成达标，见第 6 节 |

代码改动：新增 `autotrain/code_corpus/`、`autotrain/eval_gen/`、
`autotrain/configs/wanxiang-code-lm.yaml` / `wanxiang-code-large.yaml`、
`tests/test_code_sft.py` / `tests/test_ss.py` 及本文档；修改
`autotrain/inference/{engine,server,__main__}.py` 与
`autotrain/trainer/runner.py`（均为默认关闭或向后兼容的改动）。
全量测试 `223 passed, 1 skipped`。

---

## 2. 语料管线（Windows，可复现）

```powershell
cd D:\wanxiang
$py = '.\.venv\Scripts\python.exe'   # 全路径绕开 PowerShell 执行策略，不激活 venv

& $py -m autotrain.code_corpus collect --root '<SOURCE_ROOT>' `
    --out runs/code-corpus/sources/code.jsonl `
    --max-bytes 67108864 --sample-files 40000 --seed 42

& $py -m autotrain.code_corpus build `
    --source runs/code-corpus/sources/code.jsonl --out runs/code-corpus `
    --seed 42 --tokenizer-sample-chars 8388608

& $py -m autotrain.code_corpus verify --corpus runs/code-corpus
& $py -m autotrain.corpus verify --corpus runs/code-corpus   # 通用校验器同样通过
```

参数：`--max-bytes` 控制语料规模（当前 64 MiB，可放大）；`--sample-files` 上限；
`--seed` 决定确定性抽样顺序。换任何参数都会改变 `corpus_id` → 新尺子。

采集规则：白名单扩展名（排除 `.xml/.json/.sln/.csproj/.meta` 等配置噪声）；
剪掉 `.git`/`builds`/`Library`/`Temp`/`obj`/`bin`/`node_modules`/`ThirdParty`/
`vendor`/`extern`/备份目录；跳过二进制、超大文件、过小样板；按内容去重。

---

## 3. SFT 对话层（数据，Windows 已生成）

```powershell
& $py -m autotrain.code_corpus sft-build `
    --source runs/code-corpus/sources/code.jsonl `
    --tokenizer runs/code-corpus/tokenizer.json --out runs/code-sft `
    --max-samples 200000
& $py -m autotrain.code_corpus sft-verify --corpus runs/code-sft
```

样本形态（单轮）：用户粘贴 8 行代码前缀 + `complete the code below ({lang}):`，
助手回复后续 12 行；`sft_labels.bin`（uint8）只把回复位置标 1，
`TrainerRunner(label_file=...)` 因此是真正的 answer-side loss。
chat markers 用纯 ASCII（`user:` / `code:` / `sys:`）以适配代码 BPE 词表，
`ChatTemplate` 版本与推理服务兼容。SFT holdout **只来自预训练 holdout 文档**。

---

## 4. 训练

### 4.0 Windows GPU（本机首选）

本机是 RTX 5080（驱动 617.42 / CUDA 13.4 / 16GB）。默认 `.venv` 装的是
`torch 2.15.0+cpu`，所以另建独立 `.venv-cuda` 装
`torch==2.14.1+cu130`（`arch_list` 含 `sm_120`），不覆盖 CPU 验证环境：

```powershell
cd D:\wanxiang
& '.\.venv-cuda\Scripts\python.exe' -m autotrain.trainer `
    --config runs/code-lm/config-stage1.yaml `
    --data runs/code-corpus/train_tokens.bin `
    --out runs/code-lm/train-gpu --log-level WARNING
```

`TrainerRunner` 在 CUDA torch 下自动选 `cuda`，命令无需改。稳态实测
**391K tok/s**，1200 步 **100 秒**完成（`final_val_loss=3.7691`），显存无 OOM。

> **重要**：正式配置里的 `loss_spike_sigma=10.0` 会在 loss 平稳下降后因
> `std` 极小而误判 spike，触发「回滚到 step 300 → 相同数据重跑 → 再回滚」
> 的无限循环（LR 被反复减半到 0，checkpoint 永不前进）。本次使用
> `runs/code-lm/config-stage1-stable.yaml`（`loss_spike_sigma=1e9`）关闭误触发
> 的检测器后才正常完成。这是 `LossSpikeDetector` 的已知缺陷，见
> [monitor.py](<autotrain/trainer/monitor.py>) 与 runner 里的回滚注释。

`max_steps` 是预算旋钮，改回 4800 对同一 `--out` resume 即合法续训。

导出与验证仍用现有 CPU `.venv`（bundle 是 CPU 张量 + safetensors，与 torch
版本无关），见第 5 节。

### 4.0.1 SFT 对话层（本机 GPU 实测）

`config_hash` 必须逐字一致才能 resume，所以 SFT 复制预训练配置、只把
`train.max_steps` 改成 2000（预算旋钮不参与 hash）—— 因为 resume 会把
`step` 恢复到 1200，而循环是 `while step < max_steps`，预算必须大于 1200。
把 `ckpt_step_1200.pt` 复制进新 run 的 `checkpoints/`，用
`lr_peak_override=1e-4` 钉住预训练最终 LR 避免 cosine reheat：

```powershell
.\.venv-cuda\Scripts\python.exe runs/code-sft-train/train.py
```

脚本用 `TrainerRunner(..., label_file='runs/code-sft/sft_train_labels.bin',
device='cuda')` 做真正的 answer-side loss。实测 800 步 / **58 秒**，
`final_val_loss 3.7691 → 3.5828`，产出 `runs/code-sft-train/bundle`。
chat 格式 prompt 下已能续写出 `if (m_Texture == null) { ... }` 这类真实
C# 结构；启动带 chat 端点的本地服务见第 5 节。

> 以下 4.1–4.3 是 **Mac MPS 备选路径**，本机有 GPU 时不需要。

### 4.1 同步清单

**先 commit + push** 新增代码（不含任何源码）：
`autotrain/code_corpus/`、`autotrain/configs/wanxiang-code-lm.yaml`、`CODE_LM.md`。

**手动拷贝数据**（不进 git，勿贴进仓库）：
- `runs/code-corpus/` → Mac 同名目录（`train_tokens.bin` / `holdout_tokens.bin` /
  `tokenizer.json` / `manifest.json`）
- `runs/code-sft/` → Mac 同名目录（SFT 数据 + `chat_template.json` + `tokenizer.json`）

### 4.2 预训练（补全模型，从 step 0）

配置：[wanxiang-code-lm.yaml](<autotrain/configs/wanxiang-code-lm.yaml>)
（4 层 / 256 dim / 4 heads / 8192 词表 / **seq_len 512** / fp32，约 3.5M 参数，
CPU 可跑）。`config_hash` 含 `data.mix`/`seq_len`/`model.*`，这是**新实验**，
不能合法 resume 维基 step-4800 权重。

`python -m autotrain.trainer` 没有 `--device`，Mac 上会退到 CPU。用下面这段
显式走 MPS（不落盘、不需要改仓库文件）：

```bash
cd /path/to/wanxiang
python - <<'PY'
from autotrain.config import load_config
from autotrain.trainer.runner import TrainerRunner
cfg = load_config('autotrain/configs/wanxiang-code-lm.yaml')
r = TrainerRunner(cfg,
                  'runs/code-corpus/train_tokens.bin',
                  'runs/code-lm/train',
                  device='mps')
print(r.train())
PY

# 导出 bundle
python -m autotrain.export export \
  --checkpoint runs/code-lm/train/checkpoints/ckpt_step_<N>.pt \
  --config autotrain/configs/wanxiang-code-lm.yaml \
  --out runs/code-lm/bundle
```

把 `runs/code-lm/bundle/` 同步回 Windows。

### 4.3 SFT（对话层）

合法 resume 要求 `config_hash` 相同 → 复用同一份配置，只换数据文件、
加 `label_file`、调 `max_steps`（预算旋钮，不参与 hash）。
把预训练 checkpoint 复制进新 run 目录让 `resume()` 捡起；
`TokenDataset` 在 cursor 越界时会自动 wrap 到 0，安全。

```bash
cp autotrain/configs/wanxiang-code-lm.yaml runs/code-sft-train/config.yaml
# 编辑 runs/code-sft-train/config.yaml 的 train.max_steps 为 SFT 预算（不影响 hash）
mkdir -p runs/code-sft-train/checkpoints
cp runs/code-lm/train/checkpoints/ckpt_step_<N>.pt runs/code-sft-train/checkpoints/

python - <<'PY'
from autotrain.config import load_config
from autotrain.trainer.runner import TrainerRunner
cfg = load_config('runs/code-sft-train/config.yaml')
r = TrainerRunner(cfg,
                  'runs/code-sft/sft_train_tokens.bin',
                  'runs/code-sft-train',
                  device='mps',
                  label_file='runs/code-sft/sft_train_labels.bin')
assert r.resume(), 'SFT must resume from the pretrained checkpoint'
print(r.train())
PY

python -m autotrain.export export \
  --checkpoint runs/code-sft-train/checkpoints/ckpt_step_<M>.pt \
  --config runs/code-sft-train/config.yaml \
  --out runs/code-sft-train/bundle
```

把 `runs/code-sft-train/bundle/` 同步回 Windows。

---

## 5. Windows 侧：验证（本地 CPU）

```powershell
cd D:\wanxiang
$py = '.\.venv\Scripts\python.exe'

# 契约校验
& $py -m autotrain.export describe --bundle runs/code-lm/bundle

# 尺子 A：同一把代码 holdout 上的严格 A/B
& $py -m autotrain.eval_lm score --bundle runs/code-lm/bundle --corpus runs/code-corpus `
    --seq-len 512 --batch-size 8 --device cpu --out runs/code-lm/reports/code_trained.json
& $py -m autotrain.eval_lm compare `
    --baseline runs/code-lm/reports/code_step0.json `
    --candidate runs/code-lm/reports/code_trained.json

# 尺子 B：冻结 SFT holdout 上的 answer-side CE / top-1
& $py -m autotrain.code_corpus sft-score --bundle runs/code-sft-train/bundle `
    --corpus runs/code-sft --seq-len 512 --batch-size 8

# 本地补全推理
& $py -m autotrain.inference generate --bundle runs/code-lm/bundle `
    --tokenizer runs/code-corpus/tokenizer.json `
    --prompt 'public void Update() {' --max-tokens 32 --device cpu --threads 4

# 本地对话服务（补全 + chat 端点）
& $py -m autotrain.inference serve --bundle runs/code-sft-train/bundle `
    --tokenizer runs/code-sft/tokenizer.json `
    --chat-template runs/code-sft/chat_template.json `
    --host 127.0.0.1 --port 8080 --device cpu --threads 4
# POST /v1/chat/completions
# {"messages":[{"role":"user","content":"complete the code below (cs):\npublic void Update() {"}]}
```

两次评测统一 `--device cpu`：`COMPAT_KEYS`/`eval_lm_hash` 绑定设备，
跨设备报告会判 `INCOMPARABLE`。

---

## 5.1 生成质量尺子（eval_gen）与解码结构

teacher-forced 尺子看不到 autoregressive 循环退化，所以新增
[eval_gen](<autotrain/eval_gen>)：从冻结代码 holdout 确定性取 32 个 prompt，
固定 greedy 生成 64 tokens，用纯客观统计量量化重复 ——
`distinct_1` / `distinct_2` / `repeat_3gram_rate` / `max_3gram_count` /
`looped`。`gen_hash` 只绑定 prompt 集 / 长度 / seed / corpus，**不含解码配置**
（解码是被测变量，所以不同配置仍可 A/B）。

推理引擎 [engine.py](<autotrain/inference/engine.py>) 的 `GenerationConfig`
新增三个**默认关闭**的结构控制，已暴露到 HTTP 与 CLI：
`repetition_penalty`（软）、`frequency_penalty`（按出现次数递增）、
`no_repeat_ngram_size`（硬禁制）。默认全关，历史行为逐字节不变。

实测（step-5600，同一 `gen_hash`）：

| 指标 | 历史 greedy | 仅软约束 | 软 + 硬约束 |
|---|---|---|---|
| looped | 93.75% | 53.1% | **0%** |
| repeat_3gram_rate | 60.5% | 13.5% | **0%** |
| distinct_1 | 25.5% | 58.3% | **69.4%** |
| distinct_2 | 34.9% | 79.8% | **93.2%** |

**消融**：只用软约束（不加硬 n-gram 禁制）仍 `IMPROVED`（5 改善 / 1 微退），
说明改善不是硬禁制「自己给自己设答案」。全阶段基线（历史 greedy 下
`looped`）：step-0 / 1200 / 2000 全 **100%**，step-4800 **90.6%**，
step-5600 **93.8%** —— 加 step 只能缓慢缓解，**SFT 反而让生成退化**
（teacher-forced answer CE 变好、循环变多），是 exposure bias 的直接证据。

> **边界**：这把尺子只测「重复」，不测「正确性」。开抑制后输出多样但
> 仍不是可靠代码（`// The is notifies, ... #pragma once`）。它证明的是
> 循环退化可解，**不是**模型能写对代码。

```powershell
& '.\.venv\Scripts\python.exe' -m autotrain.eval_gen score --bundle <B> `
    --corpus runs/code-corpus --out runs/eval_gen/off.json
& '.\.venv\Scripts\python.exe' -m autotrain.eval_gen score --bundle <B> `
    --corpus runs/code-corpus --out runs/eval_gen/on.json `
    --repetition-penalty 1.2 --no-repeat-ngram 3 --frequency-penalty 0.1
& '.\.venv\Scripts\python.exe' -m autotrain.eval_gen compare `
    --baseline runs/eval_gen/off.json --candidate runs/eval_gen/on.json
```

HTTP 用法：`/v1/chat/completions` body 里加
`"repetition_penalty": 1.2, "no_repeat_ngram_size": 3, "frequency_penalty": 0.1`。

## 5.2 SFT 数据重构（block mode）

[sft.py](<autotrain/code_corpus/sft.py>) 新增 `--mode block`，针对「固定
8 + 12 行滑窗把函数切在中段、模板代码教模型重复」：

- `_find_blocks` 按**花括号块边界**切分，prompt 长度按块自适应（3–12 行），
  answer 更短且可变（≤10 行）
- `_is_template` 过滤纯注释块 / getter-setter / 重复行模板
- answer 按内容 hash **全局去重**，去重集合跨 train / holdout 共享，
  防止模板答案泄漏进评测
- `window` 模式保留旧行为；`tests/test_code_sft.py` 锁定全部纯函数

新数据 `runs/code-sft-block/`：`corpus_id=64e7f55089ace48a`，
43,953 train / 2,275 holdout 样本、8.4M tokens（旧：52,520 / 2,744 / 15.3M）。
chat template 不变（`template_hash=d89c4bdb…`）。换数据即换尺子：
新 `sft_hash=2f4acd1d99e1caa3`，与旧 `ca0791379eeaf147` **不可比**。
从 step-4800 权重重训 800 步 / 77 秒 → `runs/code-sft-block-train/bundle`。

**用冻结 `eval_gen` 尺子严格 A/B**：`gen_hash` 绑定 code-corpus，不受 SFT
数据影响，所以新旧模型在同一把尺子上可比。

| 指标 | 旧 SFT (window) | 新 SFT (block) | delta |
|---|---|---|---|
| looped | 93.75% | **84.38%** | −9.4pp |
| repeat_3gram_rate | 60.5% | **53.9%** | −6.7pp |
| distinct_1 | 25.5% | **28.5%** | +3.0pp |
| distinct_2 | 34.9% | **40.3%** | +5.4pp |
| max_repeat_run | 1.125 | 2.312 | **+1.19（退化）** |

`verdict: IMPROVED`（5 改善 / 1 退化）。叠加解码控制后 `looped` 0%、
`distinct_1` 67.8%。`eval_lm` 对 step-4800 为 `NO_CHANGE`
（无灾难性遗忘，也无提升）。

> **诚实结论**：改善**有限**。`looped` 仍 84%，且单 token 连续重复变严重
> （chat 样例 `// The fold is a a a a a...`）。数据重构缓解了模板驱动的循环，
> **没有解决 exposure bias**；生成仍不是可靠代码。真正的下一步是训练目标层
> （scheduled sampling / 训练时暴露模型自身的生成前缀），或容量与语料规模
> 一起动 —— 而不是继续换 SFT 切法或加步数。

## 5.3 训练目标层：exposure bias 干预（已证伪，勿启用）

[runner.py](<autotrain/trainer/runner.py>) 新增 `ss_max_prob` /
`ss_ramp_steps` / `ss_start_step` / `ss_seed` 构造参数与
`prediction_aware_perturb`：训练时按 ramp 概率把输入位置 `t` 换成模型在
`t-1` 的 argmax 预测（**targets 保持真实**），让模型练习从自己的上下文恢复。
参数**不进 config**（保持 `config_hash` 一致以便合法 resume step-4800），
默认全关 → 历史行为逐字节不变；`tests/test_ss.py` 锁定纯函数语义。

从 step-4800 + block 数据重训两轮（`ss_max_prob=0.3 / 0.1`，其余逐字一致，
与 `block_off.json` 只差训练目标一项）：

| 指标 | block 基线 | SS 0.3 | SS 0.1 |
|---|---|---|---|
| looped | 84.38% | **100%** | **90.62%** |
| distinct_1 | 28.5% | 24.7% | 26.2% |
| max_repeat_run | 2.31 | **11.5** | 1.25 |
| eval_gen verdict | — | `REGRESSED` | `REGRESSED`（5 退化/1 改善） |

两个概率都退化，消融排除了「参数太激进」。其余尺子一致：
SFT 尺子（数据未变，`sft_hash=2f4acd1d99e1caa3` 可比）answer CE
2.319 → **2.452**、top-1 55.4% → 53.5%；`eval_lm` ppl 11.06 → 12.86
（`NO_CHANGE`）；teacher-forced val_loss 2.4846 → 2.6358 / 2.5198。

**为什么失败（判断）**：这是单步、非自回归的近似。只替换一个输入位置，
不覆盖生成时多步累积的偏离；且 argmax 在模型不自信处往往给出高频/重复
token（空格、换行、`a`），把它们注入上下文反而强化重复模式 ——
`max_repeat_run` 飙到 11.5 是直接证据。真正的 scheduled sampling 需要
自回归地用模型整段生成替换输入（每步多 T 次 forward）或序列级目标；
在当前 3.5M / 64 MiB 规模下这个高效近似不成立。

> **结论：此路已证伪。`ss_max_prob` 保持默认 0，不要启用。**
> 当前最佳仍是 5.2 的 block 数据模型 + 5.1 的解码控制
> （`looped` 0%、`distinct_1` 67.8%）。继续调 SS 参数碰运气不符合项目纪律。
>
> 注意：SS 0.3 的某个 chat 样例输出 `public void Start() { ... }` 看起来像
> 代码，但 32 个冻结 prompt 的客观统计明确更差 —— 这正是不能用挑样例
> 代替尺子的例子。

## 6. 已知局限（不要当成已达标的模型）

- **规模小**：4L/256d ≈ 3.5M 参数，本地 CPU 秒级推理是硬约束，
  复杂代码生成/真实问答能力有限。这是可跑通的原型，不是 Codex 级模型。
- **greedy 重复退化（实测；解码可压制，根因未解决）**：见 5.1 / 5.2。
  step-1200 / 2000 / 4800 / 5600 四个
  bundle 在 greedy 解码下都会陷入 `override override void OnOnOn...`、
  `// the focus to the focus to the focus...` 这类循环；加
  `--do-sample --temperature 0.8 --top-k 40` 也只把重复换成无意义碎片。
  teacher-forced 尺子（ppl / answer CE / top-1）随步数单调改善，但生成质量
  没有同步改善 —— 这与 [EVAL3.md](<EVAL3.md>) 记录的副作用同类。
  **当前模型不能当作可用的代码助手上线**；按项目纪律不为 Demo 好看去换
  采样或改评测口径，要修就从模型容量 / 数据 / 解码本身修。
- **SFT 数据是自动补全对**，不是真实用户指令；`both` 形态里的"对话"
  目前只会续写代码，不会解释、不会多轮推理。要真正对话需要更高质量的
  指令数据（人工或更强模型生成），那已超出本地源码自举的范围。
- **样本按行窗口切分**，没有 AST 级函数边界，可能切在函数中段。
- **语料 64 MiB 是起点**：`--max-bytes` 可放大，但放大后编码时间、
  Mac 训练时长、wrap 遍数都会变；每次换参数都是新尺子。
- **词表 8192** 对多语言 + 中文注释偏紧；若改 `vocab_size`，
  `config_hash` 变，bundle 契约与全部报告都要重做。
- 报告里的数值是本机复现结果，不承诺跨设备/跨快照一致。

## 7. 复现命令速查

| 目的 | 命令 |
|---|---|
| 重建语料 | `python -m autotrain.code_corpus collect` → `build` → `verify` |
| 重建 SFT 数据 | `python -m autotrain.code_corpus sft-build` → `sft-verify` |
| 打分（预训练） | `python -m autotrain.eval_lm score --corpus runs/code-corpus --seq-len 512` |
| 打分（SFT） | `python -m autotrain.code_corpus sft-score --corpus runs/code-sft` |
| 检查匿名性 | `sft-verify` / `verify` 会拒绝 manifest 中的路径标记 |

## 8. 大改：容量 + 语料（新尺子，已完成）

### 8.1 换尺子 = 所有旧报告作废

| | 旧家族 | 新家族 |
|---|---|---|
| 语料 | 64 MiB / 23.7M tokens | **512 MiB / 175.2M train + 9.8M holdout** |
| corpus_id | `7d2c1051a78b28f5` | `2609497795a6581b` |
| 词表 | 8192 | **16384**（11,642 子词） |
| 模型 | 4L/256d ≈ 5.2M | **12L/768d/12h ≈ 97.9M** |
| 精度 / 每步 tokens | fp32 / 32768 | bf16 / 16384 |
| 预算 | 4800 步 | **8000 步** |
| gen_hash | `619cf5d7ddb5e491` | `99a54a74b0f2fae6` |

`eval_lm_hash` / `gen_hash` / `sft_hash` **全部换代**。旧的 ppl 10.75、
`looped=0%`、各次 `IMPROVED` 报告一律 `INCOMPARABLE`，不能用来证明「变好了」。
新家族只能在自己的 step-0 baseline 上做 A/B。

配置 [wanxiang-code-large.yaml](<autotrain/configs/wanxiang-code-large.yaml>)，
语料 `runs/code-corpus-v2/`，SFT 数据 `runs/code-sft-v2/`
（`corpus_id=876366ca7de21392`，300,000 train / 17,120 holdout 样本、54.4M tokens）。

### 8.2 前置验证

- 100M smoke（20 步 bf16）：显存峰值 2.5 GiB、free 13.4 GiB，**无 OOM**
- [build.py](<autotrain/code_corpus/build.py>) 改为 `array('H')` 流式写盘，
  输出与旧 `np.asarray('<u2')` **字节完全一致**；否则 512 MiB 的
  Python `list[int]` 会占数 GiB 内存 OOM
- 编码吞吐 4.41 M chars/s → 512 MiB 约 2 分钟
- step-0 基线（新 `gen_hash`）：`looped=100%`、`distinct_1=5.66%`、
  `repeat_3gram_rate=85.7%`（随机权重，符合预期）
- `loss_spike_sigma=1e9` 关闭误触发检测器（见 4.0）

### 8.3 训练与结果（已完成）

- 首轮 8000 步预算在 **step 3900 被 `early_stop_patience=24` 提前中止**
  （val loss 仍在降，属平台期误判），只看 63.9M / 175.2M tokens。用同一
  config 从 step-3900 resume、运行时禁用 early stop，续训到 step 8000
  （131M tokens，0.75 遍语料），`val_loss 4.0477 → 3.6845`。
- 再从 step-8000 用 `runs/code-sft-v2/`（30 万 block 样本）做 answer-side
  SFT 1000 步 → step 9000，`val_loss 3.5452`。

四点对比（**全部在同一把新尺子上**：`eval_lm_hash=1c76c2574b4fdfe0`、
`gen_hash=99a54a74b0f2fae6`、`sft_hash=463f49491e8f687b`）：

| 指标 | step-0 | step-3900 | step-8000 | SFT step-9000 |
|---|---|---|---|---|
| eval_lm ppl | 19669 | 46.21 | **31.41** | 32.51 (`NO_CHANGE`) |
| sft answer CE | 9.889 | 4.185 | 3.844 | **3.622** |
| sft answer top-1 | 0.006% | 30.9% | 35.7% | **38.5%** |
| gen `looped` | 100% | 100% | **100%** | **100%** |
| gen `repeat_3gram_rate` | 85.7% | 86.0% | 78.9% | **75.3%** |
| gen `distinct_1` | 5.66% | 8.69% | 12.5% | **14.3%** |
| gen `max_repeat_run` | 38.6 | 3.66 | 2.31 | **1.50** |
| gen + 解码控制 `looped` | — | 0% | 0% | **0%** |
| gen + 解码控制 `distinct_1` | — | 45.9% | 51.8% | **53.3%** |

每个阶段 `eval_gen` A/B 都是 `IMPROVED`（5 改善 / 0 退化）；`eval_lm`
在 SFT 后为 `NO_CHANGE`（无灾难性遗忘）。

### 8.4 诚实结论

大改**成功了一半**：容量 20× + 语料 7.4× + 词表 2× 换来语言建模与
多样性的持续单调改善（ppl 19669 → 31.41、answer top-1 0 → 38.5%、
重复 3-gram 85.7% → 75.3%）。与旧家族不同，新家族的 SFT 没有让生成退化。

但**核心目标未达成**：`looped` 从 step-0 到 step-9000 始终是 **100%**，
greedy 样例仍是 `public void void void... override override...`；开解码控制
后输出虽多样（`public: return; } #if UNITY_EDITOR ...`），仍不是可靠代码。
**唯一能把循环压到 0 的还是推理层的解码控制，不是模型**。

这说明 autoregressive 循环是当前「小模型 + next-token 自监督 + 本地源码」
范式的固有失败，扩容到 98M / 512 MiB 仍不足以突破。要真正可用需要再大
几个数量级（模型、语料、训练预算）或换训练范式 —— 已超出「本地电脑上
能跑」的自举范围。按项目纪律，不把 teacher-forced ppl 下降包装成
「模型能写代码」。

当前最佳产物：`runs/code-large-sft/bundle`（对话）/
`runs/code-large/bundle_step8000`（补全）；推理时必须带解码控制参数（见 5.1）。

## 9. 换起点：开源底座 + QLoRA（新家族，链路已跑通）

前面 8 节证明的是「小模型 + next-token 自监督 + 本地源码」范式的边界：
扩容到 98M / 512 MiB 后 greedy `looped` 仍是 100%。真正的生产路径是
**拿一个已在大规模数据上预训练好的开源底座做领域适配**，本地源码正好
用作微调语料。这是一条**全新且不可比**的路线。

### 9.1 选型（含一处必须纠正的许可问题）

| | 主 | 兜底 | 排除 |
|---|---|---|---|
| 型号 | `Qwen2.5-Coder-7B-Instruct` | `Qwen2.5-Coder-1.5B-Instruct` | ~~3B~~ |
| 许可 | Apache 2.0 | Apache 2.0 | **Qwen RESEARCH LICENSE，仅非商用** |

> **纠正**：HF API 显示 3B 的 `cardData.license=other`，其 LICENSE 第 2 条
> 明确 "FOR NON-COMMERCIAL PURPOSES ONLY"、商用需单独申请。专有工程
> 不能用。7B / 1.5B 才是 `apache-2.0`。**选底座必须逐型号核许可**，
> 不能按系列名推断。

底座放在 `runs/qwen-base/{1.5b,7b}/`（不进 git）。两者 tokenizer 实测
完全相同（vocab 151643、eos 151645、chat template 文本与 id 一致），
所以同一份适配数据可共用；尺子仍按型号分开，不跨型号比较。

### 9.2 环境与模块（已完成）

`.venv-cuda` 装 `transformers 5.24.1 / peft 0.22.0 / bitsandbytes 0.50.2 /
accelerate 1.16.0 / datasets 5.2.0`。实测 **bitsandbytes 4-bit NF4 在
Windows + cu130 上可正常加载并前向**，1.5B 显存仅 ~2 GB、free 14 GB。

新增 [autotrain/qwen_sft](<autotrain/qwen_sft>)：

| 模块 | 作用 |
|---|---|
| `data.py` | 用**底座 tokenizer** 从脱敏源 jsonl 重建指令样本；文档级冻结 train/holdout、block 对齐、answer 去重、answer-side labels(-100)、seal/verify、路径泄漏检查 |
| `train.py` | 4-bit NF4 QLoRA：r=16/alpha=32、7 个 target module、bf16、gradient checkpointing、paged 8-bit Adam、batch1×accum8 |
| `ruler.py` | 重建冻结生成质量尺子，复用 `eval_gen.sequence_metrics` / `compare` 纯函数 |
| `__main__.py` | CLI：`build` / `verify` / `train` / `score` / `compare` |

**不能复用** `runs/code-sft-v2/*.bin`：那是 16384 词表自定义 BPE 编码的，
与底座 15 万词表完全不兼容。数据 `runs/qwen-sft/`
（`corpus_id=5868c5114c16c0f4`，20,000 train / 19,183 holdout，
文档顺序确定性 shuffle 避免截断偏差）。

### 9.3 已验证的链路线

- 4-bit 加载 + 前向：`logits (1,5,151936)`，无 OOM
- QLoRA smoke（1.5B / 5 步 / 32 样本）：loss 2.90 → 1.51、可训练参数
  18.5M、adapter 保存成功、2.25 s/步
- 7B 4-bit 生成同样跑通，显存无压力
- 数据 `verify`、尺子 `score` / `compare` 均通过

### 9.4 基线（新尺子，与前 8 节所有报告 INCOMPARABLE）

| 尺子 / 模型 | looped | distinct_1 | repeat_3gram | max_repeat_run |
|---|---|---|---|---|
| `gen_hash=99086acf0b70cd80` 1.5B 裸模型 greedy | 50.0% | 46.3% | 28.6% | 1.125 |
| 同上 + 解码控制 | **0%** | **78.9%** | 0% | 1.125 |
| `gen_hash=b7052f66cc3ac3a1` 7B 裸模型 greedy | 62.5% | 48.3% | 27.6% | 1.125 |
| 同上 + 解码控制 | **0%** | **80.9%** | 0% | 1.0 |

**最重要的观察**：裸底座的 `looped` 只有 50–63%，而从零训到 98M / step-9000
的模型始终是 100%。两把尺子不可比，但量级差异直接印证了「换起点」是对的。

smoke adapter（5 步 LoRA）在尺子上判 `REGRESSED`（0 改善 / 4 退化），
幅度极小且符合预期 —— 它只是链路验证，不是训练结论。

### 9.5 100 步 QLoRA 的真实结论（两把尺子缺一不可）

在 `runs/qwen-sft/` 上训 100 步（241 秒，loss 1.502，LoRA 18.5M 参数），
在同一把尺子上做 A/B。

**生成尺子 `gen_hash=99086acf0b70cd80`（管重复）**

| | 裸底座 | +100 步 LoRA |
|---|---|---|
| greedy OFF `looped` | 50.0% | 56.3% |
| greedy OFF `distinct_1` | 46.3% | 48.1% |
| `eval_gen` verdict (OFF) | — | `REGRESSED`（1 改善 / 4 退化） |
| 解码控制 ON `looped` | 0% | 0% |
| `eval_gen` verdict (ON) | — | `IMPROVED`（3/0，幅度极小） |

**适配尺子 `adapt_hash=ddaa2fe4db96dc98`（管领域适配）**

| | 裸底座 | +100 步 LoRA |
|---|---|---|
| answer CE | 1.7715 | **1.3231** |
| answer top-1 | 68.4% | **73.0%** |
| verdict | — | **`IMPROVED`** |

**结论**：100 步 QLoRA 让模型在本地代码分布上的 answer CE 降 25%、
top-1 升 4.7pp —— 领域适配**确实学到了东西**；但它没有改善「重复退化」
（OFF 甚至轻微变差）。这两把尺子测的是不同问题：**只用 `eval_gen` 会误判
「微调没用」，只用 `adapt` 会漏掉「生成仍会循环」**。

另一个量级参照：裸底座 answer top-1 就有 68.4%，而从零训到 98M /
step-9000 的模型在它自己的 sft 尺子上只有 38.5%（跨尺子不可比，但
印证换起点的价值）。

### 9.6 还没做的

7B 上的正式 QLoRA 训练（只验证了 4-bit 生成可跑）、更多步数与数据、
GGUF/Ollama 或 llama.cpp 本地部署（本机均未安装）、以及「指令遵循 /
代码正确性」这类尺子 —— 当前两把尺子都**不测「写得对不对」**。
**换底座后旧家族所有报告作废**，只能用本节 `gen_hash` / `adapt_hash` 做 A/B。
