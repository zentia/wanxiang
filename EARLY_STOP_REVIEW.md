# Early-stop 判据复核：尾部 val 到底可不可信

> 复核对象：`LM_QUALITY.md` 第 7 节提出的待办「若要突破 4800 步，应先复核基于尾部 val 的 early-stop 判据」。
> 复核日期：2026-11-24。结论：**判据不可信，但不是我原先以为的那个原因**——见下方「结论修正」。

---

## 0. 结论先行（并修正我之前的说法）

| 我之前的说法 | 复核后的真实情况 |
|---|---|
| 「`final_val_loss=4.5991` 高于 ext2 的 4.0003，只是因为 val 集切自各自训练流尾部、不是同一批数据，跨实验不可比」 | **对，但不完整、且没抓到主因**。跨实验确实不可比，但导致 ext3 早停误判的是**单次训练内部 val 测量本身的噪声** |
| 「early stop 值得复核，可能停早了」 | **确认停早了，且根因已定位**：`_validate` 的 4 个 val 窗口几乎完全重叠，有效独立样本只有约 4,100 token，把一条仍在下降的曲线误读成停滞 |

**判定：early stop 在 step 4350 触发是 `PREMATURE`（过早），但模型并未收敛。**
用分散窗口重算，val loss 到 4350 仍在下降、best 稳定在终点；冻结尺子上困惑度同期从 36.67 降到 34.91，最优在 step 4300。

---

## 1. 机制复核（读代码得到的确切事实）

### 1.1 val 段是物理隔离的，没有被训练污染

`autotrain/trainer/data.py`：

```python
self.n_val  = max(1, int(n_tokens * val_fraction))
self.n_train = n_tokens - self.n_val

get_train_batch(cursor): if cursor + need > self.n_train: cursor = 0   # wrap 只在 [0, n_train)
get_val_batch(offset):   start = self.n_train + (offset % ...)
```

训练游标 wrap 只覆盖 `[0, n_train)`，`get_val_batch` 只读 `[n_train, n_total)`。
**所以第 3 篇那类「留白被训练游标反复扫过」的污染，在这里不存在。** 这一点必须先澄清：
这次的问题是「测量太吵」，不是「测量被污染」。两者是完全不同的失效模式。

### 1.2 但 val 段随每次扩语料而更换

| run | n_total | n_train | val 段起点 | val 段大小 |
|---|---:|---:|---:|---:|
| lm-scaled2 | 11,578,205 | 11,346,641 | 11,346,641 | 231,564 |
| lm-scaled3 | 15,609,856 | 15,297,659 | 15,297,659 | 312,197 |

两次 val 段起点相差约 395 万 token，是**完全不同的两批文档**。
→ **训练内部 val loss 跨实验不可比**（这条纪律成立，继续保留）；
但它不能解释「为什么 ext3 自己会在 4350 停」，因为早停只看**同一次训练内部**的相对变化。

### 1.3 真正的缺陷：4 个 val 窗口几乎完全重叠

`runner._validate(n_batches=4)` 调用 `get_val_batch(..., offset=i)`，`i = 0,1,2,3`。
而 `get_val_batch` 把 `offset` 当作**token 起始偏移**（不是第 i 个独立窗口）：

```
offset=0 -> start = 15,297,659
offset=1 -> start = 15,297,660
offset=2 -> start = 15,297,661
offset=3 -> start = 15,297,662
```

4 个窗口起点只差 **3 个 token**，去重后有效独立样本 ≈ **4,100 token**，
占 val 段 **1.31%**、占全训练流 **0.0268%**。字面上的「4 × 4096 = 16,384」是假象。

**early stop 的全部依据，就是在反复读同一段约 4K token。** 这是本次复核最重要的发现。

### 1.4 判据本身过于敏感

`EarlyStopTracker.update`：`val_loss < best - 1e-5` 才算改善，`patience=12`，
`val_every_steps=50` → 连续 12 次评估（600 步）无改善即停。
`min_delta=1e-5` 意味着任何 0.00001 级的抖动都算「没变好」。配合 1.3 的极小采样面，
噪声可以轻易凑满 12 次 bad round。

---

## 2. 训练日志复现（确认触发过程）

解析 `runs/lm-scaled3/train.log` 共 88 个 val 评估点：

- 全程最低 `val_loss = 4.5739 @ step 3750`
- 之后 val 在 `4.5868 ~ 4.6653` 之间**震荡**（不是单调恶化，甚至回到过 4.5868）
- `final = 4.5991 @ 4350`，从 best 起又跑 600 步、回升 +0.0252
- 模拟 `EarlyStopTracker(patience=12, min_delta=1e-5)`：在 **step 4350 恰好 bad_rounds=12**，与日志 `CONVERGED_EARLY` 完全一致

---

## 3. 决定性检验：换采样位置，best 就漂移

在 6 个不同的 val 采样位置、每个位置用 8 个分散窗口重算各 checkpoint 的 val loss
（`runs/lm-scaled3/reports/val_robustness.json`）：

| 采样位置 | best-step | 该位置 best val |
|---|---:|---:|
| orig | **4350** | 3.9950 |
| 20% | **4350** | 4.0476 |
| 40% | **4350** | 3.7460 |
| 60% | **4350** | 4.3028 |
| 80% | **4350** | 3.6762 |
| 99% | **4350** | 4.0592 |

（各列绝对水平不同是正常现象：val 段内不同位置的文本难度本就不同；判收敛只看**同一列内的相对趋势**。）

**6/6 位置 best 都在终点 step 4350，没有一处出现「3750 之后停滞」。**
对比三种采样策略在同一批 checkpoint 上的曲线：

| step | 现状(4 重叠) | 4 分散 | 8 分散 |
|---:|---:|---:|---:|
| 3700 | 4.6366 | 3.9536 | 4.0348 |
| 3800 | **4.5920（现状 best）** | 3.9628 | 4.0485 |
| 4000 | 4.6438 | 3.9421 | 4.0235 |
| 4200 | 4.6060 | 3.9172 | 4.0024 |
| 4350 | 4.5991 | **3.9071** | **3.9925** |

现状采样在 3800 见底后震荡；分散采样则一路下降到 4350。
**「best@3750」是采样噪声造出来的，不是模型收敛信号。**

> 说明：该策略对比只用了整百步 checkpoint，评估密度约为真实的一半，
> 因此 patience 绝对计数不精确；但 best 位置漂移的方向是确定的，且与冻结尺子结论一致。

---

## 4. 冻结尺子上的独立佐证

在 `eval_lm_hash=2b70b6f6793432ab` 上密评 early-stop 区间 9 个 checkpoint
（`runs/lm-scaled3/reports/earlystop_diagnosis.json`）：

| step | 冻结 holdout 困惑度 | 训练内部 val_loss |
|---:|---:|---:|
| 3600 | 38.265 | 4.6677 |
| 3700 | 37.669 | 4.6366 |
| 3800 | 36.668 | 4.5920 |
| 3900 | 36.337 | 4.6653 |
| 4000 | 35.763 | 4.6438 |
| 4100 | 36.318 | 4.6138 |
| 4200 | 36.347 | 4.6060 |
| **4300** | **34.695（冻结尺子最优）** | 4.5943 |
| 4350 | 34.914 | 4.5991 |

- 在 train-val 判定「停滞」的 3750→4350 区间，**冻结困惑度净降 1.75**（36.67 → 34.91），还在改善。
- 两把尺子在尾部几乎不相关：**`r(val_loss, frozen_ppl) = +0.285`**。
  例：step3900 val_loss 变差（4.6653）但冻结 ppl 反而更好（36.337）；
  step4300 冻结 ppl 全程最优，val_loss 却仍高于 3750 的 best。
- 冻结尺子最优在 **step 4300**，甚至不是被 early stop 选中的 4350。

---

## 5. 修复方案（最小改动，不碰训练核心语义）

按优先级，前两条是必须，第三条是可选加固：

| # | 改动 | 位置 | 验收标准 |
|---|---|---|---|
| 1 | `get_val_batch` 的 `offset` 语义从「token 偏移」改为「窗口序号」，4 个窗口在 val 段内**等距分散、互不重叠** | `trainer/data.py` | 4 窗口联合覆盖 ≥ val 段 80%；任意两窗口 token 区间交集为空 |
| 2 | 把 `n_batches` 从 4 提到 8~16，或让一次 validate 覆盖全 val 段 | `trainer/runner.py::_validate` | 单次 val 有效独立样本 ≥ 40K token（当前 4.1K 的 10×） |
| 3 | `min_delta` 从绝对 `1e-5` 改为相对阈值（如 `best * 0.001`），或对 val loss 做滑动平均再判停滞 | `trainer/monitor.py` | 用历史 ext3 checkpoint 回放：分散采样 + 新阈值下不应在 4350 触发早停 |

配套建议：

- **不要把冻结 holdout 接进训练内部做 early stop**。那会把最终评测尺子变成训练信号，
  重蹈「拿测量当目标」的覆辙；冻结尺子只用于事后报告与 A/B。
- 修好采样后，**训练内部 val loss 依然跨实验不可比**（val 段随数据流更换），
  这条纪律不变；它能变的只是「单次训练内部的收敛判定更可靠」。
- 修完需要一次干净重训来验证：预算真正跑到 4800、并在同 hash 上看困惑度是否继续下降。

---

## 6. 复现命令

```bash
cd <repo-root>
.venv/bin/python runs/lm-scaled3/diagnose_earlystop.py     # 冻结尺子密评 + 诊断报告
# 多位置稳健性 / 三策略对比脚本见本文件第 3 节，结果落 val_robustness.json
```

产物：
- `runs/lm-scaled3/diagnose_earlystop.py` / `diagnose_earlystop.log`
- `runs/lm-scaled3/reports/earlystop_diagnosis.json`
- `runs/lm-scaled3/reports/val_robustness.json`
- `runs/lm-scaled3/reports/diag_step*.json`（9 个冻结尺子快照）

---

## 7. 修复实施与干净重训验证（2026-11-24）

按第 5 节优先级实施，但**刻意只做 #1 + #2，暂缓 #3**，理由见 7.2。

### 7.1 已落地的改动

| # | 文件 | 改动 |
|---|---|---|
| 1 | `autotrain/trainer/data.py` | 新增 `TokenDataset.val_window_count(need, requested)`；`get_val_batch` 在 `n_windows>1` 时把 `offset` 当窗口序号，窗口数钳制到「val 段内能放下互不重叠窗口」的上限，窗口等距铺满尾部；`n_windows=1` 保留旧 token 偏移语义 |
| 2 | `autotrain/trainer/runner.py` | 新增代码常量 `VAL_WINDOWS=16`（**不进 config**，避免 `config_hash` 变化）；`_validate` 默认采样 `val_window_count(need, 16)` 个分散窗口 |
| — | `tests/test_data.py` / `tests/test_runner.py` | 新增 4 个回归测试：真实尺度 16 窗口不重叠且覆盖 ≥80% 尾部、小 val 段自适应钳制、`n_windows=1` 旧语义、`_validate` 确实走分散窗口模式 |

验收实测：
- 真实尺度 `n_val=312,197`、`need=4097` → `val_window_count=16`，
  窗口起点间隔 20,540 token（≫4,097，**互不重叠**），联合覆盖
  `15×20,540+4,097 = 312,197` token ≈ **100% 尾部**，有效独立样本
  从 4.1K 提升到 ~31 万（约 **76×**，远超第 5 节要求的 10× / 40K）。
- 全量 `pytest`：**211 passed**（修复前 207 + 4 新增）。

### 7.2 为什么暂缓 #3（相对 min_delta / 平滑）

一次只改一个变量，才能把重训结果归因到采样修复本身。而且复核后确认：
`min_delta=1e-5` 是「改善门槛」，取值已经很宽松，并不是 premature 的推手；
真正的推手是 4 个重叠窗口锁死了几篇特定文档，使 val loss 在 best@3750
后持续回升、耗尽 12 次 patience。采样面扩大 76× 后 val loss 已足够平滑，
无需叠加阈值改动。**若干净重训仍过早停止，再单独评估 #3。**

### 7.3 干净重训协议（单变量）

`runs/lm-fixedval/`：
- `config.yaml`、`train_tokens.bin` 与 `runs/lm-scaled3` **字节级一致**
  （sha256 `713c1514…` / `40595681…`，`config_hash` 同为 `c08fdb6b…`）；
- 从空 `checkpoints/` 起、step 0 干净重训到 **4800 步**，不 resume；
- `eval_lm_hash` 仍冻结 `2b70b6f6793432ab`，冻结 holdout **不参与**训练内部早停。

关键判据（这正是干净重训的价值）：修复是**纯测量侧**——`_validate` 在
`torch.no_grad()` 下运行，不更新权重 / 优化器 / scheduler / `data_cursor`，
也不影响基于 train loss 的 spike 检测。因此：
1. **训练轨迹必须与 ext3 逐位一致**：实测 step 1 的 train loss 两边都是
   `9.0933`（而 val loss 已从 `9.0639` 变为 `9.0380`，证明换的只是尺子）；
   `score.py` 会断言 step3600/4000/4350 的冻结困惑度与 ext3 差 <1e-6。
2. **4800 尾部才是新证据**：`reached_full_budget`、是否仍触发 early stop、
   以及 step4350→final 的冻结困惑度是否继续下降。

产物：`runs/lm-fixedval/{prepare.py,score.py,provenance.json,train.log}` 与
`reports/lm_fixedval_curve.json`。结果回填前，第 0 节结论仍以
`PREMATURE` 为准；本 run 是验证修复、不是刷新主线最佳。

### 7.4 复现命令

```bash
cd <repo-root>
.venv/bin/python runs/lm-fixedval/prepare.py
.venv/bin/python -m autotrain.trainer \
  --config runs/lm-fixedval/config.yaml \
  --data runs/lm-fixedval/train_tokens.bin \
  --out runs/lm-fixedval
.venv/bin/python runs/lm-fixedval/score.py
```
