# LM Quality：三条质量路线的严格 A/B

> 目标：在**同一把 eval_lm 尺子**上，分别验证更多数据/步数、模型容量、
> 编码粒度对通用语言建模能力的影响。三条路线全部与 `runs/lm-scaled`
> step1200（ppl 106.82）对比，eval corpus/holdout/tokenizer/manifest 全部
> 字节不动，`eval_lm_hash=2b70b6f6793432ab`。

## 1. 统一纪律

- 评测只读 `runs/corpus/lm`；训练语料无论怎么扩，都不碰 holdout 文档。
- 所有报告都带同一个 `eval_lm_hash`；跨 hash / 跨家族比较直接拒绝。
- checkpoint 按文件名中的**数字 step** 排序，避免 `900` 字典序大于 `1200`。
- 续写只是诊断，不计入分数；困惑度降不等于生成流畅。

## 2. 三条路线与控制变量

| 路线 | 训练数据 | 模型 | 步数 | 唯一变量 |
|------|----------|------|------|----------|
| ext2 长训 | base + ext1 + ext2（rows 0..1999 去 holdout） | 4L/256/4H | 2400 | 更多 disjoint 文档 + 更多步数 |
| 扩容量 | base + ext1（同 lm-scaled） | 8L/512/8H | 1200 | 模型容量 5.15M → 29.17M |
| 同 hash BPE | base + ext1（同 lm-scaled） | 4L/256/4H | 1200 | train tokenizer char → BPE |

## 3. 为什么 BPE 还能“同 hash”

常规 BPE 评测会换 tokenizer，必然改 `tokenizer_sha256`，从而改
`eval_lm_hash`；那只能做跨族诊断，不能与 char 模型严格比较。

这里改用 **char-ID-aligned BPE**：

- BPE 从冻结 base 文档学习。冻结 char tokenizer 的 4 个特殊 token 与
  7,808 个已用字符 id（4..7811）在 BPE 中**逐位相同**；
- BPE 的 380 个 subword piece 精确占用 char tokenizer 原来的
  `<unused_0..379>` 空槽（7812..8191）；
- `score_bundle` 永远读 `runs/corpus/lm/tokenizer.json` 与
  `holdout_tokens.bin`，**不重新分词 holdout**。因此 eval 规格与 hash 不变；
- 这是严格但保守的 char-level 困惑度：BPE 模型若把概率分给 subword id，
  在 char 目标流上就是答错，理应受罚。

生成诊断才用 `runs/lm-bpe/tokenizer.json` 解码，通过
`eval_lm generate --tokenizer` 覆盖；该覆盖**只作用于诊断，不影响打分**。

## 4. 数据与 seal

- ext2：`wiki_zh_ext2`，rows 1200..1999，800 篇，sha256 `3326ee6b…`；
  与 base（0..399）和 ext1（400..1199）按文档 id 零重叠。
- ext2 训练流：7,715,581 → 11,578,205 token，扩展流以 lm-scaled 流为
  字节前缀，UNK 4,490 / 3,862,624 = 0.1162%。
- BPE 训练流：base train 7,808 字符 id 与冻结流逐 token 对齐；
  BPE token 7,172,936，等效 char token 7,715,581，UNK 0.0707%。
- 冻结评测量：holdout 119,500 token，`eval_lm_hash=2b70b6f6793432ab`。

## 5. 结果

### 5.1 ext2 长训（完成）

step0→2400，train token 11,578,205，seen 78,643,200 token，validation
loss **4.0003**，耗时 1,571 秒 / 50.1k tok/s。

| 指标 | lm-scaled step1200 | ext2 step2400 | 变化 |
|------|---:|---:|---|
| holdout 交叉熵 | 4.6711 | 4.0363 | −0.6349 |
| **holdout 困惑度** | **106.82** | **56.62** | **↓47.0%** |

`comparable=true`，hash 未变，判定 **IMPROVED**。续写从
“人类的人类…”变为“否定的。 在这些语言中，这些语言中…”，重复仍在，
但已出现标点与更长短语结构。

### 5.2 扩容量（完成）

参数量 5.15M → **29.17M**（8L/512/8H），同样 ext1 文档、1200 步、
39,321,600 token，final val loss **4.7029**（小模型 4.9186），
耗时 3,740 秒 / 10.5k tok/s。

| 指标 | 4L256 step1200 | 8L512 step1200 | 变化 |
|------|---:|---:|---|
| holdout 交叉熵 | 4.6711 | 4.4729 | −0.1982 |
| **holdout 困惑度** | **106.82** | **87.61** | **↓18.0%** |

`comparable=true`，hash 未变，判定 **IMPROVED**。容量有效，但单位算力的
收益低于继续加数据/步数：5.7× 参数换来 18% ppl 下降，而 ext2 长训用
2× 步数 + 1.5× 文档换来 47% 下降。续写仍重复，但出现“1998年”等实体片段。

### 5.3 同 hash BPE（完成，负结果）

BPE token 7,172,936（等效 char token 7,715,581），1200 步，final val
loss **5.3141**（char 同数据 4.9186），耗时 846 秒 / 46.5k tok/s。
必须强调：这是 char-ID-aligned BPE 在**同一冻结 char holdout** 上的打分，
所以 hash 未变、可比成立，但 BPE 模型输出分布里有 380 个 subword id，
在 char 目标上统统算错，属于严格且保守的协议。

| 指标 | char step1200 | char-id-aligned BPE step1200 | 变化 |
|------|---:|---:|---|
| holdout 交叉熵 | 4.6711 | 5.2752 | +0.6041（更差） |
| **holdout 困惑度** | **106.82** | **195.43** | **↑83.0%（更差）** |

`comparable=true`，hash 未变，`perplexity_delta = -88.61`。
严格比较器只把正向提升判为 `IMPROVED`，因此报告 `NO_CHANGE`；**数值上是
明确退化，不是持平**。续写变成“199年，199年，198年…”，仍是重复退化。

## 6. 结论与下一步

| 路线 | ppl 变化 | 相对 step1200 | 判定 |
|------|---:|---:|------|
| ext2 长训 | 106.82 → **56.62** | ↓47.0% | `IMPROVED` |
| 扩容量 | 106.82 → **87.61** | ↓18.0% | `IMPROVED` |
| 同 hash BPE | 106.82 → **195.43** | ↑83.0% | `NO_CHANGE`（实为退化） |

优先级判断：

1. **继续堆数据 + 步数是当前最有效路线**：ext2 已把 ppl 打到 56.62，
   且 2× 步数 + 1.5× 文档的成本低于 5.7× 参数。
2. **容量有效但性价比一般**：29M 模型只降 18%，且与更多数据/步数正交，
   可在数据量继续扩大后再重试。
3. **BPE 在当前评估语义下被证伪**：char-id-aligned 方案保持了同 hash 可比性，
   但 subword 输出在 char 目标上受罚，ppl 显著变差；不进入主线。
   这也是对历史 eval3 那次“BPE NO_CHANGE”的更严格复验。

## 7. ext3 加长训练（完成：继续下降，边际递减 + early stop）

按第 6 节的优先级继续：追加 `wiki_zh_ext3`（rows 2000..2799，800 篇，
sha `1bf00933…`），四块文档 id 两两零重叠；训练流 11,578,205 →
**15,609,856 token**，以 `lm-scaled2` 流为字节前缀，UNK 0.1505%。
架构仍是 4L/256/4H，`eval_lm_hash=2b70b6f6793432ab` 字节冻结。

**预算 4800 步实际在 step 4350 触发 `CONVERGED_EARLY`**（val loss 连续
12 次评估停滞），没有 step4800 checkpoint。因此按落盘实际评测
step3600 / 4000 / 4350 三个点，并额外做 step3600→4350 的 plateau 检查：

| checkpoint | holdout 困惑度 | vs step2400 (56.62) | 相邻段降幅 |
|---:|---:|---|---:|
| step3600 | 38.27 | `IMPROVED` | — |
| step4000 | 35.76 | `IMPROVED` | −2.51 |
| **step4350（收敛点）** | **34.91** | **↓38.3%，`IMPROVED`** | −0.85 |

step3600→4350 严格比较仍 `IMPROVED`（38.27 → 34.91，delta 3.35），
**没有 plateau、没有回归**，但相邻段降幅从 2.51 收窄到 0.85，边际收益
明显递减。CE 4.0363 → 3.5529。耗时 2,957 秒 / 48.2k tok/s，
seen 142.5M token。

> **不要被训练内部的 val loss 误导**：本次 `final_val_loss=4.5991` 反而
> 高于 scaled2 的 4.0003，`early stop` 也是基于它触发的。但
> `val_fraction=0.02` 切的是**各自训练流的尾部**，ext3 流更长、尾部是
> 全新文档，两次 val 集根本不是同一批数据 —— 训练内部 val loss 跨实验
> **不可比**。唯一有效判据是冻结 holdout 困惑度，它在 4350 步仍在下降。
>
> **后续复核（2026-11-24，见 `EARLY_STOP_REVIEW.md`）修正了上面这段的侧重**：
> 跨实验不可比这条成立，但导致本次早停误判的**主因是单次训练内部 val
> 测量的噪声**——`_validate` 的 4 个窗口只相差 1 个 token、几乎完全重叠，
> 有效独立样本仅约 4,100 token（全训练流的 0.027%），加上 `min_delta=1e-5`
> 过于敏感，把一条仍在下降的曲线误读成停滞、耗尽 12 次 patience。换成分散
> 窗口重算，6/6 采样位置的 best 都稳定在终点 step 4350，冻结困惑度同期
> 净降 1.75、最优在 step 4300。判定：**early stop 是 `PREMATURE`，模型没收敛。**
> 修复方向是改 val 采样语义，不要用冻结 holdout 参与早停。
>
> **修复已实施（2026-11-24，见 `EARLY_STOP_REVIEW.md` 第 7 节）**：
> `get_val_batch` 的 `offset` 改为窗口序号，一次 validate 用 `VAL_WINDOWS=16`
> 个等距、互不重叠的窗口铺满整个 31 万 token 尾部（有效样本从 4.1K 提升到
> ~31 万，约 76×）；新增 4 个回归测试，全量 211 passed。`min_delta`/patience
> 本轮**刻意不动**（单变量纪律，便于归因），仅当重训仍过早停止再评估。
> 已启动 `runs/lm-fixedval` 干净重训到 4800 步验证，结果待回填。

续写诊断（step4350）：`"自然语言处理是"` →
`"否有一个不同的，而且是一个不同的语言，而是一个语言的语言…"`，
仍是重复退化，但短语结构比 step2400 更长。困惑度下降 ≠ 生成流畅，
这条边界不变。

### 更新后的结论

1. **加 disjoint 数据 + 步数仍然有效**：step2400 → step4350 把 ppl 从 56.62
   打到 34.91（↓38.3%），当前最佳更新为 **step4350 / ppl 34.91**。
2. **但已进入边际递减区**：末段 750 步只降 0.85，且训练内部 val 提前收敛。
   继续纯堆步数的性价比在下降，下一轮更值得动的是**数据多样性 / 更大规模
   模型 / 解码策略**，而不是把同一配方再拉长。
3. **early stop 的判据已复核并修复**：根因是 val 采样窗口重叠（有效样本
   4.1K token），已改为 16 个分散窗口；`runs/lm-fixedval` 正在用同样的
   config / train 流 / seed 干净重训到 4800 步。由于修复是纯测量侧
   （`no_grad`，不动权重与游标），step3600/4000/4350 的冻结困惑度必须与
   本表**逐位一致**，4800 尾部才是新证据。结果待回填。
