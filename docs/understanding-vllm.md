# 从零理解 vLLM —— 以 nano-vllm 为教材

> **这份文档写给谁**：你会 Python、大致了解 Transformer 是什么（知道 attention、embedding 这些词），但不了解 LLM 推理系统，想搞懂 vLLM 为什么快、它到底做了什么。
>
> **怎么读**：第 1、2 章是概念铺垫，可以脱离代码阅读；第 3 章给出项目全景；第 4 章是源码走读（重头戏，建议对照仓库一起读）；第 5 章把所有环节串成一条时间线；第 6~9 章是查表、对照与练习。所有代码引用格式为 `文件路径:行号`，可直接定位。
>
> **项目背景**：nano-vllm 是 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，一个用约 1200 行 Python 从零实现的 vLLM 复刻，只支持 Qwen3 系列模型的离线批量推理，但保留 vLLM 最核心的四大技术：**Continuous Batching、PagedAttention（分页 KV Cache）、Prefix Caching、Chunked Prefill**，外加 Tensor Parallelism、CUDA Graph、FlashAttention 集成。在 RTX 4070 Laptop + Qwen3-0.6B 的基准上，吞吐量与 vLLM 持平甚至略胜（1434 vs 1362 tokens/s，见 `README.md`）。

---

## 0. 太长不看版（一页速览）

如果你只有 3 分钟，记住这张图和这几句话：

```
                        ┌────────────────────────────────────────────┐
                        │                LLM / LLMEngine             │
                        │  generate() → add_request() → step() 循环   │
                        └──────┬───────────────────────────┬─────────┘
                               │                           │
                 ┌─────────────▼────────────┐   ┌──────────▼───────────┐
                 │        Scheduler         │   │      ModelRunner     │
                 │  waiting / running 两个   │   │  把序列列表打包成     │
                 │  队列，决定"这一步算哪些   │◄──┤  GPU 张量，调用模型， │
                 │  序列的哪些 token"        │   │  返回采样出的 token   │
                 └─────────────┬────────────┘   └──────────┬───────────┘
                               │                           │
                 ┌─────────────▼────────────┐   ┌──────────▼───────────┐
                 │      BlockManager        │   │   Qwen3ForCausalLM   │
                 │  显存的"操作系统页表"：    │   │  + Attention(Flash)  │
                 │  分页 KV Cache 的分配、    │   │  + 各类 TP 线性层     │
                 │  共享、前缀哈希缓存        │   │  + Sampler           │
                 └──────────────────────────┘   └──────────────────────┘
```

1. **LLM 推理分两个阶段**：Prefill（一次吃进整个 prompt，算力密集）和 Decode（一次吐一个 token，访存密集）。两者的性能优化手段完全不同。
2. **KV Cache** 把注意力历史存进显存，避免每步重算，但显存会爆——于是 vLLM 把它像操作系统内存一样**分页**（PagedAttention），按块分配、按块共享。
3. **Continuous Batching** 让"先生成的序列先退出、新序列随时插入"，GPU 永远满批运转，而不是等一整批一起结束。
4. **Prefix Caching** 利用"很多请求共享相同前缀"（比如系统提示词），把前缀的 KV 块做哈希登记，后续请求直接复用，跳过重复计算。
5. **Chunked Prefill** 把超长 prompt 切成多轮小块，避免一次 prefill 长时间霸占 GPU、饿死 decode。
6. 每一步（step）= Scheduler 挑出本轮要算的 token → ModelRunner 打包成张量跑一遍模型 → Sampler 采样出下一批 token → Scheduler 记账（更新 KV 块、判断序列是否结束）。

---

## 1. 为什么需要"推理引擎"

### 1.1 LLM 是怎么生成文字的

LLM 本质上是一个"下一个 token 预测器"：给它一串 token（词的碎片），它输出一个覆盖整个词表的概率分布。所谓"生成"，就是把「预测 → 采样 → 拼回输入 → 再预测」循环往复：

```
输入: [A, B, C]          → 模型 → 概率分布 P₁ → 采样出 D
输入: [A, B, C, D]       → 模型 → 概率分布 P₂ → 采样出 E
输入: [A, B, C, D, E]    → 模型 → 概率分布 P₃ → 采样出 F
...直到采样出结束符 EOS 或达到长度上限
```

这就是**自回归（autoregressive）**生成。nano-vllm 中这个循环的最外层在 `LLMEngine.step()`（`nanovllm/engine/llm_engine.py:49`）：每调用一次 `step()`，所有活跃序列各自前进一个 token（或者新序列完成一次 prefill）。

### 1.2 Prefill 与 Decode：两种截然不同的计算

上面笼统的循环里其实藏着两种形态差异极大的计算：

| | **Prefill（预填充）** | **Decode（解码）** |
|---|---|---|
| 输入长度 | 整个 prompt，几十到几千 token | 每个序列 1 个 token |
| 并行度 | 所有 prompt token 同时算 | 一个序列一步只算 1 个 token |
| 瓶颈 | **计算**（矩阵乘满载 GPU） | **显存带宽**（读 KV Cache 比算乘法慢得多） |
| 每步产出 | 每序列 1 个有效 logits（只有末位置需要采样） | 每序列 1 个 token |
| 优化方向 | 大 GEMM、少启动开销、别浪费算 logits | 大 batch 摊薄带宽、CUDA Graph、paged 读缓存 |

这个区分是理解一切推理引擎设计的起点。注意一个容易被忽略的事实：**prefill 只需要每个序列最后一个位置的 logits**（采样只需要"下一个 token"的分布），中间位置算 logits 纯属浪费——nano-vllm 对此有个精巧的处理（见 4.7.5 节 `ParallelLMHead`）。

在 nano-vllm 里，`Scheduler.schedule()` 每次返回 `(seqs, is_prefill)` 二元组（`nanovllm/engine/scheduler.py:25`），这个布尔值决定了 `ModelRunner` 走哪条数据准备路径（`prepare_prefill` 还是 `prepare_decode`，`nanovllm/engine/model_runner.py:129, 172`）和 Attention 层调用哪个 FlashAttention API（`nanovllm/layers/attention.py:64-74`）。

### 1.3 KV Cache：用显存换速度

Transformer 的自注意力要求每个新 token **看到之前所有 token 的 Key/Value 向量**。如果不用缓存，生成第 1000 个 token 时要重算前面 999 个 token 在每一层的 K/V——计算量随长度平方增长。

**KV Cache 的做法**：第一次算出每个 token 的 K/V 后就存进显存，之后每步只算新 token 自己的 K/V，注意力计算时直接读缓存。代价是显存：

```
KV Cache 大小 = 2 (K和V) × 层数 × 序列长度 × KV头数 × 头维度 × 每元素字节数
```

以本仓库自带的 Qwen3-0.6B（`Qwen3-0.6B/config.json`：28 层、8 个 KV 头、头维度 128、bfloat16）为例，**每条序列每 1000 个 token 要吃掉约 114 MB 显存**：

```
2 × 28 × 1000 × 8 × 128 × 2 B = 114,688,000 B ≈ 109 MB
```

服务几十条并发、每条几千 token 的请求，KV Cache 轻松吃掉几个 GB——比模型权重本身（0.6B 参数 ≈ 1.2 GB）还大。**推理引擎的显存管理，本质上就是在管理这块 Cache**。nano-vllm 中对应的张量在 `ModelRunner.allocate_kv_cache()`（`nanovllm/engine/model_runner.py:103`）里分配。

### 1.4 朴素方案的三个瓶颈

如果直接用 HuggingFace `transformers` 的 `model.generate()`，会遇到三个问题，vLLM 的全部设计都在回答它们：

**瓶颈一：显存浪费与碎片。** 朴素做法为每条序列预分配一段连续显存（按最大长度预留）。假设上限 4096，一条实际只有 200 token 的请求也占着 4096 的空间——内部碎片可以浪费 60%~80% 的显存；用动态增长又会产生外部碎片。**→ 答案是 PagedAttention（2.2 节）。**

**瓶颈二：批处理不连续。** 静态批处理（一次组一个 batch，全部生成完才收下一批）里，短请求生成完了还得陪长请求空等，GPU 大量时间在算"已经不需要继续的序列"。**→ 答案是 Continuous Batching（2.1 节）。**

**瓶颈三：重复计算公共前缀。** 100 个请求都带同一份 2 万 token 的系统提示词，朴素方案要把这 2 万 token 的 prefill 算 100 遍。**→ 答案是 Prefix Caching（2.3 节）。**

---

## 2. vLLM 的核心设计

### 2.1 Continuous Batching（连续批处理）

静态批处理把"一批请求"当成一个整体进、一个整体出；连续批处理把粒度缩小到**每一步（step）**：

- 每一步开始时，调度器重新决定哪些序列参与本轮计算；
- 序列一旦生成出 EOS 或达到 `max_tokens`，立即退出，腾出的显存和算力立刻给新请求；
- 新请求不需要等"凑一批"，随时可以插队进下一次 step。

在 nano-vllm 中，这个调度循环清晰到可以用十几行说清楚（`nanovllm/engine/llm_engine.py:73-76`）：

```python
while not self.is_finished():
    output, num_tokens = self.step()
```

而 `step()` 内部（`nanovllm/engine/llm_engine.py:49-55`）就是标准的"调度 → 执行 → 记账"三段式：

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()        # ① 调度：选出本轮的序列
    token_ids = self.model_runner.call("run", seqs, is_prefill)  # ② 执行：GPU 上前向
    self.scheduler.postprocess(seqs, token_ids, is_prefill)      # ③ 记账：发 token、回收显存
    ...
```

序列在调度器里只有三个状态（`nanovllm/engine/sequence.py:8-11`）：`WAITING`（还没 prefill，在等待队列）→ `RUNNING`（prefill 完成，正在逐 token 生成）→ `FINISHED`。每个 step 里，完成 prefill 的序列从 `waiting` 队列毕业进 `running` 队列，生成完毕的序列从 `running` 移除并释放全部 KV 块。**这就是"连续"二字的具体含义。**

### 2.2 PagedAttention：把操作系统搬进 GPU 显存

vLLM 最著名的设计。核心思想一句话：**KV Cache 不按序列连续分配，而是切成固定大小的"页"（page/block），像操作系统管理内存一样管理**。

- 逻辑上，序列的第 i 个 token 落在第 `i / block_size` 个**逻辑块**；
- 物理上，每个逻辑块映射到一个**物理块**（GPU 大张量里的一段），映射关系记录在该序列的**块表（block table）**里；
- 序列增长时按需分配新块，不需要预留最大长度；显存不够时甚至可以抢占（见 4.4.4）；
- 块大小小（vLLM 默认 16，nano-vllm 为 256，见下），内部碎片最多浪费"块大小 × 序列数"的尾块空间。

一张图看懂（以 block_size=4 示意）：

```
序列 A (7 tokens): 逻辑块 [0][1][2]
                    块表 →  [ 物理块 5 ][ 物理块 2 ][ 物理块 9 ]   ← 逻辑连续
序列 B (5 tokens): 逻辑块 [0][1]                                  物理不连续
                    块表 →  [ 物理块 3 ][ 物理块 7 ]               ✓ 无需预留 4096
                                                                  ✓ 无外部碎片
物理块池（一整块大显存张量）:
  [0][1][2][3][4][5][6][7][8][9]...  ← 空闲块挂 free 队列，用完归还
```

**为什么 nano-vllm 的块是 256？** `Config` 断言 `kvcache_block_size % 256 == 0`（`nanovllm/config.py:22`）。因为 decode 阶段用的 `flash_attn_with_kvcache` 要求页大小为 256 的整数倍——这是 FlashAttention 库的实现约束，而非算法选择。物理存储上，整个池子是一个形状为 `[2, 层数, 块数, block_size, KV头数, 头维度]` 的大张量（`nanovllm/engine/model_runner.py:115`），K 和 V 各占一半维度，每层注意力模块拿到属于自己的切片（`nanovllm/engine/model_runner.py:117-121`）。

### 2.3 Prefix Caching：公共前缀只算一次

观察现实负载：聊天服务里所有请求都以同一份系统提示词开头；代码补全里前缀是同一个文件；多轮对话中历史轮次完全重复。**只要两个序列的前缀 token 完全相同，它们的 KV Cache 内容就完全相同**——那为什么不算一遍、存着反复用？

实现上有三个关键问题，nano-vllm 的答案都很有代表性：

**① 怎么判断"前缀相同"？——链式哈希。** 每个块的内容哈希不仅覆盖本块 token，还把**前一个块的哈希值**作为前缀掺进来（`nanovllm/engine/block_manager.py:35-41`）：

```python
def compute_hash(cls, token_ids, prefix=-1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))   # 链式：父块哈希参入
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

于是哈希命中就意味着"从第 0 块到本块的整个前缀链都相同"，天然防止了"中间某块不同但凑巧哈希相同"的误命中（代码还做了 token 级二次比对防哈希碰撞，`nanovllm/engine/block_manager.py:66`）。

**② 命中了怎么复用？——块级引用计数 + 空闲块复活。** 这是 nano-vllm 里最精彩的细节（`nanovllm/engine/block_manager.py:75-92`）：

- 新序列 prefill 时先问 `can_allocate()`（`block_manager.py:58`）：沿着哈希链逐块查，能命中几个已缓存的块；
- 命中的块**不再重新计算**，直接挂进新序列的块表。若块正被别的序列用着，就 `ref_count += 1` 共享（多个序列读同一块 KV 是安全的，因为它永远不会被原地修改）；若块已经空闲（原序列结束了，但数据还在显存里没被挤掉），直接"复活"它，数据原封不动；
- 只有未命中的尾部块才需要分配空闲块并计算。

**③ 缓存什么时候失效？——惰性淘汰。** 序列结束时块进入空闲队列，但**哈希登记并不删除**（`block_manager.py:53-56` 的 `_deallocate_block` 不清哈希）；只有当这个空闲块被重新分配给别的内容时，才在 `_allocate_block` 里抹掉旧哈希（`block_manager.py:43-51`）。也就是说：**前缀缓存没有过期时间，纯粹"显存不够了就挤掉最老的空闲块"**，零额外维护成本。

还有一处容易忽略的细节：**最后一个块永远不做哈希登记**。`can_allocate` 的循环只走到 `num_blocks - 1`（`block_manager.py:62`），`hash_blocks` 也只哈希"本轮新写满"的整块（`block_manager.py:110-120`）。因为序列的最后一个块通常是半满的，接下来还会往里追加 token，内容会变——把一个"将来会变"的块登记进缓存是错误的。

### 2.4 Chunked Prefill：把长 prompt 切着喂

想象一个 8000 token 的 prompt 和 100 个正在 decode 的序列同时存在。如果一次性 prefill 这 8000 token，GPU 要花几十毫秒专攻这一件事，期间所有 decode 序列都卡住——用户侧观感就是"这 100 个聊天全部卡了一下"。

Chunked prefill 的做法：给每个 step 设一个 **token 预算**（`max_num_batched_tokens`，nano-vllm 默认 16384，`nanovllm/config.py:9`），prompt 超过预算就切成多轮：

```
prompt 有 20000 token，预算 8192：
  step 1: 算 token [0,     8192)   ← 序列仍处于 WAITING
  step 2: 算 token [8192,  16384)  ← 仍在 WAITING，中间不产出 token
  step 3: 算 token [16384, 20000)  ← prefill 完成，产出第 1 个生成 token，进入 RUNNING
```

nano-vllm 的实现非常克制（`nanovllm/engine/scheduler.py:42-46`）：

```python
if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
    break
seq.num_scheduled_tokens = min(num_tokens, remaining)
```

只允许**队首第一条**序列被切分（`scheduled_seqs` 非空就不再切），避免一个 batch 里出现多个"半截"序列导致调度碎片化。被切的序列留在 `waiting` 队列里，用 `num_cached_tokens` 记录"已经算到哪了"（这个字段名有误导性，它不只是缓存命中数，而是"已在前几轮被计算过的 token 数"），下一轮从断点继续。postprocess 里同样只有 prefill **完整结束**的序列才 append 采样的 token（`scheduler.py:86-87`）——中间块的末位置 logits 对应的是"假设的下一个 token"，没有意义。

### 2.5 四个配套优化技术

**① FlashAttention（`nanovllm/layers/attention.py`）。** 注意力计算的标准实现要显式生成 `L×L` 的注意力矩阵，显存和带宽开销都是平方级。FlashAttention 用 tiling（分块计算）+ online softmax 技巧，从不物化整个注意力矩阵，同时把对 KV Cache 的读取融合进少数几个大 kernel。nano-vllm 用了它的两个接口：

- prefill → `flash_attn_varlen_func`（`attention.py:67`）：处理"一个 batch 里塞了多条不等长序列"的场景，靠 `cu_seqlens_q/cu_seqlens_k`（累计长度数组）划分边界，避免 padding 浪费（详见 4.5.2）；
- decode → `flash_attn_with_kvcache`（`attention.py:72`）：直接从分页 KV Cache 里按块表读取历史 K/V。

**② CUDA Graph（`nanovllm/engine/model_runner.py:222-257`）。** decode 阶段每个序列每步只算 1 个 token，batch 小的时候一轮前向要启动几百个小 kernel，**GPU 大部分时间在等 CPU 发指令**。CUDA Graph 把"一整轮前向的所有 kernel"录制成一张图，之后每步只需把新输入拷进固定缓冲区、`graph.replay()` 一次启动全部。只有 decode 用它（prefill 计算量大、长度可变，不值得）。细节见 4.5.5。

**③ Tensor Parallelism（`nanovllm/layers/linear.py`）。** 单卡放不下或想要更高吞吐时，把模型切开到多卡。切法不是随便切——线性层有两种经典切法（column/row），配合起来可以让一次前向只需 2 次通信。细节见 4.7.2。

**④ 向量化采样（`nanovllm/layers/sampler.py`）。** "按概率分布采样一个 token"如果用 Python 循环逐序列做会拖垮 decode。nano-vllm 用了一个数学技巧（Gumbel-max 变体）把整个 batch 的采样变成一次向量化 argmax，见 4.7.6。

---

## 3. nano-vllm 全景

### 3.1 它是什么、不是什么

**是**：一个可运行、可对标 vLLM 吞吐的**离线批量推理引擎**（API 是 `LLM.generate(prompts)`，跑完一批返回一批），教学优先，代码极简。

**不是**：在线服务（没有 HTTP/OpenAI API server）、不支持流式、只实现了 Qwen3 一个模型族、无量化/投机解码/pipeline 并行。和 vLLM 的完整对照见第 7 章。

### 3.2 目录地图

```
nano-vllm/
├── example.py                    # 5 分钟上手示例（chat 模板 + generate）
├── bench.py                      # 吞吐基准：256 条随机序列对比 vLLM
└── nanovllm/
    ├── llm.py                    # LLM 类，仅仅是 LLMEngine 的别名（对齐 vLLM API）
    ├── config.py                 # 全局配置（预算、并行度、显存水位线）
    ├── sampling_params.py        # 采样参数（temperature / max_tokens / ignore_eos）
    ├── engine/                   # ── 推理引擎核心 ──
    │   ├── llm_engine.py         #   引擎主循环：generate → step 循环
    │   ├── sequence.py           #   Sequence：一条请求的全部状态
    │   ├── scheduler.py          #   调度器：每个 step 算什么
    │   ├── block_manager.py      #   块管理器：分页 KV Cache 的分配与哈希缓存
    │   └── model_runner.py       #   模型执行器：张量打包、KV 池、CUDA Graph、TP
    ├── layers/                   # ── 网络层算子 ──
    │   ├── attention.py          #   Attention（FlashAttention + Triton 写缓存 kernel）
    │   ├── linear.py             #   线性层家族 + TP 分片 + 权重加载协议
    │   ├── embed_head.py         #   词表并行 Embedding / LM Head
    │   ├── layernorm.py          #   RMSNorm（含残差融合版）
    │   ├── rotary_embedding.py   #   RoPE 位置编码
    │   ├── activation.py         #   SiLU 门控激活
    │   └── sampler.py            #   向量化采样器
    ├── models/
    │   └── qwen3.py              # Qwen3 模型定义（组装上面的积木）
    └── utils/
        ├── context.py            # 全局调度上下文（把调度信息递给 attention）
        └── loader.py             # safetensors 权重加载器
```

### 3.3 对象关系图

```
LLMEngine (llm_engine.py)
 ├── tokenizer                    # HF tokenizer，进出两侧的编解码
 ├── scheduler : Scheduler
 │    ├── waiting: deque[Sequence] # 等 prefill 的序列
 │    ├── running: deque[Sequence] # prefill 完、正在 decode 的序列
 │    └── block_manager : BlockManager
 │         ├── blocks: list[Block] # 物理块数组（含 ref_count / hash / token_ids）
 │         ├── free_block_ids      # 空闲块队列
 │         └── hash_to_block_id    # 前缀缓存哈希表
 ├── model_runner : ModelRunner    # rank 0 在主进程；TP>1 时其余 rank 在子进程
 │    ├── model : Qwen3ForCausalLM # 各层模块持有 kv_cache 的切片
 │    ├── sampler : Sampler
 │    ├── graphs / graph_vars      # CUDA Graph 池与静态输入缓冲
 │    └── (TP>1) shm + events      # 与子进程 rank 的共享内存广播通道
 └── ps / events                   # TP 子进程列表及其唤醒事件
```

### 3.4 一次 `generate()` 的时间线

以 `example.py` 的两条 prompt 为例，从调用到返回的完整时间线：

```
llm.generate(prompts, sampling_params)
 │
 ├─ 逐条: tokenizer.encode → Sequence(prompt) → scheduler.waiting 入队
 │
 ├─ while not scheduler.is_finished():      ← 主循环，每圈 = 1 个 step
 │    │
 │    ├─ schedule()
 │    │    ├─ waiting 里有序列？→ 组 prefill 批（查前缀缓存 / 切块），返回 (seqs, True)
 │    │    └─ 否则            → 组 decode 批（必要时抢占尾部序列），返回 (seqs, False)
 │    │
 │    ├─ model_runner.call("run", seqs, is_prefill)
 │    │    ├─ prepare_prefill / prepare_decode   # CPU: 拼 input_ids、cu_seqlens、slot_mapping、块表
 │    │    ├─ run_model                          # GPU: 前向（或 CUDA Graph replay）
 │    │    └─ sampler                            # GPU: 采出每序列 1 个 token
 │    │
 │    └─ postprocess(seqs, token_ids)
 │         ├─ block_manager.hash_blocks()        # 新写满的块登记进前缀缓存
 │         ├─ chunked prefill 未完的序列: 不 append，留在 waiting
 │         ├─ 完整序列: append_token
 │         └─ 命中 EOS / 达到 max_tokens → FINISHED，释放全部块
 │
 └─ 全部完成: 按 seq_id 排序 → tokenizer.decode → 返回 [{"text":..., "token_ids":...}]
```

---

## 4. 源码走读

建议按本章顺序读，这个顺序是"由静到动"：先看数据结构（Sequence / BlockManager），再看决策逻辑（Scheduler），最后看执行层（ModelRunner / 模型）。

### 4.1 配置：`Config` 与 `SamplingParams`

`Config`（`nanovllm/config.py`）是全局预算表：

| 字段 | 默认值 | 含义 |
|---|---|---|
| `max_num_batched_tokens` | 16384 | **每个 step 的 token 预算**（一个 step 里所有序列的 prefill token 之和上限），也是 chunked prefill 的切块尺寸 |
| `max_num_seqs` | 512 | 并发序列数上限（decode 批大小上限） |
| `max_model_len` | 4096 | 单序列最大长度，取用户值与模型 `max_position_embeddings` 的较小者（`config.py:25`） |
| `gpu_memory_utilization` | 0.9 | KV Cache 池的显存水位线（占总显存比例） |
| `tensor_parallel_size` | 1 | 张量并行卡数（1~8） |
| `enforce_eager` | False | True 则禁用 CUDA Graph（调试用） |
| `kvcache_block_size` | 256 | KV 块大小，必须是 256 的倍数（FlashAttention 约束） |
| `num_kvcache_blocks` | -1 | 块数，**不是用户设的**——由 `ModelRunner` 探测剩余显存后回填（见 4.5.4） |

`SamplingParams`（`nanovllm/sampling_params.py`）故意做得极小：只有 `temperature`、`max_tokens`、`ignore_eos` 三个参数，且**禁止 greedy**（断言 `temperature > 1e-10`）——因为采样器用的是指数竞争技巧（4.7.6），T→0 会数值爆炸。对比 vLLM 的 `top_p`/`top_k`/`n`/`seed`/... 全家桶，这是一处刻意为之的简化。

### 4.2 `Sequence`：一条请求的一生

`nanovllm/engine/sequence.py`。这是全项目最核心的数据结构——调度器、块管理器、模型执行器都围绕它读写。字段一览：

| 字段 | 含义 |
|---|---|
| `seq_id` | 全局自增 id（类级计数器，`sequence.py:16`），决定输出顺序 |
| `status` | `WAITING` / `RUNNING` / `FINISHED` 三态 |
| `token_ids` | 完整序列（prompt + 已生成部分），随 decode 逐步 append |
| `num_prompt_tokens` | prompt 长度，划分 prompt 与 completion 的界线 |
| `num_tokens` | 当前总长（`len(seq)` 返回它） |
| `num_cached_tokens` | **已被计算过的 token 数**（含前缀缓存命中 + 历轮 chunked prefill 已算部分） |
| `num_scheduled_tokens` | 本轮 step 被调度的 token 数（prefill 可以 >1，decode 恒为 1） |
| `is_prefill` | 本轮是否处于 prefill 形态 |
| `block_table` | **块表**：`[物理块id, ...]`，逻辑块 i → `block_table[i]` |
| `last_token` | 最后一个 token（decode 时它就是本轮模型的输入） |

派生属性里有两个值得注意：`num_blocks = ceil(num_tokens / block_size)`（`sequence.py:56-57`）和 `last_block_num_tokens`（最后一块里现有的 token 数，`sequence.py:59-61`，decode 时用于计算写入位置，见 4.5.3）。

**最巧妙的一处：`__getstate__`/`__setstate__`（`sequence.py:72-83`）。** 这对方法定制了 pickle 行为，专门服务于张量并行：TP>1 时，rank 0 每一步都要把序列列表 pickle 后写共享内存广播给其他 rank（见 4.5.6）。decode 时其他 rank 其实只需要每个序列的 `last_token` 和块表，完全不需要几千个 token 的完整列表。于是：

```python
def __getstate__(self):
    last_state = self.last_token if not self.is_prefill else self.token_ids
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.num_scheduled_tokens, self.block_table, last_state)
```

decode 步只序列化**一个 int**而非整个列表；prefill 步（新序列刚进来）才传完整列表。一个不到十行的优化，把每步的进程间通信量从 O(序列长度) 降到 O(1)。

**序列状态机**（结合调度器看）：

```
            add_request()
                │
                ▼
            WAITING ──────────────┐
                │  prefill 完整算完 │  显存不足被抢占
                ▼                  │ (preempt: 释放全部块,
             RUNNING ──────────────┘  回到 waiting 队首, 重新 prefill)
                │
                │ EOS 或 max_tokens
                ▼
             FINISHED（释放全部块）
```

### 4.3 `BlockManager`：显存的"操作系统"

`nanovllm/engine/block_manager.py`，121 行，实现 2.2/2.3 节的全部概念。数据结构：

```python
self.blocks: list[Block]             # 物理块数组；Block = {block_id, ref_count, hash, token_ids}
self.hash_to_block_id: dict          # 前缀缓存：内容哈希 → 块 id
self.free_block_ids: deque           # 空闲块队列（FIFO，老的块先被重新使用）
self.used_block_ids: set             # 在用块集合
```

六个方法分为两组，对应序列生命周期的两端：

**进入时（prefill 前）**：
- `can_allocate(seq) → num_cached_blocks | -1`（`block_manager.py:58-73`）：双重职责。沿哈希链数能复用几个块；同时统计去掉共享块后还需要几个新块，**空闲块不够就返回 -1**（调度器看到 -1 就暂停派发 prefill，等 running 里的序列结束释放显存）。注意循环上界是 `num_blocks - 1`——半满的最后一块不参与缓存匹配；
- `allocate(seq, num_cached_blocks)`（`block_manager.py:75-92`）：命中块挂引用（在用的 `ref_count += 1` 共享；空闲的直接从 free 队列捞回，数据复活），未命中的尾部块从 free 队列分配新块，写入 `seq.block_table`。

**运行中（decode 每步）**：
- `can_append(seq)`（`block_manager.py:103-104`）：decode 每步先问"这个序列这步要不要新块"。答案浓缩在一行布尔表达式里：

  ```python
  return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
  ```

  为什么是 `% block_size == 1`？关键在时序：**decode 调度发生在上一步 token append 之后**。设序列现有 L 个 token（最后一个是刚 append 的、本轮要喂给模型的），本轮它的 K/V 要写入 slot `L-1`。当 `L % 256 == 1` 时，`L-1 = 256k` 恰好是一个**新块的第 0 格**，此时才需要新块；否则写进现有最后一块的空位，零分配。举例：L=257（prompt 恰好 256 + 刚生成 1 个）→ 本轮写 slot 256 → 需要第 2 个块；L=513 → 写 slot 512 → 需要第 3 个块。**平均每 256 步 decode 才有一次分配**，这就是分页的摊销效果；
- `may_append(seq)`（`block_manager.py:106-108`）：真正执行上述条件下的分配。

**退出/记账时**：
- `hash_blocks(seq)`（`block_manager.py:110-120`）：prefill 完成后调用。把本轮**新写满**的整块（区间 `[num_cached_tokens/bs, (num_cached_tokens+num_scheduled_tokens)/bs)`）沿哈希链登记进 `hash_to_block_id`，供后来的序列命中。链头从上一块的哈希续起，保证全局前缀语义；
- `deallocate(seq)`（`block_manager.py:94-101`）：序列结束（或被抢占）时，逐块 `ref_count -= 1`，归零的块回 free 队列。**不清哈希**——这就是前缀缓存能跨序列存活的原因（直到该块被重新分配才在 `_allocate_block` 里除名，`block_manager.py:47-48`）。

### 4.4 `Scheduler`：两态循环

`nanovllm/engine/scheduler.py`，93 行，是整个引擎的"心脏"。`schedule()` 的结构是先 prefill 后 decode 的两段式，**且互斥**：只要本轮凑出了 prefill 批就直接返回，不等 decode（`scheduler.py:54-55`）。

#### 4.4.1 Prefill 段（`scheduler.py:29-52`）

```python
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]
    remaining = self.max_num_batched_tokens - num_batched_tokens
    if remaining == 0: break
    if not seq.block_table:                       # 新序列：查前缀缓存
        num_cached_blocks = self.block_manager.can_allocate(seq)
        if num_cached_blocks == -1: break         # 显存不足，本轮不发 prefill
        num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
    else:                                         # chunked prefill 续传：从断点接着算
        num_tokens = seq.num_tokens - seq.num_cached_tokens
    if remaining < num_tokens and scheduled_seqs: break   # 只允许切队首
    if not seq.block_table:
        self.block_manager.allocate(seq, num_cached_blocks)
    seq.num_scheduled_tokens = min(num_tokens, remaining)
    ...
```

要点：
- **FIFO 公平性**：永远从 `waiting[0]`（队首）取，切也只能切队首这一条，不会饿死后面的序列；
- 新序列的"需要计算的 token 数"要**扣除前缀缓存命中的块**（`num_cached_blocks * block_size`），这就是 prefix caching 落到调度层的效果——命中越多，本轮 prefill 越便宜；
- 序列只有**完整 prefill 完**（`num_cached_tokens + num_scheduled_tokens == num_tokens`）才从 `waiting` 弹出、进入 `running`（`scheduler.py:48-51`）。被 chunk 切开的序列仍留在 `waiting` 队首，下轮优先续传；
- `can_allocate` 返回 -1 时直接 `break` 而不是抢占——nano-vllm 对 prefill 的显存不足采取"躺平等待"策略（等待 running 序列自然结束），比 vLLM 简单，代价是极端情况下新请求排队时间不可控。

#### 4.4.2 Decode 段（`scheduler.py:57-73`）

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        if self.running:
            self.preempt(self.running.pop())      # 抢占队尾（最新加入的）
        else:
            self.preempt(seq); break               # 队列只剩自己：抢自己
    else:
        seq.num_scheduled_tokens = 1
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)
assert scheduled_seqs
self.running.extendleft(reversed(scheduled_seqs))  # 保持原顺序放回
```

decode 阶段的每个序列每步只需要 1 个 token 的计算，唯一的资源约束是**新块分配**。当空闲块耗尽时触发**抢占（preemption）**：牺牲 `running` 队尾（最新加入、沉没成本最低）的序列，把它的 KV 块全部释放，回 `waiting` 队首（`scheduler.py:75-79`）：

```python
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)   # KV 全部丢弃
    self.waiting.appendleft(seq)         # 之后要重新 prefill（重算）
```

注意 nano-vllm 的抢占是**纯重计算（recomputation）策略**：被抢序列的 KV Cache 直接丢弃，恢复时从头再 prefill 一遍（好在有前缀缓存，已写满的块可能直接命中复活）。vLLM 的完整版还有 swap（换出到 CPU 内存）选项，这里省略了。还有个细节：抢占自己之后 `break` 跳出 `while-else`，该序列本轮不参与计算——而 `assert scheduled_seqs`（`scheduler.py:71`）表明系统保证至少能调度出一条序列（最坏情况：抢占掉所有其他序列后，自己必然分得到块）。

#### 4.4.3 Postprocess（`scheduler.py:81-92`）：每步的收尾

```python
def postprocess(self, seqs, token_ids, is_prefill):
    for seq, token_id in zip(seqs, token_ids):
        self.block_manager.hash_blocks(seq)              # ① 登记新写满的块（前缀缓存）
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            continue                                     # ② chunk 未完：不 append
        seq.append_token(token_id)                       # ③ append 新 token
        if (not seq.ignore_eos and token_id == self.eos) \
           or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED         # ④ 终止判定
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

注意 `zip(seqs, token_ids)`：模型对 batch 里每个序列各采样一个 token，按顺序一一对应。终止条件是 **EOS 或达到 max_tokens**（bench 里 `ignore_eos=True` 强制跑满，方便统计吞吐）。

### 4.5 `ModelRunner`：从序列列表到 GPU 张量

`nanovllm/engine/model_runner.py` 是引擎与模型之间的桥梁，也是"系统代码"和"深度学习代码"的交汇处。

#### 4.5.1 初始化（`model_runner.py:17-48`）

按顺序做了 7 件事，顺序本身就有讲究：

1. `dist.init_process_group("nccl", ...)`——先建 NCCL 通信组（即使 TP=1 也建，代码就不用分叉）；
2. `torch.set_default_dtype(bf16)` + `set_default_device("cuda")`——之后所有裸 `torch.empty(...)` 直接落在正确的设备和 dtype 上；
3. 构建 `Qwen3ForCausalLM`（此时权重是随机垃圾值）；
4. `load_model` 从 safetensors 加载真权重（4.8 节）；
5. **`warmup_model()`**（4.5.4）——用假数据跑一次最大规模 prefill，只为**测量推理时的峰值显存**；
6. **`allocate_kv_cache()`**（4.5.4）——根据剩余显存决定 KV 池大小；
7. 若非 eager，`capture_cudagraph()`（4.5.5）。

之后若 TP>1：rank 0 创建 1MB 共享内存（`SharedMemory(name="nanovllm")`），其余 rank 进入 `loop()` 待命（4.5.6）。

#### 4.5.2 `prepare_prefill`（`model_runner.py:129-170`）：varlen 打包

prefill 批是"多条不等长序列拼在一起"，如果用 padding 对齐到最长，短序列浪费大量算力。业界标准做法是 **packed/varlen 格式**，nano-vllm 用 FlashAttention 的约定：

```
batch 里有 2 条序列，长度 3 和 2（切过的从断点起算）：

input_ids    = [t0 t1 t2 | t3 t4]         ← 直接拼接，无 padding
positions    = [ 0  1  2 |  0  1]         ← 各序列内部从自己的断点偏移起算
cu_seqlens_q = [0, 3, 5]                  ← Q 的累计长度：第 1 条占 [0,3)，第 2 条占 [3,5)
cu_seqlens_k = [0, 3, 5]                  ← K 的累计长度；命中前缀缓存时会是 [0, 6, 9]
                                          │   （含缓存命中的历史 token，见下）
max_seqlen_q/k                            ← 供 kernel 启动配置用
```

`cu_seqlens`（cumulative sequence lengths）是 varlen 世界的"分隔符"：attention kernel 靠它知道哪些 token 属于同一条序列、只允许序列内部互相注意（配合 `causal=True` 再加"只看过去"）。

**slot_mapping——KV 写入地址表。** 本轮每个 token 算出的 K/V 要写进分页缓存的具体哪个格子？答案是 `物理块id × block_size + 块内偏移`。构造代码（`model_runner.py:151-161`）遍历每个序列覆盖到的块区间，把 `[slot_start, slot_end)` 逐格展开。这个一维数组随后交给 Triton kernel 完成散射写入（4.7.1）。

**前缀缓存命中时的特殊路径。** 若 `cu_seqlens_k[-1] > cu_seqlens_q[-1]`（K 总长大于 Q 总长，说明有缓存前缀），额外准备 `block_tables`（`model_runner.py:162-163`），并在 `set_context` 里带上。此时 attention 层会切换为"K/V 全部从缓存读"模式（4.7.1）——本轮新算的 K/V 也已经由 `store_kvcache` 写进缓存了，于是"历史（缓存）+ 本轮（刚写入）"在缓存里连成一体，FlashAttention 拿着块表直接读完整上下文。

最后 `set_context(...)` 把所有这些张量塞进全局上下文（4.6 节），供模型内部（主要是 attention 层）取用。

#### 4.5.3 `prepare_decode`（`model_runner.py:172-188`）：每序列一个 token

decode 的打包简单得多：每个序列贡献 1 个 `last_token`、位置 `len(seq)-1`、上下文长度 `len(seq)`。写入位置（slot）的算法浓缩成一行（`model_runner.py:181`）：

```python
slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
```

即"最后一块的起点 + 该块已占格数 - 1"。结合 4.3 节的时序分析：刚 append 的 token 是本轮输入，它的 K/V 写进"现有内容的下一格"，而 `last_block_num_tokens - 1` 正是这一格的块内偏移。

#### 4.5.4 KV Cache 池的容量测算（`model_runner.py:91-121`）

这是"自适应显存管理"的样板代码。目标：**在不爆显存的前提下，把剩余显存尽可能全给 KV Cache**。

```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ...  # 构造 4 条 4096-token 的假序列（全 0 token），跑一次 prefill
    self.run(seqs, True)
    torch.cuda.empty_cache()
```

warmup 时 KV 池还没分配（`k_cache` 还是空 tensor，attention 层判断 `k_cache.numel()` 为假就跳过写缓存，`attention.py:62`），所以测出的峰值纯粹是"权重 + 前向激活"。随后：

```python
free, total = torch.cuda.mem_get_info()                      # GPU 实际剩余/总量
peak    = torch.cuda.memory_stats()["allocated_bytes.all.peak"]   # warmup 峰值（含激活）
current = torch.cuda.memory_stats()["allocated_bytes.all.current"]# 当前实际占用（权重）
num_kv_heads = hf_config.num_key_value_heads // world_size   # TP 下每卡的头数
block_bytes = 2 * layers * block_size * num_kv_heads * head_dim * dtype.itemsize
config.num_kvcache_blocks = int(total * gpu_memory_utilization
                                - used - (peak - current)) // block_bytes
```

公式解读：`total × util` 是我们允许使用的总显存；`used` 是进程已占的基线（权重）；`peak - current` 是前向过程中**临时激活的峰值增量**，必须为它留出余量；剩下的全换算成块数。测出来后回填进 `config.num_kvcache_blocks`（所以 Config 里它默认 -1），再分配形状为 `[2, layers, num_blocks, block_size, kv_heads, head_dim]` 的大张量，并把每层 attention 模块的 `k_cache/v_cache` 指向自己的切片（`model_runner.py:117-121`）。

用 Qwen3-0.6B 在 8GB 卡上估算：`block_bytes = 2×28×256×8×128×2B ≈ 28MB`，`0.9×8GB − 1.2GB权重 − 激活余量 ≈ 5.5GB` → 约 **190 个块 ≈ 4.9 万 token 的 KV 容量**——同时服务 512 条 4096-token 序列当然不够，所以抢占机制是必需品。

#### 4.5.5 CUDA Graph（`model_runner.py:195-257`）

**为什么只有 decode 需要**：见 2.5②。**实现三件套**：

1. **静态缓冲区**：`input_ids/positions/slot_mapping/context_lens/block_tables/outputs` 全部按最大 batch（`min(max_num_seqs, 512)`）预分配（`model_runner.py:227-233`）。Graph 录制的是"这些固定地址上的计算"，所以回放前只需往固定地址拷新数据；
2. **按 batch size 分档录制**：不可能为每个 bs（1~512）都录一张。分档 `graph_bs = [1, 2, 4, 8] + [16, 32, ..., 512]`（`model_runner.py:234`），运行时选 **≥ 实际 bs 的最小档**（`model_runner.py:202`），多出来的槽位当作哑巴槽——`slot_mapping` 填 -1，Triton kernel 里 `if slot == -1: return` 直接跳过（`attention.py:23`），不产生副作用；
3. **从大到小录制 + 共享池**（`for bs in reversed(self.graph_bs)`，`model_runner.py:238`）：CUDA Graph 的经典技巧。所有 graph 共享一个 `graph_pool`，先录最大的，后面小的 graph 可以复用大 graph 已占用的内存池，避免 20 多张 graph 各自拷贝一份权重/激活导致显存翻倍。

运行时路径（`model_runner.py:196-212`）：

```python
if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
    return self.model.compute_logits(self.model(...))       # 直接跑
else:
    graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
    graph_vars["input_ids"][:bs] = input_ids                # 拷入静态缓冲
    ...
    graph.replay()                                          # 一次调用，整图启动
    return self.model.compute_logits(graph_vars["outputs"][:bs])
```

prefill（长度不定）、eager 模式、超大批次（启动开销占比已可忽略）都走正常路径。

#### 4.5.6 张量并行的进程模型（`model_runner.py:41-48, 61-89`）

TP>1 时的拓扑很特别：**rank 0 生活在主进程里**（就是 `ModelRunner` 对象本身），rank 1..n-1 是 `LLMEngine` spawn 出来的子进程（`llm_engine.py:24-30`），各自也实例化了一份 `ModelRunner` 并在 `loop()` 里死循环等任务。

通信分两层，各司其职：

- **控制面**（传"该干活了、干哪条序列"）：rank 0 把 `(方法名, 参数)` pickle 后写进共享内存，然后 `event.set()` 敲醒所有子进程（`model_runner.py:76-84`）；子进程 `event.wait()` 醒来读取、执行、清事件（`model_runner.py:68-74`）。这里 Sequence 的 pickle 瘦身（4.2 节）直接决定了每步的控制面开销；
- **数据面**（传中间张量）：NCCL。`RowParallelLinear` 里的 `all_reduce`、`VocabParallelEmbedding` 里的 `all_reduce`、`ParallelLMHead` 里的 `gather`（见 4.7），都走 GPU 上的 NCCL 集合通信，与控制面互不干扰。

`call()` 方法（`model_runner.py:85-89`）是这套机制的入口：rank 0 广播后自己也执行一遍。于是每一步前向，所有 rank 各自算自己那 1/n 的张量，通过 NCCL 对齐，只有 rank 0 负责采样并返回结果（`model_runner.py:218`）。

#### 4.5.7 `run()`：一步执行的完整清单（`model_runner.py:214-220`）

```python
def run(self, seqs, is_prefill):
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    reset_context()
    return token_ids
```

注意所有 CPU→GPU 的搬运都带 `pin_memory=True` + `.cuda(non_blocking=True)`（如 `model_runner.py:164-168`）——锁页内存 + 异步拷贝，让 H2D 传输与 CPU 后续工作重叠。

### 4.6 `Context`：一个"不优雅但很实用"的全局变量

`nanovllm/utils/context.py` 只有一个 dataclass 和三个函数：`set_context` / `get_context` / `reset_context`，维护一个模块级单例 `_CONTEXT`。

它解决的问题：attention 层每一步需要 `slot_mapping`、`cu_seqlens`、块表等**调度信息**，但这些信息只有 `ModelRunner` 知道，而模型 forward 的调用链（`Qwen3ForCausalLM → Qwen3Model → Qwen3DecoderLayer → Qwen3Attention → Attention`）有四五层，其中间层根本不关心这些。选择有二：把 context 一路透传（污染所有中间层签名，还与 HuggingFace 模型结构不兼容），或者塞全局变量（`layers/attention.py:60`、`layers/embed_head.py:57` 直接 `get_context()` 取用）。

nano-vllm 选了后者。工程上这是"必要之恶"——真 vLLM 同样有类似的 AttentionMetadata 机制（挂在 forward_context 上）。值得学习的点是它用 `reset_context()`（`model_runner.py:219`）保证生命周期干净：一步结束，全局状态归零，绝不跨 step 存活。

### 4.7 模型层：积木如何搭成 Qwen3

`nanovllm/models/qwen3.py` 是纯组装层，把 `layers/` 里的积木拼成模型。自底向上看：

#### 4.7.1 `Attention`（`layers/attention.py`）：一写一读

attention 的 forward 只有四步（`attention.py:59-75`）：

```python
def forward(self, q, k, v):
    context = get_context()
    if k_cache.numel() and v_cache.numel():
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)   # ① 写
    if context.is_prefill:
        if context.block_tables is not None:      # 前缀缓存命中
            k, v = k_cache, v_cache               #    K/V 全部改从缓存读
        o = flash_attn_varlen_func(q, k, v, ..., block_table=context.block_tables)  # ② 读
    else:
        o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                    cache_seqlens=context.context_lens,
                                    block_table=context.block_tables, ...)          # ③ 读
    return o
```

- **① 写**：`store_kvcache` 是一个手写 Triton kernel（`attention.py:10-40`）：grid 按 token 数启动，每个 program 处理一个 token 的全部 KV 头（`D = num_heads × head_dim`），从 `slot_mapping` 读出目标格子地址，把新算的 K/V 散射（scatter）进分页缓存。这就是 PagedAttention 的"写路径"；
- **② prefill 读**：普通情形直接用刚算出的 k/v（不用碰缓存，除了刚才顺手写入的那份）；**命中前缀缓存时**切换为从缓存读——因为"缓存里的历史前缀 + 刚写入的本轮 token"在缓存里物理上是连续上下文，FlashAttention 拿块表即可读到全部；
- **③ decode 读**：`flash_attn_with_kvcache` 原生支持分页缓存，传 `cache_seqlens`（每序列当前上下文长度）和块表即可。

`q/k/v` 从哪来？在 `Qwen3Attention.forward`（`models/qwen3.py:72-88`）里：`qkv_proj` 一次算出 QKV → 拆开 → QK-norm → RoPE → 送进 `Attention`。GQA（16 个 query 头共享 8 个 KV 头）在形状上自然体现：`num_kv_heads < num_heads`，缓存里只存 KV 头。

#### 4.7.2 线性层家族（`layers/linear.py`）：TP 切分的教科书

Transformer 里的线性层按"怎么切权重"分两类，nano-vllm 各有实现：

**ColumnParallelLinear（按输出维切，`linear.py:54-73`）**：权重 `W[out, in]` 沿输出维切成 n 段，每卡持有 `[out/n, in]`。各卡独立算 `x @ W_i^T` 得到输出的不同**列段**，互不通信。

**RowParallelLinear（按输入维切，`linear.py:131-156`）**：权重沿输入维切，每卡持有 `[out, in/n]`。各卡算 `x_i @ W_i^T` 得到**部分和**，最后 `dist.all_reduce` 加起来才是完整输出。

**为什么 QKV 用 column、o_proj 用 row？** 把两者配对放在一个 attention 块里，中间不需要任何通信：

```
            hidden (每卡完整)                        hidden (每卡完整)
                 │                                        ▲
        ┌────────┴────────┐                        all_reduce (唯一通信点)
        ▼        ▼        ▼                              ▲
     rank0     rank1    rank2      →  attention(本地头) → ┘
     QKV投影   QKV投影   QKV_proj      各卡只算自己那几个   o_proj: row 切分
   (column: 各卡持有部分Q头和KV头)      头的 attention      (各卡对本地头输出求部分和)
```

同理 MLP：`gate_up_proj` 用 column（`MergedColumnParallelLinear`，把 gate/up 两个矩阵拼成一个权重，`linear.py:76-93`），`down_proj` 用 row。**结论：一个 decoder 层每卡只需 2 次 all_reduce（attention 一次 + MLP 一次），与卡数无关**——这是 Megatron-LM 确立的经典 TP 布局，vLLM 原样继承。

**`QKVParallelLinear`（`linear.py:96-128`）的额外麻烦**：Q、K、V 三个投影合并成一个矩阵，但三者的切分粒度不同（Q 头 16 个、KV 头 8 个，各自除以卡数）。加载权重时要按 `loaded_shard_id`（"q"/"k"/"v"）把 HF 的三个独立权重分别塞进合并矩阵的正确区段，每段再取自己 rank 的 chunk。这就是 `weight_loader` 协议存在的意义（4.8 节）。

**权重加载协议**：`LinearBase` 给每个 `nn.Parameter` 挂一个 `weight_loader` 方法（`linear.py:26-27`）。每个并行变体 override 自己的版本，描述"这个参数应该从全局权重的哪个切片加载"。注意 `RowParallelLinear.forward` 里 `bias` 只在 rank 0 加（`linear.py:153`）——bias 加了再 all_reduce 就重复 n 次了。

#### 4.7.3 `VocabParallelEmbedding` / `ParallelLMHead`（`layers/embed_head.py`）

词表维度（Qwen3 是 151936）也要切：每卡持有 `vocab/tp` 行 embedding。查表时，落在别的卡的 token id 被 mask 掉，本卡的正常查，然后 `all_reduce` 求和（mask 位置贡献 0，`embed_head.py:34-42`）。

`ParallelLMHead` 继承它但 forward 有两处不同（`embed_head.py:56-66`）：

1. **只算末位置**：prefill 时先 `x = x[cu_seqlens_q[1:] - 1]`——`cu_seqlens_q[1:]` 是每条序列的结束偏移，减 1 即最后位置。一个 batch 的 prompt 合计几千 token，但 logits 只需要在 `批大小` 个位置上算，词表 15 万的情况下这省掉的是**几十亿次**乘法；
2. **TP 时 gather 而非 all_reduce**：每卡算出自己词表段的 logits，`gather` 到 rank 0 沿词表维拼接（`embed_head.py:62-65`）。

另外 `Qwen3ForCausalLM.__init__` 里 `tie_word_embeddings=True` 时直接让 `lm_head.weight` 共享 `embed_tokens.weight` 的 data（`models/qwen3.py:202-203`）——0.6B 小模型省一份 151936×1024×2B ≈ 300MB 的权重。

#### 4.7.4 `RMSNorm` / `RotaryEmbedding` / `SiluAndMul`

三个小算子，共同点是**处处用 `@torch.compile`**（`layernorm.py:16, 28`、`rotary_embedding.py:37`、`activation.py:8`）：

- `RMSNorm`：Qwen 系的归一化，没有均值中心化，只有均方根缩放。实现了融合残差版 `add_rms_forward`（相加、归一化一次 kernel 完成，返回更新后的 residual，`layernorm.py:28-40`）——decoder 层的两个 norm 各融合一次残差相加，省两次全量读写；
- `RotaryEmbedding`：预计算全部位置的 cos/sin 表存 buffer，forward 按 positions 查表后对 Q/K 做旋转。`get_rope` 用 `@lru_cache(1)`（`rotary_embedding.py:51-59`）保证所有层共享同一份表（28 层 × 4 万位置 × 256 维只存一份）；
- `SiluAndMul`：SwiLLM 门控 MLP 的激活，`silu(x) * y`，输入是 gate/up 拼接的 2 倍宽向量，`chunk(2)` 后逐元素乘。

`@torch.compile` 在这里是零成本抽象：首次调用触发编译（几秒），之后每个小算子融合成单 kernel，对小 batch decode 的收益显著。

#### 4.7.5 `Sampler`（`layers/sampler.py`）：指数竞争采样

完整实现只有 4 行（`sampler.py:8-12`）：

```python
@torch.compile
def forward(self, logits, temperatures):
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
    return sample_tokens
```

奥秘在最后一行：`argmax(probs / Exp(1))` 在分布上**等价于**从 `Categorical(probs)` 采样一个样本。这叫指数竞争（exponential race，Gumbel-max trick 的孪生版本）：若 `E_i ~ Exp(1)` 独立同分布，则 `argmax_i (p_i / E_i)` 落在 i 的概率恰为 `p_i / Σp`。证明梗概：`p_i / E_i` 的分布是 `Exp(p_i)`，n 个独立指数钟最先响的概率与速率成正比。

为什么要绕这个弯？对比朴素写法 `torch.multinomial(probs, 1)`：multinomial 内部要走 CUDA 随机数 + 逆 CDF，batch 小 kernel 多；而这里是纯逐元素除法 + argmax，**整条链路（softmax→除法→argmax）可被 torch.compile 融合，且与 temperature 天然组合**（除在 logits 上）。代价是不支持 greedy（T=0 除零），所以 `SamplingParams` 索性禁止。

#### 4.7.6 Qwen3 模型组装（`models/qwen3.py`）

一层 decoder 的 forward（`qwen3.py:146-159`）画出残差流：

```
hidden ──┬─────────────────────────────────────────┬──────────────► 下一层
         │                                         │
   input_layernorm(hidden, residual)         post_attention_layernorm
         │                                         │
   self_attn (QKV→QK-norm→RoPE→attn→o_proj)  mlp (gate_up→silu*→down)
         └──► + ─────────────────────────────┘
        (residual 一路贯穿，由 RMSNorm 的融合版维护)
```

Qwen3 相对 Qwen2 的结构差异点在 `Qwen3Attention.__init__`（`qwen3.py:68-70`）：**QK-norm**——对 Q 和 K 各做一次 head 维 RMSNorm（在 RoPE 之前），训练稳定性手段，判断条件是 `not qkv_bias`（有 attention_bias 的旧式模型不带 QK-norm）。

`Qwen3ForCausalLM.packed_modules_mapping`（`qwen3.py:187-193`）是一张"名字翻译表"：HF 权重里叫 `q_proj/k_proj/v_proj`，本实现合并叫 `qkv_proj`；HF 的 `gate_proj/up_proj` 合并成 `gate_up_proj`。加载器靠它找到合并后的参数和正确的 shard_id。

### 4.8 权重加载（`utils/loader.py`）

只有 28 行，却体现了"协议化"设计的全部好处（`loader.py:12-28`）：遍历目录下所有 safetensors 文件 → 对每个权重名查 `packed_modules_mapping` 看是否需要改名/拆分 → `model.get_parameter(name)` 拿到目标参数 → 调用参数自带的 `weight_loader(param, loaded_weight, shard_id)` 完成加载。

**妙处在于加载逻辑跟着参数走，而不是跟着模型走**：`ColumnParallelLinear` 的参数知道自己要取全局权重的哪一段；`QKVParallelLinear` 的参数知道 q/k/v 各该塞到哪个偏移。loader 本身对 TP、合并、tie 一无所知。新增一种并行策略 = 新写一个 weight_loader，加载器零改动。

---

## 5. 端到端串讲：两条 prompt 的完整旅程

把所有环节串起来。假设 `example.py` 的两条 prompt（设 tokenize 后分别为 15 和 22 个 token，`temperature=0.6, max_tokens=256`），TP=1、非 eager。

**T0 引擎启动**（`LLM(path, enforce_eager=False)`）：构造 Config（读 `Qwen3-0.6B/config.json`，`max_model_len` 压到 4096）→ 加载 tokenizer → 建模型、载权重（bf16 权重约 1.2GB 落 GPU）→ warmup 跑一次 4×4096 假序列测出峰值显存 → 算出约 190 个 KV 块（`Sequence.block_size` 设为 256，`num_kvcache_blocks` 回填进 config）→ Scheduler 拿到块数构建 BlockManager → 逐档录制 20 多张 CUDA Graph。

**T1 提交请求**：`generate()` 把两条 prompt 各自 `tokenizer.encode` → 两个 `Sequence` 进 `scheduler.waiting`。

**T2 Step 1（prefill）**：`schedule()` 看 waiting 非空 → 逐条检查。序列 0：新序列，`can_allocate` 沿哈希链找——池子刚建全是空块，命中 0，需要 1 个新块（15 token 一块装得下），空闲充足 → 分配块 `b0`，`num_scheduled_tokens=15`，完整算完 → 移入 running。序列 1 同理拿块 `b1`。合计 37 token ≤ 16384 预算。返回 `([seq0, seq1], True)`。
`ModelRunner.prepare_prefill`：拼出 `input_ids=[37个]`、`positions=[0..14, 0..21]`、`cu_seqlens_q=[0,15,37]`、slot_mapping 指向 b0/b1 的格子。前向：embedding → 28 层（每层 QKV→QK-norm→RoPE→**Triton 写 KV 进 b0/b1**→FlashAttention→o_proj→MLP）→ 末层 norm → LMHead 只取两个末位置的 hidden 算 logits `[2, 151936]` → Sampler 除以 0.6、指数竞争采出 2 个 token。
`postprocess`：`hash_blocks`（两块都没满 256，无事可登记）→ 两序列都完整 prefill 完 → 各 append 第一个生成 token → 都未到 EOS/256 → 继续。

**T3 Step 2..N（decode）**：waiting 已空 → decode 段。从 running 取 2 条，`can_append` 检查（长度 16/23，`% 256 ≠ 1`，不需要新块）→ `num_scheduled_tokens=1`。`prepare_decode`：`input_ids=[两序列的 last_token]`、slot_mapping 指向各自块内的下一格。`run_model` 走 CUDA Graph：bs=2 选中档位 2 的 graph，拷入静态缓冲，`replay()`。attention 走 `flash_attn_with_kvcache` 从块表读全部历史。每步各序列长 1 个 token。

**T4 途中**：序列长度爬过 256 的倍数时（如 seq0 到第 257 步），`can_append` 发现 `len % 256 == 1`，需要新块——空闲块充足就分配；如果这时池子耗尽（假设有很多并发序列），decode 段开始**抢占队尾**：最年轻的序列被 deallocate、扔回 waiting 队首（等下次轮到它时带着完整 token_ids 重新 prefill——已写满的旧块可能还在哈希表里，能直接命中复活，不必全重算）。

**T5 生成完成**：某序列采出 EOS（151645）→ postprocess 置 FINISHED → `deallocate` 归还所有块（哈希登记保留，供后续相同前缀的请求复用）→ 从 running 移除。最后一条也结束后 `is_finished()` 为真，主循环退出。

**T6 返回**：按 `seq_id` 排序（保证与输入 prompt 顺序一致，`llm_engine.py:88`）→ `tokenizer.decode` 还原文本 → 返回 `[{"text": ..., "token_ids": ...}, ...]`。进程退出时 `atexit` 触发 `engine.exit()` → rank 0 广播 `"exit"` → 共享内存清理 → NCCL 收摊（`llm_engine.py:37-41`，`model_runner.py:50-59`）。

---

## 6. 性能优化技术对照表

| 技术 | 解决什么问题 | 代码位置 |
|---|---|---|
| Continuous Batching | 静态批处理的空转浪费 | `scheduler.py`（waiting/running 双队列 + 每步重组批） |
| PagedAttention | KV Cache 内/外碎片 | `block_manager.py`（块表、free 队列）+ `attention.py`（Triton scatter 写 + paged 读） |
| Prefix Caching | 公共前缀重复计算 | `block_manager.py:35-41`（链式哈希）、`:43-51`（惰性淘汰）、`:75-92`（引用计数共享/复活） |
| Chunked Prefill | 长 prefill 饿死 decode | `scheduler.py:42-46`（预算切块）+ `config.py:9`（16384 预算） |
| Preemption（重计算式） | 显存耗尽时活序列保命 | `scheduler.py:57-79`（抢占队尾、回 waiting 重 prefill） |
| varlen 打包（无 padding） | 短序列 padding 浪费 | `model_runner.py:129-170`（cu_seqlens）+ `flash_attn_varlen_func` |
| 只算末位置 logits | prefill 中间 logits 无用 | `embed_head.py:58-60`（`cu_seqlens_q[1:] - 1`） |
| CUDA Graph | 小 batch decode 的 kernel 启动开销 | `model_runner.py:222-257`（分档录制、共享池、-1 哑槽） |
| torch.compile | 小算子融合 | `layernorm.py` / `activation.py` / `rotary_embedding.py` / `sampler.py` 各处的装饰器 |
| 残差融合 RMSNorm | 残差相加的额外读写 | `layernorm.py:28-40`（`add_rms_forward`） |
| 指数竞争采样 | 采样的 Python/多 kernel 开销 | `sampler.py:8-12` |
| Tensor Parallelism | 单卡放不下 / 更高吞吐 | `linear.py`（column/row 配对）、`embed_head.py`、`model_runner.py:61-89`（控制面） |
| Sequence pickle 瘦身 | TP 控制面每步通信量 | `sequence.py:72-83` |
| pin_memory + async H2D | CPU→GPU 拷贝阻塞 | `model_runner.py:164-168` 等 |
| 按剩余显存自适应 KV 池 | 不同卡/模型手动调参 | `model_runner.py:91-121`（warmup 测峰值 → 反推块数） |
| 权重合并（QKV/gate_up） | 多次小 GEMM 启动开销 | `linear.py:76-128`、`qwen3.py:187-193`、`loader.py` |

## 7. nano-vllm vs 真 vLLM：为了教学砍掉了什么

理解"砍掉了什么"和理解"保留了什么"同样有价值——前者就是从 12 万行到 1200 行的距离：

| 维度 | nano-vllm | vLLM |
|---|---|---|
| 模型支持 | 仅 Qwen3 | 百余种架构（还有多模态） |
| 服务形态 | 离线批量 `LLM.generate` | 离线 + OpenAI 兼容 server（异步、流式、取消） |
| 调度策略 | prefill 优先，prefill/decode 互斥 | 默认 decode 优先（保首 token 延迟），prefill/decode 可混批 |
| KV 块大小 | 256（FlashAttention 约束） | 默认 16，可配 |
| 显存不足 | 等待（prefill）/ 纯重计算抢占（decode） | 重计算或 CPU swap 两种抢占模式 |
| 采样 | temperature only | top_p/top_k/min_p/penalties/beam/n>1/seed... |
| 并行 | TP ≤ 8 单机 | TP/PP/EP/PP+TP、多机 Ray |
| 量化 | 无 | GPTQ/AWQ/FP8/INT8/Marlin... |
| 其他 | — | 投机解码、structured output、prefix cache 细粒度化、prefill chunk 可调、CUDA Graph 分数批... |

有两处差异值得特别留意，因为它们是**行为差异**而不只是功能多少：

1. **调度优先级**：nano-vllm 只要 waiting 非空且资源够就全力 prefill（`scheduler.py:54-55` 的提前 return），decode 会被持续到来的 prefill 推迟；vLLM 默认策略反过来（decode 优先 + chunked prefill 混批），因为在线服务的交互延迟主要看 decode 迭代间隔。做吞吐基准时两者差别不大（bench.py 正是吞吐场景），但换到在线延迟敏感场景，nano-vllm 的策略会显著拉高尾延迟。
2. **`can_allocate` 失败即躺平**：新 prefill 因显存不足失败时，nano-vllm 干等 running 序列自然结束；vLLM 会主动 preempt/swap 腾地方。极端负载下 nano-vllm 的排队延迟不可控。

## 8. 建议的学习路径与练习

**路径（按周计划也可以按天，取决于投入）**：

1. **跑起来**：`example.py` 用本地 `Qwen3-0.6B/` 跑通，改 `enforce_eager=True/False`、`temperature`、`max_tokens` 感受差异；
2. **读静态结构**：`sequence.py` → `block_manager.py`（逐行）→ `config.py`；
3. **读动态循环**：在 `llm_engine.py:49` 的 `step()` 里打条件断点，观察每步 `schedule()` 返回的 `(seqs, is_prefill)` 和每条序列的 `num_cached_tokens / num_scheduled_tokens / block_table` 变化；
4. **读执行层**：`model_runner.py` 的 prepare 系列（拿纸笔画出每个张量的形状）→ `attention.py` → `linear.py`；
5. **对照真 vLLM**：从 nano 学到的概念去 vLLM 仓库搜同名物（`BlockManager`、`Scheduler`、`Block`、`cu_seqlens`），看工业版多做了什么。

**练习（由易到难）**：

1. *加一个采样参数*：实现 `top_k`（提示：在 `sampler.py` softmax 之后做 masked top-k，再归一化）；
2. *支持 greedy*：允许 `temperature=0`（提示：`argmax(logits)` 单独分支）；
3. *打印调度轨迹*：给 `schedule()` 加日志，复现本文 5 节的旅程，验证你对 chunked prefill 的理解；
4. *做一个"前缀缓存命中率"指标*：在 `can_allocate` 里统计命中块数/总块数，用 100 条相同 system prompt 的请求验证收益；
5. *实现 beam search*（大工程）：需要一序列拆多序列共享前缀块——恰好是 `ref_count` 已支持的能力；
6. *换一个模型*：照着 `qwen3.py` 写 `llama.py`，体会 `layers/` 积木的复用性（LLaMA 缺 QK-norm、bias 处理不同）。

## 9. 术语表

| 术语 | 含义 |
|---|---|
| **Prefill / Decode** | 处理 prompt 的一次性前向 / 逐 token 生成的循环前向 |
| **KV Cache** | 已算过的 Key/Value 向量的显存缓存，避免重算历史 |
| **PagedAttention** | 把 KV Cache 切成固定块分页管理的注意力方案（vLLM 核心专利技术） |
| **Block / Page** | KV Cache 的分配单位；nano-vllm 中 256 token 一块 |
| **Block Table（块表）** | 序列的"逻辑块 → 物理块"映射表 |
| **Continuous Batching** | 每 step 重组批次，序列随完随走、随到随进的调度方式 |
| **Prefix Caching** | 相同前缀的 KV 块复用（本文：链式哈希 + 引用计数 + 空闲块复活） |
| **Chunked Prefill** | 把长 prompt 按每 step 预算切成多轮计算 |
| **Preemption（抢占）** | 显存不足时强制回收低优先级序列的资源（重计算式：丢 KV 重新 prefill） |
| **GQA** | Grouped-Query Attention：多个 Q 头共享一组 KV 头（Qwen3-0.6B 为 16Q:8KV） |
| **RoPE** | Rotary Position Embedding：以旋转编码注入位置信息 |
| **QK-norm** | 对 Q/K 做 RMSNorm（Qwen3 引入的稳定化手段） |
| **varlen / cu_seqlens** | 变长序列打包格式；`cu_seqlens` 为累计长度数组，作序列边界标记 |
| **slot_mapping** | "本轮各 token 的 K/V 写入缓存的格子地址"表 |
| **Tensor Parallelism（TP）** | 把层内权重切开多卡各算一部分的并行方式（column/row 切分配对，层内仅 2 次通信） |
| **CUDA Graph** | 预录制整轮 kernel 依赖图、一次 replay 全部启动的机制 |
| **FlashAttention** | 分块计算注意力、不物化注意力矩阵的 IO 感知算法 |
| **Exponential Race / Gumbel-max** | 用 `argmax(p/E)` 等价实现按概率采样的向量化技巧 |
| **tie_word_embeddings** | 输入 embedding 与输出 lm_head 共享权重（小模型常见） |
| **enforce_eager** | 关闭 CUDA Graph，逐 kernel 即时执行（调试用） |

---

*本文基于 nano-vllm v0.2.0（commit `bb823b3`）源码撰写。文档中的行号引用以该版本为准，后续演进可能偏移。*
