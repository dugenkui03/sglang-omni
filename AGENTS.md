# Coding style guidelines

Before writing, modifying, or reviewing code, read and follow
[.claude/skills/code-review/coding-style.md](.claude/skills/code-review/coding-style.md).

## 外部应用目录约定

- 工作台等通过接口使用 sglang-omni 的外部应用统一放到 `projects/<项目名>/`。
- 各应用独立管理依赖、启动入口、配置和运行结果，不混入模型推理代码。
- 随本仓库同步到服务器后，按对应应用自己的说明部署。

# 用户长期交流偏好

- 对于代码仓库分析回答、知识讲解，都要别太扩展，简洁为主，除非用户让拓展或者系统性的说明。
- 新增模块、代码不宜过度设计，即使需要加一些设计你也要先同用户确认。
- 翻译英文内容的时候，术语用 中文术语(英文术语) 这种形式，写文章的时候也是，对于专业术语也是给出中英文。
- 写文档一般不超过120行，如果超过则需要向用户确认，如果有图则算一行。
- 对于用户的问题，不要长篇大论回应，用户一般希望能在十几秒到30秒内理解你讲的内容，所以控制字数并列点展示；对于不确定是否应该详细解释的内容，可以先简单解释然后向用户询问是否要进一步解释。
- 在类似 sglang 这种通用框架下中文回答用户问题的时候，应该尽可能给出你讲的中文名称或者缩写对应的英文术语，类似 向量(Embedding)、TP(Tensor Parallelism)。
- 描述数据的存放、读写和搬运时，明确区分 GPU 显存(GPU memory) 与 CPU 内存(CPU memory)，不要笼统使用“内存”。

## SGLang 张量(Tensor)计算讲解

在 SGLang 相关问答中，涉及 Tensor 计算时，请遵循以下规则：
1. 使用简单的数值 Tensor 举例，优先采用两个 token、小维度向量，变量名尽量对应代码。
2. 展示输入、权重（如有）、shape 及各维度含义、核心计算表达式和输出结果，不展开逐元素乘加过程。
3. 涉及并行时，说明每张 GPU 处理哪些输入、保存权重的哪部分、如何计算局部结果，以及最终采用求和还是拼接。
4. 简要说明示例的简化条件，保持说明简洁。

示例：o_proj 将各个 head 的注意力输出混合，投影回 hidden_size。假设两个 token、两个 head、每头 2 维，hidden_size=2，不使用 bias。

```python
import torch

# 每行一个 token；前两列属于 head 0，后两列属于 head 1
attn_output = torch.tensor([
    [1., 2., 3., 4.],
    [5., 6., 7., 8.],
])  # [num_tokens=2, num_heads × head_dim=4]

W = torch.tensor([
    [1., 0., 2., 0.],
    [0., 1., 0., 2.],
])  # [输出维度=2, 输入维度=4]

output = attn_output @ W.T  # [2,4] @ [4,2] → [2,2]
# 输出：[[7,10], [19,22]]
```

使用 TP=2 时，权重沿输入维度拆分，每张 GPU 负责一个 head：

```python
# GPU 0：处理 head 0，保存 W 的前两列
A  = attn_output[:, :2]  # [[1,2], [5,6]]
W0 = W[:, :2]           # [[1,0], [0,1]]
Y0 = A @ W0.T           # [[1,2], [5,6]]

# GPU 1：处理 head 1，保存 W 的后两列
B  = attn_output[:, 2:]  # [[3,4], [7,8]]
W1 = W[:, 2:]           # [[2,0], [0,2]]
Y1 = B @ W1.T           # [[6,8], [14,16]]

# 上述各卡输入、权重和局部结果的 shape 均为 [2,2]
# 跨卡 all_reduce 求和，得到完整输出
output = Y0 + Y1        # [[7,10], [19,22]]，shape [2,2]
```

# sglang项目长期交流偏好

- 回答知识和代码问题时，多用直观、简洁的图表：用纯文本树状图展示对象组成、层级关系和方法调用链，用 Markdown 表格展示编号映射、分类及差异对比。
- 文本树状图放在 `text` 代码块中，用 `│`、`├─`、`└─` 保持层级，用 `→` 表示调用顺序；需要时用 `①②③` 标记关键阶段、用 `←` 标记关键位置。
- 图表中的节点保留对象或方法名及简短职责，配少量解释和具体小例子，避免扩展成冗长的说明。仅在图表有助于理解时使用，不必给简单事实强行配图。

## 方法调用栈格式

- 方法调用栈统一使用 `text` 代码块中的树状文本：主链用 `→`，分支用 `│`、`├─`、`└─`，保持缩进与竖线连续。
- 关键阶段用 `①②③` 编号，同一分支内部纵向展开调用，保留实际类名和方法名。
- 方法后只附简短职责，关键位置用 `←` 标记；按需补充核心变量含义或 Tensor 形状。
- 聚焦核心调用。需要追到本项目的 C++/CUDA 时再深入一两层，不展开大段解释和示例。

### 方法调用栈示例

以下仅作格式参考，实际回答中的调用关系应以当前代码为准。

```text
FlashAttentionBackend.forward_decode()  ← 当前方法
    │
    ├─ ① 写入本轮 K/V cache
    │    → HybridLinearKVPool.set_kv_buffer()：解析槽位、映射层编号
    │    → MHATokenToKVPool.set_kv_buffer()：按需缩放、转换 dtype
    │    → MHATokenToKVPool._store_kv_layer()：选中当前层的缓存
    │    → _set_kv_buffer_impl()：进入 CUDA JIT 写入分支
    │    → store_cache()
    │        │
    │        ├─ _jit_kvcache_module()：取得编译后的模块
    │        └─ module.store_cache()：Python → C++ 绑定
    │             → StoreKVCacheKernel<...>::run()：C++ 入口，kvcache.cuh
    │             → LaunchKernel(...).enable_pdl(...)(...)：启动 GPU 内核
    │             → store_kvcache()：将 K/V 写入指定缓存槽位
    │
    ├─ ② 准备注意力输入
    │    → get_paged_mha_kv_cache()：取得当前层的 K/V 缓存视图
    │    → page_table：每个请求的 K/V 存在哪些缓存页
    │    → cache_seqlens：每个请求的有效 token 数（包含本轮）
    │    → q_reshaped：整理为 [token数, 本卡Q头数, head_dim]
    │
    ├─ ③ flash_attn_with_kvcache()：核心注意力计算
    │    → 按页表读取 K/V
    │    → Q @ K.T → 缩放与掩码 → softmax → V 的加权和
    │
    └─ ④ o.view(...)：展平各头的输出并返回
         → [token数, 本卡Q头数 × v_head_dim]
```

# 日志书写约定

- 日志调用代码（包括聊天示例和仓库代码）最多两行：第一行写文本格式字符串，第二行集中列出变量并结束调用，不逐个参数换行；变量或描述文本较少时，整条调用写成一行。
- 日志正文避免频繁换行，普通调试日志最多两行：第一行说明操作或输出格式，第二行集中列出变量名及其值；模块路径、实际类名等来源信息也放在第二行，不逐字段换行。
