---
layout: "../../layouts/PostLayout.astro"
title: "昇腾 PyTorch 编译缓存适配：Dynamo 预编译包与静态恢复"
description: "围绕 torch_npu 的 triton_experimental（Triton-ex）后端，梳理 FX/AOT 缓存身份、生成代码配置、Triton 工件收集和静态 autotuner 最佳配置恢复，并给出独立进程验证的证据与边界。"
date: "2026-11-25"
tags: ["昇腾", "PyTorch", "torch_npu", "Triton", "编译缓存"]
category: "编译与缓存"
---

一次 `torch.compile` 调用结束，kernel 已经生成，结果也正确。把编译产物保存下来，换一个进程加载，就能跳过编译吗？

这次在昇腾 PyTorch 上做缓存适配时，最容易误判的就是这一步：图缓存命中，只说明某一层找到了可复用结果。消费端是否还会生成候选、准备 launcher、重新调优，要看恢复路径保留了什么信息。

本文讨论的是 PTA（昇腾的 PyTorch 适配层，对应 `torch_npu`）中的 **`triton_experimental` 后端，后文简称 Triton-ex**。它接入 PyTorch Inductor，使用 Triton-Ascend 编译并运行 NPU kernel。三者职责不同：Inductor 负责图到代码的编译流程，Triton-ex 提供 NPU 后端适配，Triton-Ascend 负责更下层的 kernel 编译与运行支持。

工作起点是让 Dynamo 预编译 package 支持这个后端。沿着恢复流程继续检查，修改范围自然延伸到了 FX/AOT 缓存身份、生成代码中的配置传递、kernel 工件收集，以及静态 autotuner 的最佳配置恢复。这些修改中，有些也影响普通 `torch.compile` 的缓存命中路径。

## 先分清：命中了哪一层缓存

这里涉及四类主要缓存，另有保存调优选择结果的 autotune 缓存。它们负责的对象不同。

| 层次 | 保存或复用的主要内容 | 命中后仍需关注什么 |
|---|---|---|
| Dynamo package | Dynamo 生成的代码、guard 及其后端产物关联 | 环境是否兼容、guard 是否满足、后端依赖能否恢复 |
| AOTAutograd 缓存 | AOT 编译条目，包括前后向编译结果或其引用及相关元数据 | 对应的前后向结果及其依赖是否可用 |
| FXGraph 缓存 | Inductor 编译结果，例如 wrapper 和相关元数据 | kernel 工件、运行资源及调优状态能否恢复 |
| Triton kernel 缓存 | kernel 编译产物；NPU 二进制通常包含 `.npubin` | 是否找到匹配产物、设备是否能够加载并执行它 |
| Autotune 缓存 | 某次候选搜索选出的最佳配置，本地常见载体是 `.best_config` | 搜索身份是否一致、选中配置是否有可用二进制 |

这些是理解本文所需的主要层次，不是 PyTorch 所有缓存的完整清单。

普通编译中，较早阶段命中缓存，可以省去后续的一部分编译工作；Dynamo package 直接恢复已有后端产物时，则可能根本不再进入 AOT/FX key 的生成和查找流程。恢复过程中仍然要做适用性检查、依赖加载和运行资源准备，不能把“跳过编译”理解为“跳过所有检查”。

存储形式也要分开看：编译过程中，各层缓存有自己的位置；导出时，`torch.compiler.save_cache_artifacts()` 可以收集支持导出的缓存工件，再由 `load_cache_artifacts()` 恢复。**这不意味着任何一个单独的 Dynamo 磁盘条目都天然包含所有下层缓存文件。**

![普通编译与 package 恢复的主要路径；直接恢复可跳过 AOT 和 FX 的重新查找，但仍要恢复 kernel 运行依赖](/media/triton-ex-cache-reuse/cache-paths.svg)

[查看编译与恢复路径大图](/media/triton-ex-cache-reuse/cache-paths.svg)

上图省略了图中断、延迟反向等分支，只表达一个边界：**下层 key 的正确性和最外层 package 的适用性检查，各自承担责任。**

## 缓存身份必须反映实际参与编译的状态

先看配置。Triton-ex 有自己的配置模块，下文记作 `ncfg`。其中有些设置影响图优化，有些影响代码生成，还有些影响候选配置的构造。

如果这些配置没有进入 FX/AOT key，就可能出现“配置改了，key 没变”的情况。适配采用的是上游提供的后端配置注册入口 `device_custom_config`，让 Triton-ex 配置参与原生的缓存身份生成，而不是另写一套平行的 FX 缓存。

但注册配置只解决了一半问题。实际执行判断也必须读取相同的有效值。下面是一个简化例子：

```python
# 模块导入时，只复制当时的布尔值。
linearize_enabled = ncfg.codegen_linearize

def generate():
    if linearize_enabled:
        ...
```

如果配置模块后来被修改，这个局部别名不会自动跟着变化。它不是指向配置项的动态引用。于是可能发生：key 记录的是新值，生成代码却仍使用导入时的旧值。

因此，相关判断改为在需要时读取 `ncfg.codegen_linearize`；确实需要固定在生成时的值，再明确保存到生成产物的元数据中。关键是区分“现在应该读取的配置”和“这份产物生成时已经确定的配置”。

另一个身份来源是自定义 pass。pass 是对 FX 图进行变换的处理步骤。上游需要知道这些步骤是否改变，才能判断旧编译结果是否可复用。按 `CustomGraphPass` 协议提供稳定的 `uuid()`，就是在声明它的缓存身份。

Triton-ex 中的 `TeSourcePinPass` 本身不修改图，它把后端源码摘要、`torch_npu` 版本和 Triton-Ascend 发行版本纳入身份。同一实现应产生稳定身份；相关源码或版本变化后，应区分旧结果。这里记录的是编译实现身份，后端配置值仍由原生配置注册机制提供。

这项修改服务于下层 FX/AOT 缓存。它不能被解释成“任何 package 加载都会重新核对这些 key”。

## 为什么生成的代码还要携带配置

一个常见疑问是：kernel 不是已经编译好了吗，消费端为什么还会读配置？

因为保存结果的形态不只有一份可直接调用的设备函数。有些路径恢复 wrapper 和 Triton 源码模块，导入模块时还要构造 autotuner；另一些路径直接反序列化静态 autotuner，恢复已经编译成功的候选结果。

这里有三个容易混淆的对象：

- **wrapper**：图级调用代码，组织输入输出并调用各个 kernel。
- **autotuner**：管理候选配置、编译结果及最佳候选选择的对象。
- **launcher**：把参数、stream 等交给运行时、实际启动某个 kernel 的可调用对象。

有 wrapper 源码，不等于当前进程已经创建好 launcher；有候选配置列表，也不等于列表里的每个候选都已有可用二进制。

生成源码中使用的 `pointwise(...)`、`reduction(...)` 等入口，还会按 shape 提示和配置构造候选，并创建 autotuner。它们与 Inductor 调度阶段的融合优化不是同一件事，名字相近也不代表执行时点相同。

所以，影响后续候选生成和编译准备的生产端配置，要随生成代码的 `inductor_meta["npu_backend_config"]` 一起保存。普通 kernel 和模板 kernel 都要覆盖；模板 kernel 是按特定算法模板生成的 kernel，例如 FlexAttention 的相关实现，并非只通过普通逐点代码生成入口产生。

可以用一个示意场景理解：

```text
生产端设置 → 构造候选 [A, B, C, D] → 选中 B
消费端设置 → 如果重新构造为 [A, B, C, ..., M]
```

第二次构造出的搜索空间已经不同，原来的最佳配置记录不一定还能按相同身份命中。读生产端保存的配置，是为了让这份生成代码在恢复时保持原来的含义。

这不等于忽略消费端的请求。package 入口是否接受本次请求，是另一层判断；`inductor_meta` 负责的是**被接受的产物如何按生成时的语义恢复**。配置进入 key、配置随产物传递、入口检查请求，三件事不能互相替代。

## put 和 put_winner 不是两次保存同一个东西

Triton 工件进入包，需要参与当前收集过程。编译结束、磁盘上已经出现 `.npubin`，并不自动意味着 package 已经登记了它。

这里的两个接口分别提供不同信息：

| 接口 | 提供的信息 |
|---|---|
| `TritonBundler.put(...)` | 哪个 kernel 缓存目录/工件需要参与收集 |
| `TritonBundler.put_winner(...)` | 哪个已登记的编译配置最终被选用 |

按本文所对应实现的收集规则，winner 集合为空时收集登记的工件；集合非空时，会按 winner 过滤。考虑两个 kernel：

```text
kernel A：只有 A1，不需要多候选试跑
kernel B：有 B1、B2，试跑后选择 B2
```

如果只在多候选调优时登记 winner，就会得到：

```text
entries = {A1, B1, B2}
winners = {B2}
```

A1 虽然是必需工件，也已经登记，却会在 winner 过滤中被排除。因此单候选 kernel 的最终选择同样需要登记。

![混合单候选与多候选 kernel 时，漏记 A1 的 winner 会导致必要工件被过滤；完整记录应保留 A1 和 B2](/media/triton-ex-cache-reuse/bundle-winners.svg)

[查看工件过滤大图](/media/triton-ex-cache-reuse/bundle-winners.svg)

这也解释了为什么要检查多个调用位置：本进程刚编译完成、主进程接收 worker 的编译结果、已经运行过的 launcher 再参与一轮新的收集，都可能需要登记。worker 通常是编译子进程，它的内存记录不会自动成为主进程的收集状态。

适配的目标是让每条实际路径把所需工件交给当前 bundle。登记不等于重新编译，也不等于复制一份新的 kernel 源码。

**winner 集合还不等于最佳配置缓存。** 前者决定本轮收集保留哪些工件；后者使消费端知道“在这个候选搜索身份下，之前选的是哪一个”，从而避免再次试跑。

## 静态恢复最容易丢的是原始候选身份

静态 autotuner 恢复尤其容易暴露这个问题。

假设生产端一开始有三个候选：

```text
原始候选：[A, B, C]
编译结果：A 成功，B 成功，C 失败
调优结果：B 胜出
```

保存最佳配置时，搜索身份来自原始候选集合 `[A, B, C]`。但静态快照里的成功编译结果只有 `[A, B]`。C 编译失败，自然没有可恢复的成功产物。

如果恢复时从成功产物反推搜索身份，就会用 `hash([A, B])` 查找先前以 `hash([A, B, C])` 保存的记录。最佳配置文件可能就在磁盘上，但查找身份已经不一致。

需要保留的对象应明确区分：

| 对象 | 在这个例子中的内容 | 用途 |
|---|---|---|
| 原始候选身份 | `[A, B, C]` | 判断是否是同一次候选搜索 |
| 成功编译结果 | `[A, B]` | 表示当前快照可以恢复哪些二进制 |
| 最佳配置 | `B` | 表示调优选中了谁 |

这里的 A/B/C 是说明机制的示例，不是某次真实硬件失败的测量记录。

![原始候选、成功产物和最佳配置分别保存；恢复用原始候选身份读取 B，再从成功产物中匹配 B](/media/triton-ex-cache-reuse/static-restore.svg)

[查看静态恢复大图](/media/triton-ex-cache-reuse/static-restore.svg)

适配在 `NPUCachingAutotuner` 中覆盖了 `recheck_autotune_cache()`。上游的静态恢复生命周期调用它，在准备 launcher 前完成选择。主要处理顺序是：

1. 读取快照保存的原始候选身份，不用成功子集重新定义身份。
2. 遵守缓存禁用设置，并确定消费端当前的缓存读写位置。
3. 使用 NPU 一致的候选哈希读取最佳配置。
4. 在成功编译结果中找到参数一致的产物，将列表收窄到它。
5. 最佳配置缺失、旧快照缺少身份或 winner 产物不存在时，保留相应的回退行为，不宣称已经命中。

候选身份还必须包含 NPU 的有效编译参数。只有 tile 大小相同并不够：两个候选可能具有相同的 `kwargs`、`num_warps`、`num_stages`，却在 `extra_options` 中选择了不同的编译行为。恢复时忽略它们，可能匹配到另一份二进制。

`NPUConfig` 因而负责在转换、复制和序列化时保留这些参数；`_npu_config_key()` 负责生成一致的候选身份。**这里的 Config 是某个 kernel 的候选配置，不是前面整个后端的 `ncfg` 配置模块。**

路径同样不能混入身份。生产端绝对目录在消费端不一定存在；能够相对定位的源码路径，要按消费端缓存根重新绑定。手写 kernel 不在标准缓存树内时，则通过源码恢复入口取得当前文件位置，不能随意猜测。

## 怎么证明真的复用了

只看“输出正确”不够：消费端完全重编译一遍，也可能输出正确。只看“FX cache hit”也不够：图复用了，候选仍可能再次调优。

本次验收使用 2026 年 11 月 12 日的固定版本：`torch_npu` 提交 `1646cb508365862d641ddc25ba29e6ef10aa1002`，PyTorch `2.15.0+cpu`，Python 3.11.17。PyTorch 的 `+cpu` 是基包标识，NPU 能力由匹配源码完整构建的 `torch_npu` 原生扩展提供。

| NPU | CANN | Triton-Ascend | 上游 package 用例 | 本地补充用例 |
|---|---|---|---|---|
| Ascend910B3 | 9.2.0-beta.2 | 3.2.2 | 24 项符合预期 | 52 项通过 |
| Ascend910B3 | 9.2.0-beta.2 | 3.6.0.post2 | 24 项符合预期 | 52 项通过 |
| Ascend950DT | 9.2.0 | 3.2.2 | 24 项符合预期 | 52 项通过 |
| Ascend950DT | 9.2.0 | 3.6.0.post2 | 24 项符合预期 | 52 项通过 |

上游选择了 19 个方法，参数化展开为每组 24 个实例。每组有 23 项普通通过，另 1 项 `test_nn_module` 符合上游既有 `expectedFailure` 约定，实际异常为 `MODULE_MATCH guard cannot be serialized`。没有 skip；这不表示该 guard 的序列化限制已经修复，也不表示跑完了整个上游测试文件。

本地 52 项分别覆盖 package 相关契约、静态最佳配置恢复、pass 身份以及代码生成回归。其中部分候选失败、旧快照和工件缺失等边界由夹具和 mock 验证，不把它们当作真实 UB 超限模型验收。

另做了真正的独立进程测试，工作负载为 FP32、shape `[65536]` 的 `sin(x) + cos(x) * 2`，开启严格静态 launcher 模式，并关闭原始二进制字节保留。

- **普通 FX 路径**：关闭 package 和 AOTAutograd 缓存，生产和消费进程使用同一缓存根，单独观察 FX 恢复。
- **package 路径**：通过 `save_cache_artifacts()` 导出，移开生产端缓存目录，再在新进程、空消费目录中用 `load_cache_artifacts()` 恢复。

四组环境、两条路径均得到：

```text
FX 路径：fxgraph_cache_hit = 1
package 路径：dynamo_cache_hit = 1
最佳配置状态：autotune_cache_state = "hit"
消费端 _precompile_config 调用次数 = 0
消费端 autotune_to_one_config 调用次数 = 0
输出与 eager 对照一致
```

Ascend910B3 的成功结果列表从 4 个收窄到 1 个，Ascend950DT 从 5 个收窄到 1 个。这些数量是恢复检查前后的成功编译结果数，不能拿来当原始搜索空间大小。

零调用只表示所测恢复过程中没有再次进入这两个被监测入口。二进制加载、设备句柄和 launcher 准备仍属于正常运行工作。这组结果也不是性能对比：没有据此宣称冷启动缩短多少、winner 达到全局最优，或所有模型都能零调优恢复。

## 这次适配的边界

这次工作把三个问题接在了一起：缓存身份对应实际编译状态，导出包含执行所需工件，恢复使用相同候选身份找到可用结果。Dynamo package 是其中一个入口，普通 FX 静态恢复同样受益。

还需要保持两条界限。

第一，下层配置 key 和源码身份不能代替最外层 package 的环境、请求适用性检查。芯片、CANN、后端请求和编译器版本等公共检查，需要在相应入口落实。后续单独评估的 SystemInfo / Triton-Ascend 完整版本补丁，不算作这里已经验收的后端适配内容。

第二，本次证明最佳配置能够随完整缓存工件导出入口恢复，不能扩大为“任何独立 DiskDynamoStore 条目都自动包含 `.best_config`”。读写入口不同，就要分别验证携带了哪些工件。

同样，这些历史测试不代表跨芯片、跨 Triton-Ascend 版本可以共用一个包，也没有完成全量模型、所有 dtype 或全量 CI 的验收。它们支持的结论是：**在列出的四组环境和所测路径中，图与工件能够恢复，最佳配置正确命中，消费端没有重复候选编译或调优。**

## 代码与测试入口

- [Ascend/pytorch !47583](https://gitcode.com/Ascend/pytorch/pull/47583)：本文所述后端适配。
- [Triton-ex autotuner 与候选身份实现](https://gitcode.com/Ascend/pytorch/blob/1646cb508365862d641ddc25ba29e6ef10aa1002/torch_npu/_inductor/triton_experimental/npu_triton_heuristics.py)：`NPUConfig`、`recheck_autotune_cache()` 及工件登记。
- [Triton-ex pass 身份实现](https://gitcode.com/Ascend/pytorch/blob/1646cb508365862d641ddc25ba29e6ef10aa1002/torch_npu/_inductor/triton_experimental/fx_passes.py)：`TeSourcePinPass` 及源码摘要。
- [PyTorch package 上游用例](https://github.com/pytorch/pytorch/blob/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/test/dynamo/test_package.py)：所选上游验收的版本来源。

本文以该固定提交及其历史验证为准；后续 PyTorch、Triton-Ascend 的接口和默认行为可能变化。

