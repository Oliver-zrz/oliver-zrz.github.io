---
layout: "../../layouts/PostLayout.astro"
title: "Dynamo 断图后，谁还在引用优化器的 Tensor？"
description: "从一个 CPU 最小复现出发，分析优化器销毁后参数仍被恢复图持有的原因，以及跨 graph break 传递归属、按图注册清理和处理共享参数的修复设计。"
date: "2026-11-25"
tags: ["PyTorch", "Dynamo", "graph break", "内存管理", "优化器"]
category: "编译与生命周期"
---

一次参数更新已经执行完，优化器也销毁了，参数为什么还活着？

排查这个问题时，关键线索是一个对照：执行同样的原地更新，不经过 Dynamo 的 graph break，参数能够释放；经过 graph break 再恢复捕获，生成的图却可能继续持有它。计算结果正确，引用的生命周期出了问题。

这项修复提交在 [PyTorch PR #205014](https://github.com/pytorch/pytorch/pull/205014)，对应 [Issue #204339](https://github.com/pytorch/pytorch/issues/204339)。本文按提交 `5c46acf` 解释实现；截至 2026 年 11 月 25 日，PR 仍为 Draft，尚未合入。下文的验证结果是本地回归记录，不代表上游 CI 已通过。

## 从一个只用 CPU 的复现开始

Issue 中，维护者给出了一个更小的复现。它使用真实的 SGD `step()`，无需 NPU、CUDA，也不调用 Inductor：

```python
import gc
import weakref

import torch

p = torch.nn.Parameter(torch.ones(2))
p.grad = torch.ones(2)
opt = torch.optim.SGD([p])


@torch.compile(backend="eager")
def step():
    params = opt.param_groups[0]["params"]
    opt.step()
    with torch.no_grad():
        params[0].add_(1)


step()
ref = weakref.ref(p)
del p, opt
gc.collect()
print("leaked" if ref() is not None else "released")
```

代码来自 [Issue 中的最小复现](https://github.com/pytorch/pytorch/issues/204339#issuecomment-6829422902)。在本次基线环境中输出 `leaked`，应用补丁后输出 `released`。

这里的 `backend="eager"` 仍会让 Dynamo 捕获 Python 执行过程，只是捕获后的图交给 eager 后端执行。因此，这个复现能把问题收窄到 Dynamo 的引用管理，不必先讨论 kernel 生成。

`p` 是要更新的参数，`p.grad` 是这次更新使用的梯度。`opt` 的参数组引用同一个 `p`，局部变量 `params` 又拿到了参数组中的列表。`opt.step()` 更新参数，随后代码通过这个列表继续使用参数。基线中的执行会经历恢复捕获；本文后面的回归还用显式 `torch._dynamo.graph_break()` 控制中断次数。

最后的 `weakref.ref(p)` 只是一个观察窗口，不会阻止 `p` 释放。删掉 `p` 和 `opt` 后，如果编译器没有留下多余的强引用，`ref()` 就应该返回 `None`。`gc.collect()` 不能回收仍被可达对象引用的参数。

这个判断有前提：用户侧没有其他仍需使用参数的对象。如果模型、另一个优化器或用户列表还保存着参数，它继续存活是正常的。

## 为什么优化器参数会被放进图里

Dynamo 不只捕获模型的 `forward()`。只要编译入口里包含 Tensor 运算，优化器更新也可能进入它的处理范围。

通常，一次训练迭代会先计算损失，再用 `backward()` 得到梯度，最后调用 `optimizer.step()` 更新参数。下一次迭代仍使用同一组参数对象，只是其中的数值已经变化。Adam 还会保存每个参数的历史状态，例如梯度的一阶、二阶统计量，并在后续 `step()` 中继续更新这些状态。

参数和状态反复使用，为编译器提供了优化机会。Dynamo 的优化器专门处理路径会标记一些 Tensor 为静态地址，并可能把它们保存为生成图对象的属性。这里的“静态”约束地址与引用方式，**不表示参数数值固定不变**。

`GraphModule` 是承载图和相关属性的 Python 对象。它的 `_parameters`、`_buffers` 可以保存 Tensor。某些后端还会把属性提取成参数列表，供编译后的运行函数使用。这样，用户代码和编译器内部都可能引用同一个参数。

普通的动态中间 Tensor 常作为图输入、图输出或运行时局部值传递，不一定被长期保存在图对象中。优化器静态 Tensor 则需要一项明确约定：**优化器生命周期结束时，编译器应解除为这项优化额外保留的引用。**

## 断图之后，旧清理机制漏了什么

Graph break 可以理解为：Dynamo 捕获到一处不能继续放进当前图的执行，先结束这一段，执行必要的 Python 逻辑，再从后续位置恢复捕获。后面可能产生新的 `GraphModule`。前一段只有初始化操作时，也可能没有生成非空的计算图。

旧实现的清理围绕 `OptimizerVariable._init_group()` 展开。这个函数负责优化器初始化的专门处理。以 Adam 为例，它会准备参数、梯度和状态列表。

旧代码大致做了三件事：

1. `wrap_tensor()` 把需要清理的图属性名放进 `static_tensor_names`。
2. `_init_group()` 在当前 `OutputGraph` 上排入一个图完成后的注册动作。
3. 图生成后，注册 `weakref.finalize(optimizer, callback)`；优化器销毁时，回调删除图中的相应属性。

这套机制在同一段捕获内有用。但 graph break 之后，恢复代码会创建新的捕获上下文，同一个参数还可能通过 `params[0]` 等别名再次进入新图。新图未必重新执行 `_init_group()`，也没有对应的清理注册。即使把旧属性名复制过去，新图使用的名字也可能不同。

结果就是：优化器已经销毁，恢复图的 `_parameters` 里仍有一条指向参数的强引用。Issue 中的一个例子是恢复图保留了 `params___0`。

![旧机制将清理注册留在前段；修复后通过 resume 元数据传递优化器与 Tensor 的归属，并让新图按自己的实际引用注册清理](/media/dynamo-optimizer-cleanup/resume-ownership.svg)

[查看断图与清理注册大图](/media/dynamo-optimizer-cleanup/resume-ownership.svg)

修复需要传递的是“这个 Tensor 与哪些优化器的生命周期相关”，而不是上一张图的一份属性名清单。

## 把清理从初始化位置移到实际持有引用的图

补丁保留了静态地址优化，调整的是归属记录和清理注册的方式。

### 用弱引用记录归属

`OutputGraph` 新增 `optimizer_tensors`，概念上是下面这样的表：

```text
optimizer A → {parameter p, Adam state a, Adam state b}
optimizer B → {parameter q, optimizer state c}
```

外层和内层都使用 `WeakIdKeyDictionary`：按对象身份区分键，而且记录本身不把键强行留住。身份语义也能覆盖自定义优化器不可哈希、或两个不同优化器通过 `__eq__` 被判断为相等的情况。

归属在两处补齐：读取 `param_groups`、标记参数静态地址时，先记录参数；`_init_group()` 后续包装参数和状态时，再记录相应 Tensor。前者覆盖初始化之前就取出参数列表、后面再通过别名访问的情况。补丁保留了梯度来源的专门分支，没有把所有遇到的 Tensor 都算作静态归属对象。

### 让归属跨过 graph break

`InstructionTranslatorBase.create_resume()` 会把当前归属表合并进恢复代码的 `CodeContext`。恢复捕获创建新的 `OutputGraph` 时，再从这里读出归属。

这里没有复制 Tensor 数据，也没有复制上一张图的清理回调。传递的是弱引用元数据。恢复代码可能被缓存复用，所以使用合并，而不是直接覆盖已有记录。

### 每张图按自己的引用注册

新函数 `register_optimizer_finalizers(gm)` 在后端处理后执行。它查看当前图的 `_parameters`、`_buffers`，以及编译上下文中的 `params_flat` 列表，再与归属表匹配。

这一步才确定“当前图实际保存了什么、使用什么属性名”。所以，即使参数在恢复图中换了名字，仍能找到应清理的位置。后端可能把图属性移进参数列表，这也是不能只看图属性的原因。

注册放在 `try/finally` 中，覆盖该段后端编译失败的路径：后端即使抛出异常，也可能已经保存了图或参数引用。

| 文件 | 这次改动的职责 |
|---|---|
| `variables/optimizer.py` | 在参数和状态进入专门处理路径时记录归属，移除旧的属性名集合与注册方式 |
| `symbolic_convert.py` | 创建恢复代码时传递归属元数据 |
| `code_context.py` | 声明恢复代码上下文中的新元数据字段 |
| `output_graph.py` | 接收归属，按实际图属性和参数列表注册清理，建立生命周期 guard |
| `test/dynamo/test_optimizers.py` | 验证断图、别名、共享参数、不同后端等场景 |

实现可对照固定版本的 [optimizer.py](https://github.com/pytorch/pytorch/blob/5c46acfadbf4eb284df55788dd2501314c594167/torch/_dynamo/variables/optimizer.py)、[恢复捕获逻辑](https://github.com/pytorch/pytorch/blob/5c46acfadbf4eb284df55788dd2501314c594167/torch/_dynamo/symbolic_convert.py) 和 [OutputGraph](https://github.com/pytorch/pytorch/blob/5c46acfadbf4eb284df55788dd2501314c594167/torch/_dynamo/output_graph.py)。

## 为什么还要逐项清理参数列表

假设一张图同时涉及优化器 A 和 B。编译后的运行函数保存了参数列表：

```python
params_flat = [p_a, p_b]
```

现在 A 被销毁，B 仍活着。只删除图中的 `param_a` 属性还不够，因为原列表仍引用 `p_a`；直接 `params_flat.clear()` 又会把 B 的参数一起移走。

补丁做的是：

```python
params_flat[0] = None
# 原列表现在是 [None, p_b]
```

原列表可能被编译后的运行函数引用，因此必须修改原列表的对应位置。只改一份副本，原引用还在。保留槽位也能让其他参数的位置不变；删除中间元素会使后面的索引移动。

![优化器 A 销毁后，移除图对 p_a 的引用，并将原参数列表的对应槽位置为 None；优化器 B 的参数及索引保留](/media/dynamo-optimizer-cleanup/reference-cleanup.svg)

[查看引用清理大图](/media/dynamo-optimizer-cleanup/reference-cleanup.svg)

这不是要让一个已经失效的 A 图继续拿 `None` 计算。补丁还复用 Dynamo 的弱引用存活 guard：关联生命周期结束后，旧捕获不能继续被当作有效结果复用。

Tensor 子类会多一层对应关系。例如两个外层 Tensor 可能拆成四个内部 Tensor，编译上下文用索引表记录它们分别属于哪个外层槽位。清理时既匹配直接引用，也根据这个映射清理对应的内部槽位，避免只清掉外层、内部 Tensor 仍被保存。

## 一个参数被两个优化器共享，什么时候清理

如果 A 和 B 都使用参数 `p`，A 销毁时就删除 `p` 的所有编译引用，会干扰仍活着的 B。

补丁把“具有相同优化器集合的 Tensor”分成一组。只有一个 owner 时，直接以该优化器作为生命周期对象；有多个 owner 时，创建 `_OptimizerTensorLifetime` 对象，作为这组 Tensor 的共同存活标记。

每个 owner 的 finalizer 都保留一份对标记的引用。A 销毁时释放 A 的那一份；B 仍活着，标记就仍存在。最后一个 owner 销毁后，标记才消失，触发这组引用的清理。这个标记不反向强引用优化器。

![共享参数等待最后一个已跟踪 owner 销毁；各优化器独有的状态分别按自己的生命周期清理](/media/dynamo-optimizer-cleanup/shared-lifetime.svg)

[查看共享参数生命周期大图](/media/dynamo-optimizer-cleanup/shared-lifetime.svg)

这里的归属范围是捕获过程中已跟踪的优化器。图中的 guard 检查生命周期是否仍有效；它不表示参数数值没变，也不免除其他正常的复用检查。

## 清理回调自己也不能留住旧图

补丁审查中还需要解决一个引用方向问题：`weakref.finalize` 会保存回调。如果回调的闭包强引用 `gm`，只要优化器一直活着，已经退出使用的旧图也可能跟着一直存活。形状变化导致多次重编译时，这一点尤其值得检查。

最终实现通过 `weakref.ref(gm)` 访问图。回调执行时，图还活着就删除相应属性；图已经回收就跳过这部分。

对编译上下文也作了区分：回调保留清理所需的原参数列表和索引映射，而不保留整个 `TracingContext`。这些列表可能仍被运行函数持有，需要后续定向解除引用；上下文对象本身则应能独立回收。

对应回归保持优化器存活，触发两次形状相关编译，再释放编译函数并清理 Dynamo 缓存，检查旧 `GraphModule` 和 `TracingContext` 能否被回收。它验证的是“修复清理问题时，没有把旧图的寿命反过来延长”。

## 验证到了哪一步

最终修订基于 `1dcfb27`，PR 提交为 `5c46acf`。验证既做过 CPU nightly 上的 Python 源码覆盖，也完成了该 PR 提交的原生 CPU 源码构建。后者环境为 aarch64 Linux、Python 3.12.13、GCC 11.4.0，构建版本 `2.16.0a0+git5c46acf`。

| 检查 | 基线 | 补丁版本 |
|---|---|---|
| 同一组 50 个优化器测试实例 | 17 PASS、32 FAIL、1 SKIP | 49 PASS、0 FAIL、1 SKIP |
| 恢复函数测试 | — | 1 PASS |
| 静态地址相关测试 | — | 2 PASS |
| 额外 Tensor 子类检查 | — | 2 PASS |
| 额外并发销毁检查 | — | 6 PASS |
| 维护者的最小复现 | `leaked` | `released` |

优化器测试覆盖了 0/1/2 次显式断图、真实 Adam 更新、参数别名、共享参数与独立状态、resume 缓存复用次数，以及后端编译失败。`aot_eager` 和 CPU `inductor` 路径都有实际执行；CPU Inductor 生成并运行了 C++ kernel。

额外检查中，Tensor 子类使用 `TwoTensor` 验证两个外层参数到四个内部 Tensor 的映射和定向清理；并发检查覆盖共享 owner 被不同线程销毁，以及另一优化器销毁时独立的编译执行继续运行。

1 个 SKIP 是已有的 PrivateUse1 加速器专属测试。没有把它计作通过，也没有将 CPU 结果扩展成加速器验收。上述并发场景也不能证明任意线程交错都已经覆盖。具体测试与验证范围记录在 [PR 的 Validation](https://github.com/pytorch/pytorch/pull/205014) 中。

## 这个补丁的边界

Graph break 会影响很多执行状态，本次修复专门处理优化器静态 Tensor 的归属与清理传播。它没有把所有前段回调直接复制给恢复图。

原因是这些回调的职责并不相同。例如，图构建完成后的某些回调负责优化当前图里的通信节点，并不是对象销毁时的清理函数；上下文状态回滚、代码缓存失效也各有机制。它们是否需要跨图传播，必须按各自语义判断。

这次问题能在 CPU 上独立复现，根因落在 Dynamo 对恢复捕获的引用管理。修复要守住的约定是：**图可以缓存可复用的计算，但它额外保存的优化器 Tensor 引用，需要随相关生命周期结束而解除；仍被其他活跃对象使用的 Tensor 则应保留。**

## 源码与讨论

- [Issue #204339：问题描述与最小复现](https://github.com/pytorch/pytorch/issues/204339)
- [PR #205014：实现、测试与验证记录](https://github.com/pytorch/pytorch/pull/205014)
- [固定修订的完整差异](https://github.com/pytorch/pytorch/commit/5c46acfadbf4eb284df55788dd2501314c594167)
- [优化器回归测试源码](https://github.com/pytorch/pytorch/blob/5c46acfadbf4eb284df55788dd2501314c594167/test/dynamo/test_optimizers.py)
