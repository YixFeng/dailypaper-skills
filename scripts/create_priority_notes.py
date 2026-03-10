#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


VAULT = Path("/Users/yixfeng/Library/Mobile Documents/iCloud~md~obsidian/Documents/daily-paper")
NOTES = VAULT / "PaperNotes" / "_inbox"
ASSETS = NOTES / "assets"
CONCEPTS = VAULT / "PaperNotes" / "_concepts"
DAILY = VAULT / "DailyPapers" / "2026-03-10-论文推荐.md"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_first_image(pdf_path: str, stem: str) -> str:
    ensure_dir(ASSETS)
    tmpdir = Path("/tmp") / f"{stem}_imgs"
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["/opt/homebrew/bin/pdfimages", "-png", pdf_path, str(tmpdir / "img")], check=True)
    candidates = sorted(tmpdir.glob("*.png"))
    chosen = None
    for candidate in candidates:
        if candidate.stat().st_size > 10_000:
            chosen = candidate
            break
    if chosen is None and candidates:
        chosen = candidates[0]
    if chosen is None:
        return ""
    out = ASSETS / f"{stem}_fig1.png"
    shutil.copyfile(chosen, out)
    return f"![[assets/{out.name}]]"


def write(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def concept_note(title: str, category: str, body: str) -> None:
    path = CONCEPTS / category / f"{title}.md"
    if path.exists():
        return
    write(path, body.strip() + "\n")


def main() -> None:
    meta_img = extract_first_image("/tmp/MetaWorld-X.pdf", "MetaWorld-X")
    inter_img = extract_first_image("/tmp/InterReal.pdf", "InterReal")
    atom_img = extract_first_image("/tmp/AtomVLA.pdf", "AtomVLA")

    write(
        NOTES / "MetaWorld-X.md",
        f"""---
title: "MetaWorld-X: Hierarchical World Modeling via VLM-Orchestrated Experts for Humanoid Loco-Manipulation"
method_name: "MetaWorld-X"
authors: [Yutong Shen, Hangxu Liu, Penghui Liu, Jiashuo Luo, Yongkang Zhang, Rex Morvley, Chen Jiang, Jianwei Zhang, Lei Zhang]
year: 2026
venue: arXiv
tags: [humanoid, loco-manipulation, world-model, mixture-of-experts, robotics]
zotero_collection: _inbox
image_source: local
arxiv_html: https://arxiv.org/html/2603.08572v1
created: 2026-03-10
---

# 论文笔记：MetaWorld-X

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Beijing University of Technology, Fudan University, University of Alberta, University of Hamburg |
| 日期 | March 2026 |
| 项目主页 | https://syt2004.github.io/metaworldX/ |
| 对比基线 | [[TD-MPC2]], [[DreamerV3]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.08572) / [HTML](https://arxiv.org/html/2603.08572v1) |

## 一句话总结

> MetaWorld-X 用 [[Mixture-of-Experts]] 把 humanoid loco-manipulation 拆成技能专家，再用 VLM 监督的路由器做语义组合，目标是让 whole-body control 不再被单一策略互相拖后腿。

## 核心贡献

1. **SEP 技能池**: 用 [[Specialized Expert Policy]] 把站立、走路、跑步、坐下、搬运、伸手、爬行等基础技能拆开学，减掉单一策略的梯度互打。
2. **IRM 语义路由**: 用 [[Intelligent Routing Mechanism]] 接收高层任务语义，把专家策略按权重混合，而不是硬切换。
3. **few-shot 组合泛化**: 路由器先吃 VLM 监督，再转到示范分布做精修，目的是把“语义组合”迁移到实际控制频率下。

## 问题背景

### 要解决的问题
高自由度 humanoid 在同时做 locomotion 和 manipulation 时，最容易死在两个地方：单策略参数共享导致 skill interference，和长时程规划里的 world model bias。

### 现有方法的局限
- [[TD-MPC2]] 和 [[DreamerV3]] 能做 latent planning，但 humanoid 高 DoF 下长 rollout 偏差会被迅速放大。
- 单一 policy 同时负责 balance、gait、arm coordination，很容易出现 motion jitter、姿态崩坏、技能切换发抖。
- 纯 task return 优化通常不关心生物力学自然性，最后学出来的是“能过任务但动作很丑”的东西。

### 本文的动机
作者要把 humanoid control 从“一个大网络硬吃所有技能”改成“专家池 + 语义路由”的层级控制，同时把 human motion prior 作为形状约束塞回训练目标。

## 方法详解

### 模型架构

MetaWorld-X 采用 **层级式 humanoid control** 架构：
- **输入**: 当前机器人状态 $s_t$、任务语义提示、专家示范统计
- **Backbone**: 技能专家池 + 路由网络 + latent world model reward/value heads
- **核心模块**: [[Specialized Expert Policy]]、[[Intelligent Routing Mechanism]]、[[Vision-Language Model]] 监督、alignment reward
- **输出**: 专家混合策略 $\\sum_i w_i \\pi_i(s_t)$

### 核心模块

#### 模块1: [[Specialized Expert Policy]]

**设计动机**: 把技能拆开学，避免走路的梯度去污染拿取，搬运的平衡需求又去拖垮跑步。

**具体实现**:
- 每个专家对应一类基础技能；
- 用 human motion prior 做 imitation-constrained RL；
- 用 alignment operator 把人类动作先投到机器人配置空间，再形成 dense reward。

#### 模块2: [[Intelligent Routing Mechanism]]

**设计动机**: 专家拆开学不难，难的是复杂任务里什么时候该让谁发力。

**具体实现**:
- VLM 先根据任务语义给路由器一个 teacher-style 指导；
- 再用 demonstration-level behavioral refinement 把权重分布拉回真实示范统计；
- 最终控制时只跑轻量路由网络，不再实时查询 VLM。

### 关键公式

### 公式1: [[Imitation Learning|SEP 目标]]

$$
J_{{SEP}} = \\mathbb{{E}}_\\pi \\left[ \\sum_{{t=0}}^T \\gamma^t \\left(R_t + \\beta H(\\pi(\\cdot|s_t)) \\right) \\right]
$$

**含义**: SEP 不是纯追任务回报，而是把 imitation alignment reward 和策略熵一起塞进目标函数里，既保自然性也保探索。

**符号说明**:
- $R_t$: 基于动作对齐的能量型 imitation reward
- $H(\\pi)$: 策略熵
- $\\beta$: 熵正则权重

### 公式2: [[Reward Shaping|alignment reward]]

$$
R_t = w e^{{-\\alpha \\lVert q_t-q_t^* \\rVert^2}} + \\lambda e^{{-\\beta \\lVert \\dot{{q}}_t-\\dot{{q}}_t^* \\rVert^2}}
$$

**含义**: 位置和速度误差都要对齐，而且用指数形式保证离目标远时梯度还不至于彻底死掉。

**符号说明**:
- $q_t, q_t^*$: 机器人与参考轨迹的姿态
- $\\dot{{q}}_t, \\dot{{q}}_t^*$: 关节速度
- $w, \\lambda$: 两类误差的权重

### 公式3: [[Mixture-of-Experts|IRM 混合策略]]

$$
\\pi(a_t|s_t) = \\sum_{{i=1}}^K w_{{t,i}} \\pi_i(s_t)
$$

**含义**: IRM 输出的是专家权重而不是直接动作，核心是让组合空间落在一组已经被 human prior 校正过的技能上。

## 关键图表

### Figure 1: Overview / 系统概览

{meta_img}

**说明**: 论文首页图直接把结构说明白了。左边是 SEP 技能池，右边是 IRM 语义路由；本质就是用 VLM 指导的专家组合来顶替 monolithic policy。

### Figure 2: Dynamic Orchestration / 动态组合

- 论文说明 MetaWorld-X 通过专家动态编排实现自然动作。
- 重点不是 MoE 这个词，而是把 humanoid primitives 和 semantic routing 绑在一起。

### Figure 3: SEP Architecture / 专家训练

- 文中明确写到：先用 operator $M$ 把 human motion prior 投到机器人配置空间，再用 alignment operator $A$ 构成 tracking signal。
- 这一步是整篇 paper 最像“正经技术细节”的部分，不只是套个路由器。

### Figure 4: VLM Prompt / 路由提示

- 作者给了 VLM prompt 的示例，说明 IRM 的 teacher supervision 不是纯 end-to-end 学出来，而是人工可解释的 task semantic bootstrapping。

### Figure 5-9: Main Results / 实验结果

- Fig.5 看基础技能训练；
- Fig.6 对比 [[TD-MPC2]] 的收敛曲线；
- Fig.7/9 看复杂操作任务成功率和操控表现；
- Fig.8 是 door-opening 上的 ablation。

## 实验

### 任务与基线

| 项目 | 内容 |
|------|------|
| Benchmark | Humanoid-bench |
| 基础技能 | Stand, Walk, Run, Sit, Carry, Reach, Crawl |
| 基线 | [[TD-MPC2]], [[DreamerV3]], PPO, SAC |
| 关注指标 | return、convergence speed、motion quality、task success |

### 主要结果

1. 相比 [[TD-MPC2]]，MetaWorld-X 在 locomotion 和 manipulation 上都拿到更高回报和更快收敛。
2. 作者强调它在 naturalness 上也更强，这一点和加入 human motion prior 是对应的。
3. ablation 证明 SEP 和 IRM 是互补关系，不是随便拿掉一个还能差不多工作。

### 真实世界信号

文中写了 “real-world” 与 motion prior 和 model accuracy 相关讨论，但主实验核心还是在 benchmark 上。别把它误读成已经完整解决真实硬件上的 whole-body manipulation。

## 批判性思考

### 优点
1. 真正抓到了 humanoid monolithic policy 的结构性问题，而不是只在 reward 上打补丁。
2. SEP + IRM 的拆法有可解释性，远比“再堆一个更大的 transformer”像工程系统。
3. few-shot semantic transfer 这条线如果做实，对 task composition 是有价值的。

### 局限性
1. 它仍然依赖人工挑选的专家技能集合，没解决专家空间本身怎么学得更完整。
2. VLM 监督能否稳定迁移到超长任务、开放场景，摘要外证据还不够。
3. “world model framework” 这个标题写得有点大，真正的新意更多在层级路由和 imitation reward，不在 world model 本体。

### 潜在改进方向
1. 把专家扩展到更细粒度的手部/躯干协调，而不是只停在基础 locomotion primitives。
2. 让路由在执行期显式处理 uncertainty，而不是只做离线 teacher imitation。
3. 去真实硬件上检验 semantic routing 是否真的稳，不然容易停留在 benchmark 体感好。

### 可复现性评估
- [ ] 代码开源
- [ ] 预训练权重
- [x] 训练结构与主要公式公开
- [x] benchmark 与基线清晰

## 关联笔记

### 对比
- [[TD-MPC2]]: 代表 latent planning world model baseline。
- [[DreamerV3]]: 代表从像素学 latent dynamics 的另一条路。

### 方法相关
- [[Specialized Expert Policy]]: 技能专家池的核心定义。
- [[Intelligent Routing Mechanism]]: 语义路由器。
- [[Mixture-of-Experts]]: 专家组合范式。

### 硬件/数据相关
- [[Humanoid-bench]]: 主要评测集。

## 速查卡片

> [!summary] MetaWorld-X
> - **核心**: 把 humanoid loco-manipulation 拆成专家技能，再用语义路由组合。
> - **方法**: SEP + IRM + human motion prior + VLM supervision。
> - **结果**: Humanoid-bench 上优于 [[TD-MPC2]] / [[DreamerV3]]。
> - **判断**: 系统设计比标题里的 “world model” 更值得看。
""",
    )

    write(
        NOTES / "InterReal.md",
        f"""---
title: "InterReal: A Unified Physics-Based Imitation Framework for Learning Human-Object Interaction Skills"
method_name: "InterReal"
authors: [Dayang Liang, Yuhang Lin, Xinzhe Liu, Jiyuan Shi, Yunlong Liu, Chenjia Bai]
year: 2026
venue: arXiv
tags: [hoi, humanoid, imitation-learning, reward-learning, real-world]
zotero_collection: _inbox
image_source: local
arxiv_html: https://arxiv.org/html/2603.07516v1
created: 2026-03-10
---

# 论文笔记：InterReal

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Xiamen University, Zhejiang University, ShanghaiTech University, TeleAI / China Telecom |
| 日期 | March 2026 |
| 项目主页 | 未提供 |
| 对比基线 | [[InterMimic]], [[CooHOI]], [[TWIST]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.07516) / [HTML](https://arxiv.org/html/2603.07516v1) |

## 一句话总结

> InterReal 把 humanoid HOI 学习里最烦的两件事拆开处理：先用 contact-aware motion augmentation 扛住对象扰动，再用自动 reward learner 代替手工调一堆 tracking 权重。

## 核心贡献

1. **HOI motion augmentation**: 在手物接触约束下改物体位置，生成多条可训练交互轨迹。
2. **automatic reward learner**: 用一个 meta-policy 根据关键 tracking error 动态分配 reward 系数。
3. **real-world deployment**: 在 Unitree G1 上验证 box-picking / box-pushing 这类真实 HOI 任务。

## 问题背景

### 要解决的问题
传统 whole-body imitation 对“人和物一起动”的 HOI 非常脆弱。真实环境里只要物体姿态、相对位置稍微偏一点，policy 就会立刻 OOD。

### 现有方法的局限
- [[InterMimic]]、[[CooHOI]] 更像物理动画或模拟角色控制，对真实机器人约束考虑不够。
- whole-body teleop controller 可以做交互，但往往稳定性和闭环精度不够。
- 手工 reward shaping 对 humanoid HOI 太痛苦，信号多、目标冲突大、调参时间长。

### 本文的动机
作者想搭一个真正可部署的 HOI imitation 框架，不只是让角色“看起来像在搬箱子”，而是让 humanoid 在真实机器人上把箱子搬起来、推过去，还能容忍姿态扰动。

## 方法详解

### 模型架构

InterReal 采用 **physics-based imitation learning** 架构：
- **输入**: humanoid state、object state、reference motion、tracking errors
- **核心模块**: [[HOI Motion Augmentation]]、[[Automatic Reward Learner]]、multi-motion multi-environment training
- **输出**: 低层交互策略，直接驱动 Unitree G1 执行 HOI 动作

### 核心模块

#### 模块1: [[HOI Motion Augmentation]]

**设计动机**: 真实部署时对象位置一偏，policy 就容易崩，因为训练只见过单一 reference。

**具体实现**:
- 用 inverse kinematics 保持 hand-object contact 一致；
- 在改变 object pose 的同时生成多条训练轨迹；
- 让策略在 object perturbation 下仍能闭环完成任务。

#### 模块2: [[Automatic Reward Learner]]

**设计动机**: 把一堆 heterogeneous reward term 手工调平衡，本质上是浪费生命。

**具体实现**:
- 观察高层目标其实就是几个关键 tracking error 最小化；
- 训练一个 meta-policy 动态给低层 RL 分配 reward 权重；
- 权重会随误差变化而调整，而不是整场训练都锁死。

## 关键公式

### 公式1: [[Reward Shaping|meta reward allocation]]

$$
\\mathcal{{L}}_{{low}} = \\sum_i \\omega_i(t) \\mathcal{{R}}_i
$$

**含义**: 虽然论文正文细节在首段里没完整展开，但摘要和实验段落已经说清楚：核心不是新 reward 项，而是动态分配系数 $\\omega_i(t)$。

**符号说明**:
- $\\omega_i(t)$: meta-policy 在时刻 $t$ 给第 $i$ 个 reward 的权重
- $\\mathcal{{R}}_i$: 各类 tracking / contact / object-related reward

### 公式2: [[Contact Constraint|contact-consistent augmentation]]

$$
q'_{{human}}, o' = \\operatorname{{IK}}(q_{{human}}, o; \\text{{contact constraints}})
$$

**含义**: augmentation 不是随便抖动物体，而是要求手和物的接触关系在 IK 约束下仍然成立。

### 公式3: [[Tracking Error|critical error guidance]]

$$
e_t = [e^{{dof}}_t, e^{{obj}}_t, e^{{contact}}_t]
$$

**含义**: meta-policy 主要盯着几个关键 tracking error，再决定 reward 权重怎么调。

## 关键图表

### Figure 1: Real-world Deployment / 真实部署

{inter_img}

**说明**: 论文第一页直接放了 Unitree G1 的 live photos。上面是 picking + walking + placing 高密度箱子，下面是连续 pushing，这个取图选择挺诚实，说明作者知道自己卖点就是 deployment。

### Figure 2: Overall Framework / 总体流程

- 包含 motion preprocessing、multi-motion multi-environment learning 和 deployment 三部分；
- 重点是 retargeting HOI motions 到 G1 body shape，再把 object observation 一起并入 DRL。

### Figure 3: Tracking Accuracy / 跟踪精度

- 对比 box-picking 任务 tracking accuracy；
- 文中说 InterReal 在关键指标上误差最低。

### Figure 4: Ablation / 元学习系数

- 研究 meta-learning 内部系数 $\\delta$；
- 说明 automatic reward learner 不是噱头，至少作者认真做了敏感性分析。

### Figure 5: Adaptive Curves / reward 系数变化

- 展示 reward-related weights 会随训练或场景动态调整；
- 这是这篇 paper 真正最值钱的证据之一。

## 实验

### 任务设置

| 项目 | 内容 |
|------|------|
| 任务 | box-picking, box-pushing |
| 平台 | Unitree G1 |
| 指标 | tracking accuracy, task success rate |
| 对比对象 | recent HOI / teleop / imitation baselines |

### 主要结果

1. 两个 HOI 任务上都拿到最低 tracking error。
2. task success rate 也是最高。
3. 真实机器人部署说明它不是纯 sim 体操。

### 值得注意的地方

- 作者强调 real-time object posture feedback，这说明策略不是只靠 open-loop reference 硬跟；
- 但任务还是围绕箱体操作，离更复杂的多接触 HOI 还有距离；
- 成功率和 tracking 之外，对 fail cases 的分析摘要里没给多少。

## 批判性思考

### 优点
1. 方向非常对，HOI policy 如果不考虑 object perturbation，本来就不配谈 real-world。
2. automatic reward learner 抓住了 humanoid imitation 最浪费时间的环节。
3. 真实机器人验证比一堆纯 simulation HOI 工作强得多。

### 局限性
1. 任务分布还窄，主要是箱体 picking / pushing，泛化到更多 object geometry 还没证明。
2. 元策略到底学到了通用 reward allocation，还是只在这两类任务上过拟合，仍需观察。
3. contact consistency 的 augmentation 依赖 IK 和 reference quality，数据源差时可能会拖后腿。

### 潜在改进方向
1. 把自动 reward learner 扩展到更复杂的多阶段 dexterous HOI。
2. 明确加入 tactile 或 force feedback，不要只盯着视觉和姿态。
3. 多做一些 failure recovery 分析，尤其是 contact slip 和 object pose estimation 偏差。

### 可复现性评估
- [ ] 代码开源
- [ ] 权重开源
- [x] real-world platform 公开
- [x] 任务与实验设置清楚

## 关联笔记

### 基于 / 对比
- [[InterMimic]]: 早期物理动画风格 HOI 基线。
- [[CooHOI]]: cooperative HOI 类参考。
- [[TWIST]]: whole-body teleoperation controller 代表。

### 方法相关
- [[HOI Motion Augmentation]]: 解决交互扰动。
- [[Automatic Reward Learner]]: 解决 reward shaping。
- [[Contact Constraint]]: augmentation 成立的物理条件。

### 硬件/数据相关
- [[Unitree G1]]: 实际部署平台。

## 速查卡片

> [!summary] InterReal
> - **核心**: 接触一致的数据增强 + 自动 reward learner。
> - **方法**: IK 保接触，meta-policy 调 reward，最终落到 real-world HOI。
> - **结果**: box-picking / pushing 上 tracking 和成功率最好。
> - **判断**: 这是今天最像“真机器人论文”的一篇。
""",
    )

    write(
        NOTES / "AtomVLA.md",
        f"""---
title: "AtomVLA: Scalable Post-Training for Robotic Manipulation via Predictive Latent World Models"
method_name: "AtomVLA"
authors: [Xiaoquan Sun, Zetian Xu, Chen Cao, Zonghe Liu, Yihan Sun, Jingrui Pang, Ruijian Zhang, Zhen Yang, Kang Pang, Dingxin He, Mingqi Yuan, Jiayu Chen]
year: 2026
venue: arXiv
tags: [vla, world-model, offline-rl, robotic-manipulation, libero]
zotero_collection: _inbox
image_source: local
created: 2026-03-10
---

# 论文笔记：AtomVLA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | INFIFORCE, HKU, HUST, Tsinghua |
| 日期 | March 2026 |
| 项目主页 | 未提供 |
| 对比基线 | [[OpenVLA]], [[SmolVLA]], [[TinyVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.08519) / [PDF](https://arxiv.org/pdf/2603.08519v1) |

## 一句话总结

> AtomVLA 把 VLA 的长时程失败归因到 instruction grounding 不够细，于是先用 GPT-4o 把任务切成 atomic subtasks，再用 [[V-JEPA2]] world model 给 offline [[GRPO]] 打分做 post-training。

## 核心贡献

1. **Stage I subtask SFT**: 用 LLM 把高层示范拆成 2-5 个 atomic subtasks，再把这些子任务说明一起送进 SFT。
2. **Stage II offline GRPO**: 用 predictive latent world model 评估 candidate action chunks，不做昂贵的在线机器人 rollout。
3. **real-world validation**: 在 Galaxea R1 Lite 上测 basic 和 long-horizon tasks，强调 deformable object manipulation。

## 问题背景

### 要解决的问题
VLA 模型做多步操作时，最常见的死法不是单步预测不行，而是语言指令太粗，导致中间目标丢失，错误一层层累积。

### 现有方法的局限
- 只做 SFT 的 VLA 往往对 intermediate guidance 很弱。
- online RL 在真机上又贵又危险。
- generative world model 会有 hallucination，拿来当 reward 评估器不稳。

### 本文的动机
作者想走一条更便宜的后训练路径：不在真实机器人上 roll，不做像素级生成，而是在 latent space 里评估 candidate action 是否更接近 subgoal 和 final goal。

## 方法详解

### 模型架构

AtomVLA 是一个 **两阶段 post-training** 框架：
- **Stage I**: LLM 分解 subtasks，辅助 SFT 做 instruction grounding；
- **Stage II**: 冻结/接入 [[V-JEPA2]] latent world model，对候选 action chunk 打分，再用 [[GRPO]] 做离线策略优化；
- **输入**: 当前 observation $O_t$、高层指令、subtask 指令、candidate action chunks；
- **输出**: 更稳的 action chunk policy。

### 核心模块

#### 模块1: Atomic Subtask Decomposition

**设计动机**: 高层 instruction 太稀疏，不足以让 VLA 在长时程任务中持续保持语义对齐。

**具体实现**:
- 用 GPT-4o 把一个任务拆成 2-5 个 subtasks；
- 同时标注每个 subtask 的起止帧；
- 把 subtask instruction 和原始 instruction 一起喂给 Stage I SFT。

#### 模块2: Predictive Latent World Model

**设计动机**: 想要 RL refinement，但不想付出在线交互和像素生成成本。

**具体实现**:
- 使用 [[V-JEPA2]] 作为冻结视觉编码器 $J(\\cdot)$；
- predictor $W_\\theta$ 根据当前 observation 与 candidate action chunk rollout 预测未来 latent；
- 用 latent goal energy 评估是否更接近当前 subgoal 和 final goal。

#### 模块3: Offline [[GRPO]]

**设计动机**: 既然有 candidate-level reward，就可以做 group-wise preference optimization，而不必每步真机反馈。

**具体实现**:
- 每个 state 采样一组 candidate trajectories；
- 用 world model reward 排序归一化成 group advantages；
- 从 Stage I checkpoint 出发做离线 post-training。

## 关键公式

### 公式1: [[Subgoal|subgoal energy]]

$$
E^{{(k)}}_{{sub}} = \\left\\lVert W_\\theta\\big(J(O_t), \\tilde{{a}}^{{(k)}}_{{t:t+N}}\\big) - J(O_{{b(t)}}) \\right\\rVert_1
$$

**含义**: 候选动作 rollout 预测出来的 latent future，和当前子任务边界帧的 latent 表示越接近越好。

### 公式2: [[Goal Conditioned Policy|final-goal energy]]

$$
E^{{(k)}}_{{goal}} = \\left\\lVert W_\\theta\\big(J(O_t), \\tilde{{a}}^{{(k)}}_{{t:t+N}}\\big) - J(O_{{T-1}}) \\right\\rVert_1
$$

**含义**: 只盯 subgoal 容易短视，所以还要看 final goal consistency。

### 公式3: [[Reward Shaping|offline reward]]

$$
r^{{(k)}} = -\\left(\\lambda_{{sub}} E^{{(k)}}_{{sub}} + \\lambda_{{goal}} E^{{(k)}}_{{goal}} + \\alpha D^{{(k)}} \\right)
$$

**含义**: 论文里明确给了系数 $\\lambda_{{sub}}=0.3$、$\\lambda_{{goal}}=0.4$、$\\alpha=0.3$。再加一个 imitation deviation term 防 reward hacking。

## 关键图表

### Figure 1: Framework / 总体框架

{atom_img}

**说明**: 首页图很清楚。左边是 GPT-4o subtask 分解，中间是 world model 评估 candidate actions，右边是 LIBERO / LIBERO-PRO 和 real-world 结果。

### Figure 2: Why SFT Alone Fails / 只做 SFT 的问题

- 典型 VLA 只做高层 instruction 的 SFT；
- AtomVLA 认为这会让 semantic grounding 不够细，尤其是 long-horizon。

### Figure 3: Training Pipeline / 训练流程

- Stage I 负责 instruction refinement；
- Stage II 用 latent world model 提供 offline reward；
- 这是整篇 paper 最有说服力的结构图。

### Figure 4-5: Real-world Tasks / 真机实验

- 包含 stacking bowls、把 fruit 放进 basket、折 T-shirt 这类任务；
- 平台是 Galaxea R1 Lite。

### Figure 6-7: Generalization & Atomic Subtasks

- Fig.6 展示 GE variations；
- Fig.7 展示 LIBERO 数据集中任务被切成 coarse-to-atomic subtasks 的样子。

## 实验

### 数据与基准

| 项目 | 内容 |
|------|------|
| Benchmark | LIBERO, LIBERO-PRO |
| LIBERO 内容 | Spatial / Object / Goal / Long 四类 suites |
| Stage I 数据 | LIBERO demonstrations |
| Real-world platform | Galaxea R1 Lite |

### 主要结果

1. LIBERO 平均成功率 `97.0%`。
2. LIBERO-PRO 成功率 `48.0%`，说明在 perturbation 下比普通 benchmark 更难。
3. Stage II post-training 相比纯 SFT 全面提升，最大增益出现在 long-horizon goal alignment。
4. 真机上能做长时程与可变形物体任务，例如 folding T-shirt。

### 关键 ablation

- 移除语言 instruction，LIBERO-Long 明显掉点；
- 增加 subtask instruction 继续提升，说明原命题是成立的；
- reward 里同时保留 intermediate subtask guidance 和 final goal consistency 最稳。

## 批判性思考

### 优点
1. 准确抓到 VLA 长时程失败不是“动作模型不够大”，而是 intermediate grounding 不够。
2. 用 [[V-JEPA2]] 做 latent reward evaluator，比生成式 world model 少很多花活。
3. 离线 [[GRPO]] 的路线对机器人社区很现实，毕竟没人想天天烧真机做在线 RL。

### 局限性
1. subtask boundary 仍然是 LLM 离线标出来的，执行期并不会动态重划分。
2. 论文默认 world model 预测的 latent 距离和实际任务完成高度相关，这个假设不是永远成立。
3. 48% 的 LIBERO-PRO 虽然不错，但也说明鲁棒性离“真能信”还有距离。

### 潜在改进方向
1. 让 subtask generation 变成 execution-time adaptive，而不是一次性离线切段。
2. 给 world model 加 uncertainty 估计，别让 reward evaluator 过度自信。
3. 把 candidate evaluation 与 contact / force / language consistency 联合起来。

### 可复现性评估
- [ ] 代码与 checkpoint 尚未公开
- [x] 训练流程和 reward 公式清楚
- [x] benchmark 与 real-world 结果完整
- [x] ablation 较充分

## 关联笔记

### 对比
- [[OpenVLA]]: 开源基础 VLA 基线。
- [[SmolVLA]]: 轻量化 VLA 路线。
- [[TinyVLA]]: 小模型高效部署路线。

### 方法相关
- [[V-JEPA2]]: latent world model 的视觉底座。
- [[GRPO]]: group relative policy optimization。
- [[Subgoal]]: intermediate guidance 的关键概念。

### 硬件/数据相关
- [[LIBERO]]: 主 benchmark。
- [[Galaxea R1 Lite]]: 真机平台。

## 速查卡片

> [!summary] AtomVLA
> - **核心**: 用 LLM 拆 subtasks，用 latent world model + GRPO 做 offline post-training。
> - **方法**: Stage I SFT，Stage II world-model reward refinement。
> - **结果**: LIBERO 97.0%，LIBERO-PRO 48.0%，并有真机长任务验证。
> - **判断**: 是今天最值得继续跟的 VLA 论文之一。
""",
    )

    concept_note(
        "Specialized Expert Policy",
        "3-机器人策略",
        """---
type: concept
aliases: [SEP]
---

# Specialized Expert Policy

## 定义
把复杂机器人任务拆成多个专门技能策略分别学习的层级控制单元。

## 数学形式
$$
\\pi(a|s)=\\sum_i w_i\\pi_i(s)
$$

## 核心要点
1. 通过拆分技能减少梯度冲突。
2. 常与路由器或 MoE 一起工作。

## 代表工作
- [[MetaWorld-X]]: 用 SEP 学 humanoid 基础技能。

## 相关概念
- [[Mixture-of-Experts]]
- [[Intelligent Routing Mechanism]]
""",
    )

    concept_note(
        "Intelligent Routing Mechanism",
        "3-机器人策略",
        """---
type: concept
aliases: [IRM]
---

# Intelligent Routing Mechanism

## 定义
根据当前状态和任务语义，为多个技能专家分配权重的路由模块。

## 数学形式
$$
\\pi(a_t|s_t)=\\sum_i w_{t,i}\\pi_i(s_t)
$$

## 核心要点
1. 把任务语义映射为专家组合。
2. 常用 demonstration statistics 或 teacher signal 监督。

## 代表工作
- [[MetaWorld-X]]: 用 VLM 监督 IRM 组合技能专家。

## 相关概念
- [[Specialized Expert Policy]]
- [[Mixture-of-Experts]]
""",
    )

    concept_note(
        "Automatic Reward Learner",
        "2-强化学习",
        """---
type: concept
aliases: [Auto Reward Learner]
---

# Automatic Reward Learner

## 定义
根据关键误差或任务阶段动态分配 reward 权重的元策略。

## 数学形式
$$
\\mathcal{L}=\\sum_i \\omega_i(t)\\mathcal{R}_i
$$

## 核心要点
1. 减少人工 reward tuning。
2. 适合高维控制和多目标任务。

## 代表工作
- [[InterReal]]: 用 meta-policy 分配 HOI imitation 的 reward 信号。

## 相关概念
- [[Reward Shaping]]
- [[Tracking Error]]
""",
    )

    concept_note(
        "V-JEPA2",
        "11-深度学习基础",
        """---
type: concept
aliases: [Visual Joint Embedding Predictive Architecture 2]
---

# V-JEPA2

## 定义
一种基于 joint-embedding prediction 的视觉表征模型，可作为冻结视觉编码器或 latent dynamics 基座。

## 数学形式
$$
z = J(O)
$$

## 核心要点
1. 直接在 latent space 做预测而不是像素生成。
2. 适合给 world model 或规划模块提供稳定视觉表征。

## 代表工作
- [[AtomVLA]]: 用 V-JEPA2 编码 observation 并评估 candidate actions。

## 相关概念
- [[World Model]]
- [[GRPO]]
""",
    )

    concept_note(
        "GRPO",
        "2-强化学习",
        """---
type: concept
aliases: [Group Relative Policy Optimization]
---

# GRPO

## 定义
基于同组候选动作相对优劣进行策略优化的方法。

## 数学形式
$$
A^{(k)} = \\operatorname{Norm}(r^{(k)}_1, \\dots, r^{(k)}_K)
$$

## 核心要点
1. 关注组内相对排序而非绝对标量回报。
2. 适合 preference-style 或 candidate-style 策略学习。

## 代表工作
- [[AtomVLA]]: 用 world-model reward 做 offline GRPO post-training。

## 相关概念
- [[V-JEPA2]]
- [[Reward Shaping]]
""",
    )

    daily_text = DAILY.read_text(encoding="utf-8")
    replacements = {
        "### 1. MetaWorld-X: Hierarchical World Modeling via VLM-Orchestrated Experts for Humanoid Loco-Manipulation\n- **作者**": "### 1. MetaWorld-X: Hierarchical World Modeling via VLM-Orchestrated Experts for Humanoid Loco-Manipulation\n- 📒 **笔记**: [[MetaWorld-X]]\n- **作者**",
        "### 2. InterReal: A Unified Physics-Based Imitation Framework for Learning Human-Object Interaction Skills\n- **作者**": "### 2. InterReal: A Unified Physics-Based Imitation Framework for Learning Human-Object Interaction Skills\n- 📒 **笔记**: [[InterReal]]\n- **作者**",
        "### 11. AtomVLA: Scalable Post-Training for Robotic Manipulation via Predictive Latent World Models\n- **作者**": "### 11. AtomVLA: Scalable Post-Training for Robotic Manipulation via Predictive Latent World Models\n- 📒 **笔记**: [[AtomVLA]]\n- **作者**",
    }
    for old, new in replacements.items():
        if old in daily_text and new not in daily_text:
            daily_text = daily_text.replace(old, new)
    DAILY.write_text(daily_text, encoding="utf-8")


if __name__ == "__main__":
    main()
