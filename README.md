# 🎮 CartPole 强化学习实战项目

本项目基于 OpenAI Gymnasium 的 **CartPole-v1**（倒立摆）环境，从零到一实现了五种从简单到复杂的强化学习算法，循序渐进地展示强化学习的核心思想与技术演进。

> **GitHub 仓库**: [https://github.com/yangyue68/Cartpole-game](https://github.com/yangyue68/Cartpole-game)

---

## 📂 项目结构

```
CartPole/
├── README.md                                       # 本文件
│
├── 实现一_随机猜测/                                 # ⭐ 随机策略（Baseline）
│   ├── 1-cartpole.ipynb                            #   Jupyter Notebook 演示
│   └── cartpole-v0-random-policy.py                #   Python 脚本
│
├── 实现二_最简单的策略梯度法/                        # ⭐ REINFORCE 算法（最简版）
│   ├── model.py                                    #   策略网络 & Agent 定义
│   ├── train.py                                    #   训练脚本
│   ├── test.py                                     #   测试/可视化脚本
│   └── policy_model.pth                            #   预训练模型权重
│
├── 实现三_REINFORCE改进的策略梯度法/                 # ⭐ REINFORCE 算法（改进版）
│   ├── model.py                                    #   策略网络 & Agent 定义
│   ├── train.py                                    #   训练脚本
│   ├── test.py                                     #   测试/可视化脚本
│   └── policy_model.pth                            #   预训练模型权重
│
├── 实现四_Actor-Critic架构的策略梯度法/              # ⭐ Actor-Critic 算法
│   ├── model.py                                    #   策略网络 & 价值网络 & Agent 定义
│   ├── train.py                                    #   训练脚本
│   └── test.py                                     #   测试/可视化脚本
│
└── 实现五_基于PPO的倒立摆游戏/                      # ⭐ PPO 算法（进阶）
    ├── model.py                                    #   策略网络 & 价值网络定义
    ├── ppo.py                                      #   PPO 算法核心实现
    ├── train.py                                    #   训练脚本
    └── test.py                                     #   测试/可视化脚本
```

---

## 🧠 五种算法介绍

### 1️⃣ 实现一：随机猜测（Random Policy）

**文件**: `实现一_随机猜测/`

作为 Baseline，智能体完全随机地选择向左推或向右推，不包含任何学习机制。用于对比验证后续算法的学习效果。

- 动作：随机选择 `{0: "向左推", 1: "向右推"}`
- 计算折扣回报（Discounted Return）
- 运行方式：

```bash
cd "实现一_随机猜测"
python cartpole-v0-random-policy.py
```

---

### 2️⃣ 实现二：最简单的策略梯度法（REINFORCE）

**文件**: `实现二_最简单的策略梯度法/`

Vanilla REINFORCE 算法（蒙特卡洛策略梯度）。使用一个简单的策略神经网络，对完整轨迹的每个时间步使用**相同的总回报 G(τ)** 来更新策略。

$$
\nabla J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G(\tau) \right]
$$

- 策略网络：`4 → 128 → 2`（全连接 + ReLU + Softmax）
- 学习率：0.0002
- 折扣因子 γ：0.98

```bash
cd "实现二_最简单的策略梯度法"
python train.py          # 训练模型（3000 回合）
python test.py           # 加载预训练模型进行可视化测试
```

---

### 3️⃣ 实现三：REINFORCE 改进的策略梯度法

**文件**: `实现三_REINFORCE改进的策略梯度法/`

在 REINFORCE 基础上改进：不再是每条轨迹只用一个全局回报 G(τ)，而是对**每个时刻 t 使用从该时刻起的折扣回报 G(t)**（即奖励-to-go），使得梯度更新更加精准。

$$
\nabla J(\theta) = \mathbb{E} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right], \quad G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k
$$

- 网络结构与实现二相同
- 区别仅在于 `update()` 方法中逆序遍历时逐步累积 G

```bash
cd "实现三_REINFORCE改进的策略梯度法"
python train.py          # 训练模型（3000 回合）
python test.py           # 加载预训练模型进行可视化测试
```

---

### 4️⃣ 实现四：Actor-Critic 架构的策略梯度法

**文件**: `实现四_Actor-Critic架构的策略梯度法/`

引入 **Critic（价值网络）** 来评估状态价值 $V(s)$，用 **TD 误差** $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 作为 Advantage 的近似，替代蒙特卡洛回报。

- **Actor（策略网络）**: $\pi_\theta(a|s)$ — 决定采取什么动作
- **Critic（价值网络）**: $V_\omega(s)$ — 评估当前状态的好坏
- **更新方式**: 每一步即时更新（在线学习），无需等完整轨迹

```
Actor 损失: -log π(a|s) * δ
Critic 损失: MSE(r + γV(s') - V(s))
```

- Actor 学习率：0.0002
- Critic 学习率：0.0005

```bash
cd "实现四_Actor-Critic架构的策略梯度法"
python train.py          # 训练模型（2000 回合）
python test.py           # 加载训练好的模型进行可视化测试
```

---

### 5️⃣ 实现五：基于 PPO 的倒立摆游戏

**文件**: `实现五_基于PPO的倒立摆游戏/`

**PPO（Proximal Policy Optimization，近端策略优化）** 是目前最主流的强化学习算法之一。它在保证训练稳定性的同时具备较高的样本效率。

核心思想：通过 ** clipped surrogate objective（截断替代目标）** 限制每次策略更新的步长，防止策略发生剧烈变化。

$$
L^{CLIP}(\theta) = \mathbb{E} \left[ \min\left( r_t(\theta) \hat{A}_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是新旧策略的比率，$\epsilon = 0.2$ 为裁剪阈值。

- 使用 **GAE（Generalized Advantage Estimation）** 计算优势函数
- 每条轨迹采样完成后，用固定数据更新 **epochs=10** 轮
- 同时更新 Actor 和 Critic 网络

```bash
cd "实现五_基于PPO的倒立摆游戏"
python train.py          # 训练模型（500 回合，配有 tqdm 进度条）
python test.py           # 加载训练好的模型进行可视化测试
```

---

## ⚙️ 环境配置

本项目的 Python 环境为 `reinforce3`（Conda），依赖如下：

```bash
# 创建并激活虚拟环境（Windows）
conda create -n reinforce3 python=3.10 -y
conda activate reinforce3

# 安装依赖
pip install torch numpy matplotlib gymnasium tqdm
```

| 依赖 | 版本（参考） |
|------|------------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0 |
| Gymnasium | ≥ 1.0 |
| NumPy | ≥ 2.0 |
| Matplotlib | ≥ 3.0 |
| tqdm | ≥ 4.0 |

> **注意**: 本项目使用 `gymnasium`（Gym 的官方维护分支），**不是**旧版 `gym`。旧版 `gym` 与 NumPy 2.x 存在兼容性问题。

---

## 🚀 快速开始

| 步骤 | 命令 |
|------|------|
| 1. 克隆仓库 | `git clone https://github.com/yangyue68/Cartpole-game.git` |
| 2. 进入目录 | `cd Cartpole-game` |
| 3. 随机策略演示 | `python "实现一_随机猜测/cartpole-v0-random-policy.py"` |
| 4. 训练 REINFORCE | `python "实现二_最简单的策略梯度法/train.py"` |
| 5. 测试 REINFORCE | `python "实现二_最简单的策略梯度法/test.py"` |

---

## 📈 算法对比

| 算法 | 更新方式 | 是否需要完整轨迹 | 是否使用价值网络 | 训练稳定性 | 收敛速度 |
|------|---------|:---:|:---:|:---:|:---:|
| 随机猜测 | ❌ 无学习 | - | ❌ | - | - |
| REINFORCE（简单） | 蒙特卡洛（整条轨迹） | ✅ | ❌ | 低 | 慢 |
| REINFORCE（改进） | 蒙特卡洛（reward-to-go） | ✅ | ❌ | 中 | 中 |
| Actor-Critic | TD 学习（每步） | ❌ | ✅ | 中 | 快 |
| PPO | 截断替代目标 | ✅ | ✅ | **高** | **快** |

---

## 📝 许可证

本项目仅供学习参考，欢迎 Star 和 Fork！
