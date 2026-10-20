# VGGT 海光 DCU 复测项目

按客户 VGGT 报告的 Phase 1–6 验证 BW100 上的环境、推理、性能、训练图与 8 卡 DDP。主要代码为测试脚本，VGGT 原项目与模型权重通过路径引用，不随本项目重复分发。

本次交付：[完整测试报告](reports/VGGT_DCU_完整测试报告_20261020.md)、[原始结果与日志](results/20261020_flash/)、[客户原报告](references/VGGT_海光DCU_验证报告与Runbook.md)。

本轮使用 Flash Attention。BF16 主干强制 `SDPBackend.FLASH_ATTENTION`，失败即报错，不允许主干静默回退。VGGT 原代码的相机头以 FP32 运行，保留其 MATH 注意力；客户步骤中的 FP32 冒烟与自一致性检查也保留。这两点不属于 MATH 性能对照。

## 1. 环境

已验证镜像：

```text
harbor.sourcefind.cn:5443/hcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10
```

当前服务器容器名为 `zk-vggt`，项目目录为 `/home/zk/vggt/validation-project`，VGGT 源码为 `/home/zk/vggt/vggt`，权重为 `/home/zk/vggt/vggt_ckpt/model.pt`。

新环境启动示例（在服务器宿主机执行，已有容器时无需重复）：

```bash
docker run -d --name zk-vggt \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd \
  --group-add video --ipc=host --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /home/zk/vggt:/home/zk/vggt \
  -w /home/zk/vggt/validation-project \
  harbor.sourcefind.cn:5443/hcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10 \
  sleep infinity
```

需要宿主机驱动与该镜像配套；设备节点按实际主机提供。本次使用已有容器，实际容器配置另存于结果目录。不要直接安装上游 `requirements.txt` 中的 CUDA torch / torchvision，也不要导入客户的 `dcu_compat.py`。

容器内补充训练 loss 依赖：

```bash
python -m pip install -r requirements-extra.txt
```

## 2. 准备源码与权重

已有客户材料时直接使用上面的 `--source` 和 `--checkpoint`。干净环境可以运行：

```bash
python scripts/prepare_source.py
python scripts/download_checkpoint.py
```

源码固定在本次验证的官方提交 `a288dd0f14786c93483e45524328726ab7b1b4ce`。下载脚本使用 `facebook/VGGT-1B` 的 `model.pt`；权重哈希由环境脚本记录。下载服务如需配置，用 `HF_ENDPOINT`、`HF_TOKEN` 环境变量，勿将 token 写入 Git。

## 3. 一键完整运行

服务器宿主机执行：

```bash
docker exec -w /home/zk/vggt/validation-project zk-vggt \
  python -u scripts/run_suite.py \
    --source /home/zk/vggt/vggt \
    --checkpoint /home/zk/vggt/vggt_ckpt/model.pt \
    --output results/my_run
```

每次复测使用新输出目录，避免混用历史结果。结果包含 JSON、完整日志、命令、退出码和 UTC 起止时间。任一阶段失败，最终退出码非零，其余阶段仍继续，方便收集完整证据。性能扫帧遇到 OOM 时记录容量边界并停止该分辨率的更大帧数，OOM 不伪装成执行通过。

结束时自动生成输出目录中的 `report.md`，也可重新生成：

```bash
python scripts/make_report.py results/my_run --output reports/测试报告.md
```

报告中的服务器、镜像与日期说明针对本次已验证环境；在其他机器复测时应相应更新这些说明。

运行顺序：

| 阶段 | 脚本 | 检查内容 |
| --- | --- | --- |
| Phase 1 | `environment.py` | 8 卡、版本、源码提交、权重 SHA256 |
| 补充 | `flash.py` | FP16/BF16，强制 Flash SDPA 与直接 Flash 接口的前向、反向；算子记录 |
| Phase 2 | `inference.py smoke` | 随机权重，4×518 FP32、1×224 FP32、4×224 BF16；形状与有限性 |
| Phase 3 | `inference.py correctness` | 真实权重，客户随机图／相同帧检查，补充官方 kitchen 图像与 Flash 算子检查 |
| Phase 4 | `inference.py benchmark` | BF16 Flash，518/224，S=1/4/8/16/32/64/128，预热 2 次、计时 5 次 |
| Phase 5 | `training.py` | camera+depth，真实 MultitaskLoss，合成 GT，单卡 S=6、224，优化一步 |
| Phase 6 | `training.py --ddp` | 8 卡 S=4、224，真实 loss，3 步，通信与同步检查 |

Phase 4 一键默认使用随机初始化权重，与客户 benchmark 脚本一致。需要真实权重性能时，单独运行 benchmark 并省略 `--random-init`。性能时间为模型前向平均耗时，不包含数据加载与输出有限性检查；显存给出 allocator 的 allocated / reserved 峰值，单位为十进制 GB。

## 4. 分阶段运行与重跑

进入容器，在项目根目录设置路径：

```bash
export VGGT_SOURCE=/home/zk/vggt/vggt
export VGGT_CHECKPOINT=/home/zk/vggt/vggt_ckpt/model.pt
export VGGT_OUTPUT=$PWD/results/manual
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export NCCL_SOCKET_IFNAME=lo PYTHONFAULTHANDLER=1

python scripts/environment.py
python scripts/flash.py
python scripts/inference.py smoke
python scripts/inference.py correctness
python scripts/inference.py benchmark --random-init
python scripts/training.py --steps 1 --frames 6
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/training.py --ddp --steps 3 --frames 4
```

也可用统一入口只重跑指定阶段：

```bash
python scripts/run_suite.py --source "$VGGT_SOURCE" \
  --checkpoint "$VGGT_CHECKPOINT" --output "$VGGT_OUTPUT" \
  --phases training ddp
```

相同输出目录的同名阶段会覆盖日志和 JSON，manifest 更新对应记录。正式复测建议新目录保留历史。DDP 各 rank 使用不同随机输入；检查全部参数梯度的有限性，并在清空梯度前，对每个可训练张量最多 257 个等距样本做跨卡一致性验证，优化后同样检查参数样本一致性及更新。采样同步检查不等同于全张量逐元素校验。训练用 BF16 autocast + GradScaler（初始 scale=65536），检查优化步未被跳过。单卡 AdamW 使用 lr=5e-5、weight_decay=0.05；DDP 按客户脚本使用 lr=1e-4、weight_decay=0.01。

### 本镜像的训练兼容处理

复测中，原生 Q/K/V 布局的 Flash 反向在 VGGT 训练中返回 NaN；降低 loss scale 后仍可复现，独立的连续张量 Flash 算子测试则正常。训练脚本默认在进入 SDPA 前执行 `q.contiguous()`、`k.contiguous()`、`v.contiguous()`，保持 Flash 后端和模型数值精度。处理仅在测试脚本的上下文内生效，不修改上游源码；也覆盖激活检查点的反向重计算。推理基准未启用这项训练处理。

可用 `--no-contiguous-qkv` 重现原始布局路径；`--anomaly` 开启反向异常定位。原始失败与诊断证据随正式结果保存。此处仅报告当前镜像和模型组合的实测现象，不泛化为所有 Flash 实现的限制。

八卡启动还曾在并行构建模型时的 CPU `trunc_normal_/erfinv` 出现 SIGSEGV。脚本默认使用 1 个 CPU 线程完成模型初始化，随后恢复 8 线程计算，完整八卡测试已通过；可通过 `--init-threads` 调整。该处理仅改变初始化线程数，未更改初始化函数或冻结模型参数。

统一入口默认设置 `NCCL_SOCKET_IFNAME=lo`（与客户单机 DDP 命令一致）和 `PYTHONFAULTHANDLER=1`，外部已设置的网卡值会保留。还提供 `diagnose_flash.py` 与 `distributed_probe.py`，分别用于最小 Flash 形状／梯度和八卡通信诊断。

## 5. 验收边界

- 这套流程与客户已执行的 VGGT 测试范围对齐；不包含材料中的独立硬件压力测试或 VGGT-Omega 报告。
- 未提供 CO3D 数据集，因此不运行官方 `training/launch.py` 的完整数据集训练；合成 GT 训练图通过不等同于完整训练收敛。
- 位姿合法、深度有限、相同帧一致只验证基本数值行为。真实 kitchen 图片是推理冒烟，没有任务 GT，不能宣称重建精度已达标。
- Flash 两种入口的一致性只证明调用路径一致，不是独立算法正确性证明。本轮按要求不做 MATH 性能或精度基准。
- 客户与本轮 torch、环境、权重／AMP 配置等有差异，跨报告性能仅作观察，不把变化全部归因于 Flash。

## 6. Git 与结果

本仓库跟踪脚本、教程、报告与本次测试证据；忽略模型权重、上游源码、大型二进制和本机连接配置。`results/initial` 为首轮调查历史，正式完整复测结果以 `results/20261020_flash` 为准。

```bash
git log --oneline
git status
```

交付包如包含 `.bundle`，可以离线恢复完整 Git 历史：

```bash
git clone vggt-dcu-validation.bundle vggt-dcu-validation
```
