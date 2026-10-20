# VGGT 海光 DCU 测试项目

本项目用于验证 VGGT 在海光 BW100 上的图片处理、Flash Attention 加速、单卡训练和八卡协同训练能力，提供测试脚本、运行教程和实测数据。

完整说明见 [VGGT 海光 BW100 测试报告](reports/VGGT_DCU_完整测试报告_20261020.md)。

## 1. 测试环境

| 项目 | 配置 |
| --- | --- |
| CPU | 2 × 海光 C86-4G（7490），合计 128 核、256 逻辑线程 |
| 系统内存 | 约 751 GiB |
| 加速卡 | 8 × 海光 BW100，单卡为 64GB 级显存 |
| 操作系统 | Ubuntu 22.04.5 LTS |
| 开发工具包 | DTK 26.04 |
| Python | 3.10.12 |
| PyTorch / torchvision | 2.9.0 / 0.24.0，海光适配版 |
| Flash Attention | 2.6.1，海光适配版 |
| 模型 | VGGT-1B |

使用的镜像：

```text
harbor.sourcefind.cn:5443/hcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10
```

## 2. 实测结果

测试日期：2026-10-20。运行环境、Flash 加速、模型基本运行、官方预训练模型、推理性能、单卡训练流程和八卡协同训练，7 个测试项目全部通过。

### 图片处理速度与显存

单卡使用 BF16 计算格式和 Flash Attention；模型参数随机初始化。每组先预热 2 次，再运行 5 次取平均。耗时对应整组图片的模型计算，不包含读取图片；1000 毫秒等于 1 秒。

| 图片数量 | 单张图片分辨率 | 结果 | 整组平均耗时 ms | 每秒图片数 | 计算使用显存峰值 GB | 框架预留显存峰值 GB |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 518×518 | 通过 | 188.7 | 5.30 | 7.56 | 8.33 |
| 4 | 518×518 | 通过 | 529.4 | 7.56 | 8.34 | 9.03 |
| 8 | 518×518 | 通过 | 1016.7 | 7.87 | 9.63 | 11.16 |
| 16 | 518×518 | 通过 | 2203.7 | 7.26 | 10.05 | 13.29 |
| 32 | 518×518 | 通过 | 5208.1 | 6.14 | 10.90 | 13.87 |
| 64 | 518×518 | 通过 | 13968.2 | 4.58 | 12.79 | 16.12 |
| 128 | 518×518 | 通过 | 42181.2 | 3.03 | 18.70 | 23.48 |
| 1 | 224×224 | 通过 | 102.7 | 9.74 | 7.01 | 7.13 |
| 4 | 224×224 | 通过 | 160.1 | 24.99 | 7.16 | 7.75 |
| 8 | 224×224 | 通过 | 225.2 | 35.52 | 7.40 | 8.07 |
| 16 | 224×224 | 通过 | 401.1 | 39.89 | 7.49 | 8.25 |
| 32 | 224×224 | 通过 | 754.7 | 42.40 | 7.64 | 8.24 |
| 64 | 224×224 | 通过 | 1562.8 | 40.95 | 8.00 | 8.69 |
| 128 | 224×224 | 通过 | 3553.3 | 36.02 | 9.12 | 9.64 |

14 组用例全部通过，两种分辨率均完成一次输入 128 张图片的测试。“计算使用显存”是分配给模型数据和计算结果的空间；“框架预留显存”还包括留待后续复用的空间，两列不能相加。

### 模型结果与训练

| 测试项目 | 实测结果 |
| --- | --- |
| Flash Attention | FP16、BF16 的图片处理计算和训练反向计算均通过；模型运行记录确认实际调用 Flash |
| 官方预训练模型 | kitchen 场景图片可输出有效结果，相同输入重复运行的主要结果完全一致 |
| 相同图片检查 | 相机参数波动 0.000068，深度波动 0.003909，均低于测试阈值 0.01 |
| 单卡训练 | 6 张 224×224 图片，完成 1 次参数更新；训练及检查耗时 6.03 秒，显存峰值 23.26GB |
| 八卡协同训练 | 每卡 4 张 224×224 图片，连续完成 3 次参数更新；单卡显存峰值最大 23.25GB |

八卡训练使用不同输入，实测数据如下。误差指标用于记录训练流程，不是实际业务准确率。

| 训练步骤 | 8 张卡的训练误差指标范围 | 训练计算抽样一致性 | 更新后参数抽样一致性 |
| ---: | --- | --- | --- |
| 1 | 13.705935–14.056774 | 通过，差异为 0 | 通过，差异为 0 |
| 2 | 27.918133–27.971882 | 通过，差异为 0 | 通过，差异为 0 |
| 3 | 23.940680–23.981804 | 通过，差异为 0 | 通过，差异为 0 |

本次验证了基础运行与短步训练能力，未包含完整数据集训练、业务精度评估和长时间稳定性测试。

## 3. 准备运行环境

### 启动容器

将项目放在 `/home/zk/vggt/validation-project`，在服务器宿主机执行：

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

容器需使用配套的宿主机驱动和设备节点。镜像已提供适配海光的 PyTorch、torchvision 和 Flash Attention，保留这些组件的版本。

### 准备模型代码与参数

进入容器和项目目录，执行：

```bash
docker exec -it zk-vggt bash
cd /home/zk/vggt/validation-project

python -m pip install -r requirements-extra.txt
python scripts/prepare_source.py
python scripts/download_checkpoint.py
```

模型代码下载到 `vendor/vggt`，预训练参数下载到 `checkpoints/model.pt`。模型代码固定在提交 `a288dd0f14786c93483e45524328726ab7b1b4ce`；参数文件为 `facebook/VGGT-1B` 的 `model.pt`。下载服务可通过 `HF_ENDPOINT`、`HF_TOKEN` 环境变量配置。

也可在运行命令中用 `--source`、`--checkpoint` 指向已准备好的模型代码和参数文件。

## 4. 一键测试

在容器内的项目目录执行：

```bash
python -u scripts/run_suite.py \
  --source vendor/vggt \
  --checkpoint checkpoints/model.pt \
  --output results/my_run
```

输出包括测试数据、运行日志和自动生成的 `report.md`。测试项目及对应脚本如下：

| 测试项目 | 脚本 | 检查内容 |
| --- | --- | --- |
| 运行环境 | `environment.py` | 设备、软件版本及模型文件信息 |
| Flash 加速 | `flash.py` | FP16/BF16 计算及两种调用方式的一致性 |
| 模型基本运行 | `inference.py smoke` | 不同图片数量、尺寸和计算格式 |
| 预训练模型 | `inference.py correctness` | 真实图片、相同输入与重复运行 |
| 性能与显存 | `inference.py benchmark` | 2 种分辨率、7 档图片数量 |
| 单卡训练 | `training.py` | 完成计算、误差分析和参数更新 |
| 八卡协同训练 | `training.py --ddp` | 通信、参数同步和连续 3 次更新 |

一键测试的性能部分使用随机初始化参数。单独运行 `benchmark` 并省略 `--random-init`，可测试预训练模型的性能。

手动生成报告：

```bash
python scripts/make_report.py results/my_run --output reports/测试报告.md
```

## 5. 分项运行

在容器内设置路径并执行所需项目：

```bash
export VGGT_SOURCE=$PWD/vendor/vggt
export VGGT_CHECKPOINT=$PWD/checkpoints/model.pt
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

统一入口也支持选择测试项目，例如：

```bash
python scripts/run_suite.py --source "$VGGT_SOURCE" \
  --checkpoint "$VGGT_CHECKPOINT" --output "$VGGT_OUTPUT" \
  --phases training ddp
```

### 训练配置

训练使用 BF16 和 Flash Attention，启用相机参数与深度估计任务。单卡 AdamW 参数为 `lr=5e-5`、`weight_decay=0.05`；八卡为 `lr=1e-4`、`weight_decay=0.01`。GradScaler 的初始缩放值为 65536。

脚本默认将 Q/K/V 数据转为连续存储后进入 Flash 计算；模型初始化使用 1 个 CPU 线程，随后恢复为 8 线程计算。相关选项为 `--contiguous-qkv`、`--init-threads`，异常追踪选项为 `--anomaly`。八卡同步检查对每组参数及训练计算结果各抽取最多 257 个数据点。

BF16 模型主干使用 Flash Attention；模型中负责相机参数计算的 FP32 部分保持原有精度设置。额外诊断工具为 `diagnose_flash.py`（Flash 计算检查）和 `distributed_probe.py`（八卡通信检查）。
