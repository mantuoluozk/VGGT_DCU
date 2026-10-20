# VGGT 海光 DCU 测试项目

本项目用于验证 VGGT 在海光 BW100 上的图片处理、Flash Attention 加速、单卡训练和八卡协同训练能力，提供测试脚本、运行教程和实测数据。

完整说明见 [VGGT 海光 BW100 测试报告](reports/VGGT_DCU_完整测试报告_20261020.md)。

## 1. 测试环境

| 项目 | 配置 |
| --- | --- |
| 服务器型号 | X7850H0 |
| CPU | 2 × 海光 C86-4G（7490），合计 128 核、256 逻辑线程 |
| 系统内存 | 约 751 GiB |
| 加速卡 | 8 × 海光 BW100，单卡显存 64GB |
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

需要准备两项内容：**VGGT 模型代码**用于构建和运行模型，**预训练参数 `model.pt`**用于加载官方已经训练好的模型。下面的命令均在容器内、本项目根目录执行。

**第一步：进入容器并安装辅助依赖。**

```bash
docker exec -it zk-vggt bash
cd /home/zk/vggt/validation-project

python -m pip install -r requirements-extra.txt
```

**第二步：下载官方模型代码。**

```bash
python scripts/prepare_source.py --destination vendor/vggt
```

脚本自动获取 VGGT 官方代码，并固定到本次测试使用的版本 `a288dd0f14786c93483e45524328726ab7b1b4ce`。代码保存在 `vendor/vggt`，其中包括模型实现、训练误差计算方法和示例图片。此步骤需要网络连接及 `git` 命令，测试镜像已提供 `git`。

**第三步：下载预训练参数。**

```bash
python scripts/download_checkpoint.py --destination checkpoints/model.pt
```

脚本从 [VGGT-1B 官方模型文件页面](https://huggingface.co/facebook/VGGT-1B/tree/main) 下载 `model.pt`，文件约 5GB，保存在 `checkpoints/model.pt`。本项目使用的是 `model.pt` 文件。

也可以从上述页面手动下载 `model.pt`，或使用下面的命令，然后放到同一位置：

```bash
mkdir -p checkpoints
curl -L --fail --retry 3 -C - \
  https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt \
  -o checkpoints/model.pt
```

**第四步：确认文件位置。**

准备完成后，主要目录应如下所示：

```text
validation-project/
├── scripts/                          # 本项目的测试脚本
├── vendor/vggt/
│   ├── vggt/models/vggt.py            # 模型代码
│   ├── training/loss.py              # 训练误差计算代码
│   └── examples/kitchen/images/      # 示例图片
└── checkpoints/model.pt              # 官方预训练参数，约 5GB
```

可用以下命令检查关键文件是否存在：

```bash
ls -lh vendor/vggt/vggt/models/vggt.py \
  vendor/vggt/training/loss.py checkpoints/model.pt
```

如需核对是否与本次实验使用相同参数文件，可运行 `sha256sum checkpoints/model.pt`，本次实验记录的校验值为：

```text
d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0
```

服务器无法联网时，可在可联网的机器上完成下载，再将整个模型代码目录和 `model.pt` 复制到服务器的对应位置。下载脚本也支持通过 `HF_ENDPOINT` 设置模型下载服务地址，通过 `HF_TOKEN` 配置需要登录时使用的访问凭据。

模型代码和参数位于其他目录时，可直接通过 `--source`、`--checkpoint` 指定，无需再次下载。例如本次测试服务器上已有的路径可这样使用：

```bash
python -u scripts/run_suite.py \
  --source /home/zk/vggt/vggt \
  --checkpoint /home/zk/vggt/vggt_ckpt/model.pt \
  --output results/my_run
```

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

## 6. 可选推理优化

提供 `--preset fused` 选项，通过重用位置数据、合并位置变换计算，以及调整深度和三维点输出的分组大小减少耗时。输入图片数量、分辨率、模型参数和输出字段保持不变；全部图片仍在主干中联合分析。

以下使用官方预训练参数重新成对测量，每组预热 1 次、计时 3 次取中位数。与前面的随机初始化性能表分别记录。

| 输入 | 图片数量 | 原实现 ms | 优化后 ms | 耗时降低 | 输出数值对照 |
| --- | ---: | ---: | ---: | ---: | --- |
| 518×518 | 8 | 1027.91 | 976.49 | 5.00% | 通过 |
| 518×518 | 32 | 5240.06 | 4871.83 | 7.03% | 通过 |
| 518×518 | 64 | 14024.36 | 13312.74 | 5.07% | 通过 |
| 518×518 | 128 | 42294.93 | 40896.30 | 3.31% | 通过 |
| 224×224 | 32 | 764.51 | 694.80 | 9.12% | 通过 |
| 224×224 | 128 | 3578.26 | 3362.23 | 6.04% | 通过 |
| 官方实景图片 | 4 | 348.26 | 314.57 | 9.67% | 通过 |

优化可能增加临时显存使用，例如 518×518、64 张时由 12.99GB 增至 15.17GB。完整数据、数值检查范围和说明见 [推理优化评估](reports/VGGT_DCU_推理优化评估_20261020.md)。

在容器内运行优化后的模型检查：

```bash
python scripts/inference.py correctness --preset fused \
  --source /home/zk/vggt/vggt \
  --checkpoint /home/zk/vggt/vggt_ckpt/model.pt \
  --output results/inference_fused
```

将 `correctness` 换为 `benchmark` 可测量预训练模型性能。`fused` 需要本次镜像已提供的 Triton，首次运行包含编译开销；性能计时前需要预热。`--preset balanced` 仅使用位置数据重用和输出分组，默认 `--preset default` 使用原运行方式。这些选项仅适用于推理。

同条件性能与输出对照：

```bash
python scripts/optimize_inference.py \
  --source /home/zk/vggt/vggt \
  --checkpoint /home/zk/vggt/vggt_ckpt/model.pt \
  --frames 8 32 64 128 --warmup 1 --iterations 3 \
  --variants baseline preset_fused --strict \
  --output results/optimization_comparison
```

添加 `--resolution 224` 可测试较低分辨率，添加 `--real-images` 可同时检查官方示例图片。`triton_rope.py` 可独立检查融合位置计算，`make_optimization_report.py` 根据本次保存的成对结果生成优化评估。
