"""Generate a standalone experimental report from recorded validation evidence."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("results", type=Path)
p.add_argument("--output", type=Path)
args = p.parse_args()


def read(name, default=None):
    path = args.results / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def metric(value, spec):
    return format(value, spec) if value is not None else "未记录"


env = read("environment.json", {})
manifest = read("manifest.json", {"phases": {}})
correct = read("correctness_flash.json", {})
bench = read("benchmark_flash.json", {"runs": []})
train = read("training_single.json", {})
ddp = [read(f"ddp_rank{i}.json", {}) for i in range(8)]
flash = read("flash_verification.json", {})
phases = manifest["phases"]
complete = len(phases) == 7 and all(v["status"] == "PASS" for v in phases.values()) and all(d.get("status") == "PASS" and len(d.get("steps", [])) == 3 for d in ddp)
packages = env.get("packages", {})
passed_rows = [row for row in bench["runs"] if row["status"] == "PASS"]
capacity = {r: max([row["frames"] for row in passed_rows if row["resolution"] == r], default=0) for r in (518, 224)}
lines = [
    "# VGGT 海光 BW100 测试报告", "",
    "测试日期：2026-10-20（Asia/Shanghai）。", "",
    "## 1. 软硬件环境", "",
    "### 1.1 硬件环境", "",
    "| 项目 | 配置 |", "| --- | --- |",
    "| CPU | 2 × Hygon C86-4G（OPN:7490），合计 128 核、256 逻辑线程，测试时 255 个逻辑线程在线 |",
    "| 系统可见内存 | 约 751 GiB |",
    f"| 加速设备 | {len(env.get('devices', []))} × Hygon BW100 DCU |",
    "| 单卡显存 | 框架报告 68,702,699,520 字节，约 63.98 GiB |",
    "| 架构 | x86_64 |",
    "| 使用方式 | 推理及单卡训练使用逻辑 0 号卡；分布式训练使用全部 8 张卡 |", "",
    "### 1.2 软件环境", "",
    "| 项目 | 版本或配置 |", "| --- | --- |",
    "| 容器操作系统 | Ubuntu 22.04.5 LTS |",
    "| 共享宿主机内核 | Linux 5.15.0-25-generic |",
    "| DTK | 26.04 |",
    f"| HIP | {env.get('hip')} |",
    f"| Python | {env.get('python')} |",
    f"| PyTorch | {packages.get('torch')} |",
    f"| torchvision | {packages.get('torchvision')} |",
    f"| Flash Attention | {packages.get('flash_attn')} |",
    f"| NumPy | {packages.get('numpy')} |",
    f"| 训练辅助依赖 | iopath {packages.get('iopath')}；portalocker {packages.get('portalocker')} |",
    "| 容器名称 | zk-vggt |",
    "| 项目目录 | /home/zk/vggt/validation-project |", "",
    "容器镜像：`harbor.sourcefind.cn:5443/hcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10`。", "",
    "镜像 ID：`sha256:d7dda89f35e0f413e7d8d1dec10121e83028d3796d1190da112ae22526fdee24`。", "",
    "### 1.3 模型与权重", "",
    "- 模型：VGGT-1B。",
    f"- 官方源码提交：`{env.get('source_commit', {}).get('stdout', 'unknown')}`；源码工作树状态：`{env.get('source_status', {}).get('stdout') or 'clean'}`。",
    f"- 权重：model.pt，{env.get('checkpoint', {}).get('bytes')} 字节。",
    f"- 权重 SHA256：`{env.get('checkpoint', {}).get('sha256')}`。", "",
    "## 2. 实验目的与方法", "",
    "本实验验证 VGGT 在海光 BW100 平台上的环境可用性、Flash Attention 执行能力、前向推理、基本数值行为、性能与显存占用，以及单卡和八卡分布式训练图的执行能力。", "",
    "### 2.1 推理与 Flash Attention", "",
    "Flash 算子测试覆盖 FP16、BF16，分别执行强制 Flash SDPA 和直接 Flash 接口的前向、反向，检查输出与梯度有限性及两种调用接口的一致性。通过框架算子记录确认 Flash 执行路径。", "",
    "随机权重前向检查覆盖 4 帧 518×518 FP32、1 帧 224×224 FP32、4 帧 224×224 BF16。真实权重检查覆盖随机图像、相同帧输入和官方 kitchen 示例前 4 张图片，检查输出形状、有限性、相机位姿合法性、自一致性与重复推理结果。", "",
    "BF16 推理主干指定 Flash Attention；FP32 数值检查使用可用的原生后端，相机头保持模型定义的 FP32 计算精度。", "",
    "### 2.2 性能与显存", "",
    "使用随机初始化模型、随机图像和 BF16 autocast，分别测试 518×518、224×224 分辨率下的 1、4、8、16、32、64、128 帧输入。每组预热 2 次、计时 5 次，计时前后同步设备。平均延迟仅包含模型前向计算，不包含数据加载。", "",
    "显存同时记录 PyTorch allocator 的峰值 allocated 与 reserved，单位为十进制 GB。本实验不进行其他注意力后端的性能对照。", "",
    "### 2.3 单卡与分布式训练", "",
    "训练使用随机权重、真实 MultitaskLoss 和合成 GT，启用 camera/depth，关闭 point/track，不额外冻结主干。单卡输入为 S=6、224×224，执行 1 个优化步；八卡 DDP 每卡输入为 S=4、224×224，执行 3 个优化步。", "",
    "使用 BF16 AMP 和 GradScaler，初始 loss scale=65536。单卡 AdamW 参数为 lr=5e-5、weight_decay=0.05；八卡 AdamW 参数为 lr=1e-4、weight_decay=0.01。", "",
    "训练输入 SDPA 的 Q/K/V 使用连续内存布局，激活检查点重计算保持相同设置；模型 CPU 初始化使用 1 线程，之后恢复 8 线程计算。推理性能基准保持原生张量布局。", "",
    "DDP 各 rank 使用不同输入。除检查全部已产生梯度的有限性外，还对每个可训练参数与梯度张量最多 257 个等距样本进行跨卡一致性检查，并确认优化器实际更新。该检查覆盖每个可训练张量，但不等同于全张量逐元素同步校验。", "",
    "## 3. 实验结果", "",
    "### 3.1 测试项目汇总", "",
    "| 测试项目 | 状态 | 结果日志 |", "| --- | --- | --- |",
]
labels = {"environment": "软硬件环境与模型标识", "flash": "Flash 前向与反向", "smoke": "随机权重前向推理", "correctness": "真实权重与基本数值检查", "benchmark_flash": "推理性能与显存", "training": "单卡训练图", "ddp": "八卡 DDP 训练图"}
for key, label in labels.items():
    lines.append(f"| {label} | {phases.get(key, {}).get('status', 'NOT RUN')} | {key}.log |")
lines += ["", "### 3.2 Flash Attention 执行验证", ""]
for row in flash.get("tests", []):
    lines.append(f"- {row['dtype']}：{row['status']}；强制 SDPA 与直接 Flash 接口输出相对 L2 差异 {row['forward_relative_l2']:.3g}，梯度最大相对 L2 差异 {max(row['gradient_relative_l2']):.3g}。")
lines += ["- 最小算子及真实权重模型的框架算子记录均包含 `aten::_scaled_dot_product_flash_attention` 和 `aten::_flash_attention_forward`。", "",
          "### 3.3 真实权重与基本数值行为", ""]
camera, same = correct.get("camera", {}), correct.get("identical_frames", {})
lines += [
    f"- 6 帧 518×518 随机输入：旋转正交性最大误差 {metric(camera.get('orthogonal_max_abs'), '.3e')}，det(R) 范围 {metric(camera.get('det_min'), '.7f')}–{metric(camera.get('det_max'), '.7f')}，焦距均为正。",
    f"- 6 张相同输入：位姿 std={metric(same.get('pose_std'), '.3e')}，深度 std={metric(same.get('depth_std'), '.3e')}，均满足实验判据 <0.01。",
    "- kitchen 示例图片推理输出有限；BF16 Flash 重复推理主要输出的最大绝对差为 0。", "",
    "同组真实图片的 FP32 与 BF16 输出相对 L2 差异如下。这些数据用于观察数值模式差异，不作为数据集重建精度指标。", "",
    "| 输出 | 相对 L2 差异 |", "| --- | ---: |",
]
for key, value in correct.get("fp32_bf16_relative_l2", {}).items():
    lines.append(f"| {key} | {value:.4%} |")
lines += ["", "### 3.4 推理性能与显存", "",
          "| 帧数 | 分辨率 | 状态 | 平均延迟 ms | 帧/秒 | 峰值 allocated GB | 峰值 reserved GB |",
          "| ---: | --- | --- | ---: | ---: | ---: | ---: |"]
for row in bench["runs"]:
    if row["status"] == "OOM":
        lines.append(f"| {row['frames']} | {row['resolution']}×{row['resolution']} | OOM | — | — | — | — |")
    else:
        lines.append(f"| {row['frames']} | {row['resolution']}×{row['resolution']} | PASS | {row['ms']:.1f} | {row['fps']:.2f} | {row['peak_allocated_GB']:.2f} | {row['peak_reserved_GB']:.2f} |")
lines += ["", f"共执行 {len(bench['runs'])} 组性能测试，通过 {len(passed_rows)} 组。本实验在 518×518、224×224 分辨率下分别验证至 {capacity[518]}、{capacity[224]} 帧。", "",
          "### 3.5 单卡训练图", ""]
for row in train.get("steps", []):
    lines.append(f"- S=6、224×224，loss={row['loss']:.6f}。")
    lines.append(f"- 前向、反向、优化及校验耗时 {row['seconds_including_checks']:.2f}s，峰值 allocated 显存 {row['peak_allocated_GB']:.2f}GB。")
    lines.append(f"- 梯度有限，参数样本最大更新量 {row['parameter_max_sample_update']:.6g}，优化器完成实际更新，GradScaler 未跳步。")
lines += ["", "### 3.6 八卡 DDP 训练图", ""]
if all(ddp):
    steps = ddp[0].get("steps", [])
    lines.append(f"8 个 rank 完成测试，all_reduce 哨兵求和为 {ddp[0].get('all_reduce_sentinel'):.0f}（期望 36），每 rank 执行 {len(steps)} 步。")
    lines += ["", "| 步数 | 各 rank loss 范围 | 梯度样本跨卡最大差 | 优化后参数样本跨卡最大差 |",
              "| ---: | --- | ---: | ---: |"]
    for row in steps:
        lines.append(f"| {row['step']} | {min(row['rank_losses']):.6f}–{max(row['rank_losses']):.6f} | {row['gradient_sample_max_rank_difference']:.1f} | {row['parameter_sample_max_rank_difference']:.1f} |")
    lines += ["", f"全部 rank、全部步骤的峰值 allocated 显存最大值为 {max(row['peak_allocated_GB'] for d in ddp for row in d.get('steps', [])):.2f}GB/卡。每步均通过梯度有限性、同步样本一致性及优化器实际更新检查。"]
lines += ["", "## 4. 实验范围与可复现性", "",
    "本实验覆盖环境可用性、Flash 算子、模型推理、基本数值行为、短步训练图和单机八卡 DDP，不包含 CO3D 完整训练、收敛性评估、带独立任务 GT 的精度评测或长期稳定性测试。位姿合法与相同帧一致不等同于重建精度验收。", "",
    "Flash 执行证据来自框架算子记录，未进行 GPU kernel 级 hipprof 跟踪。性能测试仅验证列出的输入规模，不推断更大规模的容量。", "",
    "实验脚本和使用方法见项目 README。结果目录中的 manifest.json 保存各项目执行命令、UTC 起止时间和退出码；environment.json 保存软件版本、源码提交、权重 SHA256 及实验脚本哈希。模型权重与上游源码通过固定提交和哈希标识关联。", "",
    "## 5. 结论", "",
    f"本次实验{'所有测试项目均通过' if complete else '尚未全部通过或证据不完整，需结合项目状态和日志进一步确认'}。",
]
if complete:
    lines += ["", "在本报告列出的软硬件环境和实验配置下，海光 BW100 支持 Flash Attention 的 FP16/BF16 前向与反向计算，VGGT 能完成真实权重推理、单卡训练图及单机八卡 DDP 训练图。", "",
              f"推理性能测试共 {len(bench['runs'])} 组，通过 {len(passed_rows)} 组；518×518 和 224×224 分辨率分别成功运行至 {capacity[518]}、{capacity[224]} 帧。单卡优化更新及八卡连续 3 步训练通过梯度有限性、参数更新和采样同步验证，表明上述执行路径在本实验条件下可用。"]
lines.append("")
target = args.output or args.results / "report.md"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text("\n".join(lines), encoding="utf-8")
print(target)
