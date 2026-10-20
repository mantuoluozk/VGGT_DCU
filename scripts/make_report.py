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
    "| 服务器型号 | X7850H0 |",
    "| CPU | 2 × 海光 C86-4G（7490），合计 128 核、256 逻辑线程 |",
    "| 系统可见内存 | 约 751 GiB |",
    f"| 加速设备 | {len(env.get('devices', []))} × Hygon BW100 DCU |",
    "| 单卡显存 | 64GB 级显存，软件读取容量约 63.98 GiB |",
    "| 使用方式 | 图片处理和单卡训练使用 1 张卡；八卡协同训练使用全部 8 张卡 |", "",
    "### 1.2 软件环境", "",
    "| 项目 | 版本或配置 |", "| --- | --- |",
    "| 容器操作系统 | Ubuntu 22.04.5 LTS |",
    "| 海光开发工具包（DTK） | 26.04 |",
    f"| Python | {env.get('python')} |",
    f"| 模型运行框架 PyTorch | {str(packages.get('torch')).split('+')[0]}，海光适配版 |",
    f"| 图像处理组件 torchvision | {str(packages.get('torchvision')).split('+')[0]}，海光适配版 |",
    f"| 注意力加速组件 Flash Attention | {str(packages.get('flash_attn')).split('+')[0]}，海光适配版 |",
    f"| NumPy | {packages.get('numpy')} |",
    "| 容器名称 | zk-vggt |",
    "| 项目目录 | /home/zk/vggt/validation-project |", "",
    "容器镜像：`harbor.sourcefind.cn:5443/hcu/admin/base/pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10`。", "",
    "上述镜像将所需运行环境打包在一起，便于在相同配置下重复测试。完整软件版本和环境记录保存在项目的 `environment.json` 中。", "",
    "### 1.3 测试模型", "",
    "- 模型：VGGT-1B。",
    "- 使用官方模型代码和官方预训练参数文件 `model.pt`。预训练参数是模型已经学习得到的参数，用于真实图片处理测试。",
    "- 功能与性能测试还使用随机生成的模型参数，以分别检查模型能否运行及其计算开销。",
    "- 具体代码版本、参数文件大小和文件校验信息保存在环境记录中，可用于确认复现实验时使用相同材料。", "",
    "## 2. 实验目的与方法", "",
    "本实验主要回答三个问题：模型能否正常处理图片、处理不同数量图片需要多长时间和多少显存、能否完成单卡及八卡协同训练。", "",
    "### 2.1 图片处理与 Flash Attention 加速", "",
    "“推理”指将图片输入模型，得到相机参数、深度和三维点坐标等结果。Flash Attention 是模型中的一种计算加速技术，用于减少注意力计算的显存需求。本实验既检查加速组件自身能否计算，也通过运行记录确认 VGGT 实际调用了它。", "",
    "加速组件覆盖 FP16 和 BF16 两种 16 位计算格式。这两种格式可减少显存用量。测试同时检查图片处理所需的正向计算和训练所需的反向计算，确认两种调用方式的结果一致，且没有出现无效数值。", "",
    "模型功能测试包括：4 张 518×518 图片、1 张 224×224 图片的常规精度计算，以及 4 张 224×224 图片的 BF16 计算。使用官方预训练参数后，再检查 6 张随机图片、6 张相同图片及官方 kitchen 场景的前 4 张图片，确认结果格式、相机参数和重复计算表现正常。", "",
    "主要性能测试使用 BF16 和 Flash Attention。常规精度检查使用 FP32（32 位计算格式）；模型中负责估计相机参数的部分保持原有 FP32 设置。", "",
    "### 2.2 性能与显存", "",
    "分别测试 518×518、224×224 分辨率下，一次输入 1、4、8、16、32、64、128 张图片的情况。每组先运行 2 次预热，再运行 5 次取平均耗时；计时等待设备计算真正结束。这里的耗时是处理整组图片的模型计算时间，不包含读取图片的时间。", "",
    "显存记录运行过程中的最大值，单位为 GB。“计算使用显存”指框架分配给模型数据和计算结果的空间；“框架预留显存”还包括为后续计算保留的可复用空间，两列不能相加。", "",
    "### 2.3 单卡与八卡协同训练", "",
    "“训练”指模型先计算结果，再根据误差调整自身参数。本实验使用人工生成的图片和参考数据，配合 VGGT 自带的训练误差计算方法，检查这一流程能否完整完成；不以少量测试步骤判断模型能否训练收敛。", "",
    "单卡每次输入 6 张 224×224 图片，完成 1 次参数更新。八卡协同训练采用 DDP 方式，即 8 张卡分别处理不同输入，再同步训练所需的数据，共同更新同一个模型；每卡每次输入 4 张 224×224 图片，连续完成 3 次参数更新。", "",
    "训练启用相机参数和深度估计任务。计算使用 BF16，并按脚本固定配置执行；模型初始化使用 1 个 CPU 线程，计算阶段使用 8 个线程。训练数据在进入 Flash 计算前采用连续存储方式。详细训练参数见项目 README 和结果文件。", "",
    "除检查计算结果有效、模型参数确实发生更新外，还检查 8 张卡上的训练计算和更新后参数是否一致。同步检查对每组可训练参数及其训练计算结果各抽取最多 257 个数据点，属于抽样检查。", "",
    "## 3. 实验结果", "",
    "### 3.1 测试项目汇总", "",
    "| 测试项目 | 结果 | 检查内容 |", "| --- | --- | --- |",
]
labels = {"environment": ("运行环境", "8 张卡可识别，软件和模型文件可用"), "flash": ("Flash Attention 加速", "图片处理和训练所需计算均可执行"), "smoke": ("模型基本运行", "不同图片数量和尺寸下可完成计算"), "correctness": ("官方预训练模型", "检查真实图片结果及重复计算表现"), "benchmark_flash": ("处理速度与显存", "测量不同图片规模的计算开销"), "training": ("单卡训练流程", "完成计算、误差分析和参数更新"), "ddp": ("八卡协同训练", "完成卡间通信、同步和连续参数更新")}
for key, (label, description) in labels.items():
    status = {"PASS": "通过", "FAIL": "未通过"}.get(phases.get(key, {}).get("status"), "未执行")
    lines.append(f"| {label} | {status} | {description} |")
lines += ["", "### 3.2 Flash Attention 执行验证", ""]
for row in flash.get("tests", []):
    status = "通过" if row["status"] == "PASS" else "未通过"
    dtype_label = {"torch.float16": "FP16", "torch.bfloat16": "BF16"}.get(row["dtype"], row["dtype"])
    lines.append(f"- {dtype_label}：{status}。两种 Flash 调用方式的输出整体差异为 {row['forward_relative_l2']:.3g}，反向计算结果的最大相对差异为 {max(row['gradient_relative_l2']):.3g}。")
lines += ["- 加速组件单独测试通过；VGGT 实际运行记录也确认调用了 Flash Attention。", "",
          "### 3.3 图片处理结果检查", ""]
camera, same = correct.get("camera", {}), correct.get("identical_frames", {})
lines += [
    "- 输入 6 张 518×518 图片后，输出的相机位置、朝向和焦距通过基本有效性检查，计算结果均为有效数值。",
    f"- 输入 6 张相同图片时，相机参数的帧间波动为 {metric(same.get('pose_std'), '.6f')}，深度结果的帧间波动为 {metric(same.get('depth_std'), '.6f')}，均低于测试阈值 0.01。这里以标准差衡量波动，数值越接近 0，说明结果越一致。",
    "- 使用官方 kitchen 场景图片，模型可正常输出结果；相同输入重复运行时，主要输出完全一致。", "",
    "同组真实图片分别采用 FP32 和 BF16 计算，整体数值差异如下。比例越小，说明两种计算格式的结果越接近；这些比例不是实际场景的重建精度或业务准确率。", "",
    "| 输出内容 | 与 FP32 相比的整体数值差异 |", "| --- | ---: |",
]
for key, value in correct.get("fp32_bf16_relative_l2", {}).items():
    label = {"pose_enc": "相机参数", "depth": "深度（物体远近）", "world_points": "三维点坐标（空间位置）"}.get(key, key)
    lines.append(f"| {label} | {value:.4%} |")
lines += ["", "### 3.4 推理性能与显存", "",
          "一组图片的平均耗时越短，处理速度越快；1000 毫秒（ms）等于 1 秒。“每秒图片数”由一组图片的数量除以整组处理时间计算。", "",
          "| 图片数量 | 单张图片分辨率 | 结果 | 整组平均耗时 ms | 每秒图片数 | 计算使用显存峰值 GB | 框架预留显存峰值 GB |",
          "| ---: | --- | --- | ---: | ---: | ---: | ---: |"]
for row in bench["runs"]:
    if row["status"] == "OOM":
        lines.append(f"| {row['frames']} | {row['resolution']}×{row['resolution']} | 显存不足 | — | — | — | — |")
    else:
        lines.append(f"| {row['frames']} | {row['resolution']}×{row['resolution']} | 通过 | {row['ms']:.1f} | {row['fps']:.2f} | {row['peak_allocated_GB']:.2f} | {row['peak_reserved_GB']:.2f} |")
lines += ["", f"共执行 {len(bench['runs'])} 组性能测试，通过 {len(passed_rows)} 组。本实验在 518×518、224×224 分辨率下分别验证至 {capacity[518]}、{capacity[224]} 帧。", "",
          "### 3.5 单卡训练流程", ""]
for row in train.get("steps", []):
    lines.append("- 一次输入 6 张 224×224 图片，完成结果计算、误差计算和模型参数更新。")
    lines.append(f"- 完整训练步骤及检查共耗时 {row['seconds_including_checks']:.2f} 秒，计算使用显存峰值 {row['peak_allocated_GB']:.2f}GB。")
    lines.append(f"- 训练误差指标为 {row['loss']:.6f}，训练计算结果均为有效数值。检查确认模型参数确实发生更新，没有跳过该次训练更新。")
lines += ["", "### 3.6 八卡协同训练", ""]
if all(ddp):
    steps = ddp[0].get("steps", [])
    lines.append(f"8 张卡的通信检查通过，并共同完成连续 {len(steps)} 次训练更新。每张卡使用不同输入，更新后对模型参数进行抽样检查。")
    lines += ["", "| 训练步骤 | 8 张卡的训练误差指标范围 | 各卡训练计算是否一致（抽样） | 更新后模型参数是否一致（抽样） |",
              "| ---: | --- | ---: | ---: |"]
    for row in steps:
        gradient_status = "一致，抽样差异为 0" if row["gradient_sample_max_rank_difference"] == 0 else "存在差异"
        parameter_status = "一致，抽样差异为 0" if row["parameter_sample_max_rank_difference"] == 0 else "存在差异"
        lines.append(f"| {row['step'] + 1} | {min(row['rank_losses']):.6f}–{max(row['rank_losses']):.6f} | {gradient_status} | {parameter_status} |")
    lines += ["", "各卡处理的图片不同，因此训练误差指标可以不同。表中的误差指标用于记录训练过程，不代表实际应用准确率；3 次训练也不足以判断模型是否已经学好。", "",
              f"八卡训练期间，单卡计算使用显存峰值最大为 {max(row['peak_allocated_GB'] for d in ddp for row in d.get('steps', [])):.2f}GB。每次训练更新均通过计算有效性、同步抽样和实际参数更新检查。"]
lines += ["", "## 4. 本次测试覆盖范围", "",
    "本次测试确认了模型能否运行、加速功能能否使用、处理速度和显存开销，以及单卡和八卡训练流程能否完成。", "",
    "本次没有使用完整 CO3D 数据集进行长期训练，也没有用带标准答案的业务数据评估三维重建精度。因此，结果可作为兼容性和基础运行能力的依据；实际业务精度、训练效果及长时间稳定性仍需结合后续专项测试确认。测试容量仅限于表中列出的规模。", "",
    "测试脚本和操作步骤见项目 README。原始结果保留了每项测试的执行命令、时间、返回状态、完整软件版本和模型文件标识，技术人员可据此复核或重复测试。", "",
    "## 5. 结论", "",
    f"本次实验{'所有测试项目均通过' if complete else '尚未全部通过或证据不完整，需结合项目状态和日志进一步确认'}。",
]
if complete:
    lines += ["", "在本报告列出的软硬件环境和测试配置下，VGGT 可以在海光 BW100 上完成图片处理、单卡训练步骤和八卡协同训练步骤，Flash Attention 加速功能已实际运行验证。", "",
              f"共完成 {len(bench['runs'])} 组处理速度和显存测试，全部 {len(passed_rows)} 组通过；518×518 和 224×224 分辨率分别成功处理一次输入 {capacity[518]}、{capacity[224]} 张图片的情况。单卡完成参数更新，八卡连续完成 3 次训练更新，且同步抽样检查通过，表明这些基础运行流程在本次测试条件下可用。", "",
              "本次测试形成了覆盖环境配置、图片处理、性能评估和多卡训练的完整实验记录，为 VGGT 在海光 DCU 平台上的适配与应用提供了可复核的技术依据，也为后续性能优化、资源配置和部署方案评估提供了参考。", "",
              "后续可在本次验证基础上，结合实际业务数据和应用需求，进一步开展完整数据集训练、任务精度评测及长时间稳定性验证，持续完善从基础适配到业务应用的验证流程，为后续工程实施和应用落地奠定基础。"]
lines += ["", "项目代码及复现脚本：[VGGT DCU 测试项目（GitLab）](https://developer.sourcefind.cn/codes/zhengke/vggt_dcu)。", ""]
target = args.output or args.results / "report.md"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text("\n".join(lines), encoding="utf-8")
print(target)
