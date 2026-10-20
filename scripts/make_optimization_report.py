"""Render measured inference optimization results without changing original test records."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--results-root", type=Path, default=Path(__file__).resolve().parents[1]/"results")
p.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"reports/VGGT_DCU_推理优化评估_20261020.md")
args = p.parse_args()


def read(directory, filename="optimization_screen.json"):
    path = args.results_root/directory/filename
    return json.loads(path.read_text()) if path.exists() else {}


profiles = read("optimization_20261020_screen")
short = read("optimization_20261020_presets")
large = read("optimization_20261020_large")
small = read("optimization_20261020_224")
kernel = read("optimization_20261020_kernel_v2", "triton_rope_validation.json")
cli = read("optimization_20261020_cli", "correctness_flash_fused.json")
rows = []
for experiment in (short, large, small):
    baseline = {row["input"]: row for row in experiment.get("runs", []) if row.get("variant") == "baseline" and "median_ms" in row}
    for row in experiment.get("runs", []):
        if row.get("variant") == "preset_fused" and "median_ms" in row:
            ref = baseline[row["input"]]
            agreement = row["agreement"]
            rows.append({"resolution": experiment["resolution"], "input": row["input"], "baseline": ref,
                         "optimized": row, "reduction": 1-row["median_ms"]/ref["median_ms"],
                         "max_abs": max(value["max_abs"] for value in agreement["outputs"].values())})

lines = ["# VGGT 海光 BW100 推理优化评估", "", "测试日期：2026-10-20。服务器：X7850H0，8 张 BW100，本轮使用 1 张卡，单卡显存 64GB。", "",
    "软件环境：Ubuntu 22.04.5、Python 3.10.12、DTK 26.04、海光适配版 PyTorch 2.9.0、Flash Attention 2.6.1、Triton 3.3.0。模型为 VGGT-1B，沿用完整测试报告中的镜像、模型代码和官方预训练参数。", "",
    "## 1. 评估目标与方法", "",
    "目标是在保留输入图片数量、分辨率、模型参数和输出内容的前提下，减少模型处理时间。本轮使用同一组官方预训练参数和同一输入，对原实现与优化实现重新进行成对测量。", "",
    "模型主干使用 BF16 和 Flash Attention，输出分支保持原有精度。正式对照先预热 1 次，再计时 3 次，以中位数作为耗时。计时等待设备计算结束，不包含读取图片、结果传回 CPU 和数值检查。首次使用融合计算时的编译时间不计入预热后的耗时。", "",
    "## 2. 主要耗时与优化内容", ""]
for profile in profiles.get("profiles", []):
    if profile["input"] == 32:
        total = profile["total_ms"]
        attention = profile["module_ms"]["global_attention"]
        heads = profile["module_ms"]["depth_head"]+profile["module_ms"]["point_head"]
        lines += [f"518×518、32 张图片时，跨图片注意力及相关处理约占整次计算的 {attention/total:.1%}，深度与三维点输出合计约占 {heads/total:.1%}。这里是模块级设备计时，用于定位开销，不作为单个计算内核的耗时结论。", ""]
lines += ["采用的优化包括：", "",
    "1. 重用固定的位置数据，减少重复生成和反复等待设备返回位置范围。",
    "2. 将位置变换的多次小计算合并为一次，保留原有计算格式和中间舍入行为。",
    "3. 深度与三维点输出按每组 16 张处理。主干仍对全部输入图片联合分析，分组仅用于最终输出计算。", "",
    "同时筛选了额外整理张量布局、直接调用 Flash、提前转换部分参数、不同输出分组大小和卷积自动调优等方案。未观察到稳定收益或收益较小的方案未纳入推荐组合。", "",
    "## 3. 同条件性能对照", "",
    "耗时降低比例按“1－优化耗时÷原实现耗时”计算。显存两列均为计算使用显存的峰值，单位为 GB。", "",
    "| 分辨率/输入 | 图片数量 | 原实现 ms | 优化后 ms | 耗时降低 | 原实现显存 GB | 优化后显存 GB | 输出对照 |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
for row in rows:
    label = "官方实景图片" if row["input"] == "kitchen4" else f"{row['resolution']}×{row['resolution']}"
    frames = 4 if row["input"] == "kitchen4" else row["input"]
    lines.append(f"| {label} | {frames} | {row['baseline']['median_ms']:.2f} | {row['optimized']['median_ms']:.2f} | {row['reduction']:.2%} | {row['baseline']['peak_allocated_GB']:.2f} | {row['optimized']['peak_allocated_GB']:.2f} | {row['optimized']['agreement']['status']} |")
lines += ["", "官方实景图片经原有预处理后为 350×518。逐次计时数据和每个输出的数值差异均保存在对应结果 JSON 中。优化会改变临时数据的保存和输出分组方式，显存与耗时需结合表格一起评估。", "",
    "## 4. 输出检查", "",
    f"- 融合位置计算覆盖 FP16、BF16、FP32，以及连续和转置布局，共 {len(kernel.get('tests', []))} 组独立测试；状态：{kernel.get('status', '未完成')}。通过的用例与原实现逐元素一致。",
    "- 完整模型核对相机参数、深度、三维点、置信结果及相机参数的中间输出，检查字段、形状、有效数值和数值差异。",
    "- 比较使用相对容差 0.002、绝对容差 0.002；这些是数值对照规则，不是业务精度指标。"]
if rows:
    lines += [f"- 上表全部对照中，实际观察到的最大绝对差为 {max(row['max_abs'] for row in rows):.9g}。"]
lines += [f"- 教程中的完整模型检查命令已运行，状态：{cli.get('status', '未完成')}，覆盖 FP32 功能、BF16 Flash、真实图片和重复运行。"]
lines += ["", "本次没有重新训练模型，也没有使用带业务标准答案的数据评估重建精度。优化保持计算含义和输出内容，实测数值差异用于检查执行实现的一致性。", "",
    "## 5. 使用方法", "",
    "容器内、本项目目录执行：", "", "```bash",
    "python scripts/inference.py correctness --preset fused \\",
    "  --source /home/zk/vggt/vggt \\",
    "  --checkpoint /home/zk/vggt/vggt_ckpt/model.pt \\",
    "  --output results/inference_fused", "```", "",
    "`--preset fused` 使用完整优化组合，需要当前镜像中的 Triton；`--preset balanced` 使用位置数据重用和输出分组，不启用自定义融合计算；默认 `--preset default` 保持原运行方式。优化仅用于推理，训练脚本不受影响。", "",
    "重复性能对照示例：", "", "```bash",
    "python scripts/optimize_inference.py \\",
    "  --source /home/zk/vggt/vggt \\",
    "  --checkpoint /home/zk/vggt/vggt_ckpt/model.pt \\",
    "  --frames 8 32 64 128 --warmup 1 --iterations 3 \\",
    "  --variants baseline preset_fused --strict \\",
    "  --output results/optimization_comparison", "```", "",
    "## 6. 结论", ""]
if rows:
    reductions = [row["reduction"] for row in rows]
    passed = all(row["optimized"]["agreement"]["status"] == "PASS" for row in rows)
    lines += [f"本轮已完成 {len(rows)} 组原实现与优化实现的同条件对照，耗时降低幅度为 {min(reductions):.2%}–{max(reductions):.2%}，完整模型输出检查{'全部通过' if passed else '存在未通过项'}。具体收益随分辨率和图片数量变化，应按表中配置采用。", "",
              "这些优化减少了重复计算、等待和临时数据读写，但保留了全部图片之间的联合分析。高分辨率、大图片数量时，跨图片关联的计算仍会快速增长，因此优化后仍可能出现吞吐下降。后续若需要更大幅度提升，应结合实际场景评估关键视角选择、输入规模或多卡任务调度，并另行验证重建效果。"]
else:
    lines += ["性能对照尚未完成，以结果文件状态为准。"]
lines += ["", "项目代码及复现脚本：[VGGT DCU 测试项目（GitLab）](https://developer.sourcefind.cn/codes/zhengke/vggt_dcu)。", ""]
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text("\n".join(lines), encoding="utf-8")
print(args.output)
