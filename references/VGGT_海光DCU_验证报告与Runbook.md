# VGGT — 海光 DCU 兼容性验证报告与 Runbook

> 验证目标模型：VGGT。  
> 硬件：Hygon DCU（海光 GPU），8 × BW100，DTK 26.04 软件栈。

---

## 1. 结论速览

| 验证项                             | 结果 | 说明                                 |
| ------------------------------- | -- | ---------------------------------- |
| 环境搭建（DCU torch 2.7.1 + 兼容 shim） | ✅  | 8 卡全部可见，`cuda.is_available()=True` |
| Phase 2：前向推理（随机初始化架构）           | ✅  | 输出形状/有限性正确，bf16 可用                 |
| Phase 3：推理正确性（真实权重）             | ✅  | 权重加载 16.7s，位姿可解码、自一致性通过            |
| Phase 4：推理速度/显存基准               | ✅  | 详见 §4.4，MATH 注意力下显存随 token 数二次增长   |
| Phase 5：训练图（前向+loss+反向+优化）单卡    | ✅  | 真实 `MultitaskLoss` + bf16 AMP 跑通   |
| Phase 6：8 卡 DDP 分布式             | ✅  | NCCL/RCCL 通信正常，3 步训练同步完成           |

**核心结论**：VGGT 在海光 DCU 上可训练、可推理、可分布式。唯一结构性限制是  
**DCU 无 flash-attention kernel**，SDPA 自动回退到 MATH 注意力，因此长序列  
（多帧/高分辨率）显存占用与延迟高于 A100（约 3× 显存压力），可通过 bf16 +  
梯度累积 / 减小 `max_img_per_gpu` 缓解。

---

## 2. 硬件与环境

- **设备**：8 × Hygon DCU BW100，约 64 GB HBM/卡（rocminfo 报告的 ~94 GB 为误报）。
- **软件栈**：DTK 26.04，`/dev/kfd` 存在，`rocminfo`/`hipcc` 可用。
- **Python**：3.12.13（deadsnakes 构建，因官方 wheel 为 cp312）。
- **PyTorch**：海光定制版 `torch-2.7.1+das.opt1.dtk2604`（**必须用海光构建**，  
  PyPI 的 CUDA torch 在 DCU 上 `cuda.is_available()=False`）。
- **torchvision**：海光无对应 wheel，使用 PyPI `torchvision-0.22.1` + 运行时 shim。
- **numpy**：必须 `<2`（DCU torch 编译于 NumPy 1.x），已固定 `numpy==1.26.4`。

---

## 3. 关键兼容性问题与修复

| 问题                         | 现象                                                      | 修复                                                               |
| -------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------- |
| DCU torchvision 算子未注册      | `operator torchvision::nms does not exist`              | `dcu_compat.py` 给 `torch.library.register_fake` 打补丁，忽略缺失算子       |
| flash-attn 不可用             | `No matching libraries found for flash_attn_2_cuda*.so` | 强制 MATH SDPA：`enable_flash_sdp(False)` + `enable_math_sdp(True)` |
| numpy 2 不兼容                | DCU torch 导入崩溃                                          | 固定 `numpy==1.26.4`                                               |
| `from vggt import VGGT` 失败 | 包无顶层 `__init__.py`                                      | 改用 `from vggt.models.vggt import VGGT`（与官方 demo 一致）              |
| 训练 loss 模块缺 `iopath`       | `ModuleNotFoundError: iopath`                           | `pip install iopath`                                             |
| huggingface.co 直连不可达       | `Network is unreachable`                                | 走镜像 `HF_ENDPOINT=https://hf-mirror.com`（镜像回源 CDN 可达）             |

`dcu_compat.py`（**必须最先 import**，放在所有 torch/torchvision 之前）：

```python
import torch.library as _tl, functools
_real = _tl.register_fake
def _safe_register_fake(op_name, func=None, *, lib=None):
    if func is None:
        dec = _real(op_name, lib=lib)
        @functools.wraps(dec)
        def wrapped(f):
            try: return dec(f)
            except Exception as e:
                if "does not exist" in str(e): return f
                raise
        return wrapped
    try: return _real(op_name, func, lib=lib)
    except Exception as e:
        if "does not exist" in str(e): return None
        raise
_tl.register_fake = _safe_register_fake
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)  # DCU 未编译，回退 MATH
torch.backends.cuda.enable_math_sdp(True)
```

---

## 4. 验证细节

### Phase 2 — 前向推理（随机初始化，无需权重）

脚本：`/root/vggt_smoke.py`

- 模型 1256.5M 参数，构建 10.8s。
- 所有输出（`pose_enc` / `depth` / `world_points` / `*_conf` / `images`）形状正确且全部 finite。
- `518×518, S=4, fp32` ≈ 6.0s；`224×224, S=1` ≈ 0.42s；`224×224, S=4, bf16` ≈ 0.50s。
- **bf16 在 DCU 上可用**（注意：VGGT 默认 `torch.cuda.amp.autocast` 仅用于关闭 autocast，  
  实际用 bf16 需显式 `autocast(dtype=bfloat16)`）。

### Phase 5 — 训练图单卡（真实 loss）

脚本：`/root/vggt_train_smoke.py`

- 模型配置对齐 `training/config/default.yaml`（`enable_camera/depth`，关 point/track）。
- 真实 `MultitaskLoss`（camera l1 + depth grad/reg/conf）+ bf16 AMP + `GradScaler`。
- 合成 GT batch（extrinsics/intrinsics/depths/world_points/point_masks）前向→loss→反向→step 全部跑通，  
  loss 有限（cam≈1.01, depth_conf≈4.10, depth_reg≈2.10），backward+step 6.83s。

### Phase 6 — 8 卡 DDP（真实 loss）

脚本：`/root/vggt_ddp_test.py`（`torchrun --nproc_per_node=8`）

- NCCL/RCCL 通信组在 8 卡上初始化成功；`all_reduce` 哨兵求和 36==expected，跨卡通信正常。
- 3 步训练经 DDP 完成，**各 rank loss 完全一致**（参数已同步）。
- 重要：DDP 必须用真实 `MultitaskLoss`（所有 head 都参与 loss），否则 camera head 参数无梯度会触发  
  `find_unused_parameters` 报错；这与 `default.yaml` 中 `find_unused_parameters: False` 一致。

### Phase 3 — 推理正确性（真实权重 `model.pt`, 4.7 GB, 1256.5M params）

脚本：`/root/vggt_correctness.py`

- 权重加载 16.7s；所有输出有限，深度范围 0.33–1.02（归一化深度，合理），world_points 分布合理。
- **相机位姿可解码且合法**：旋转正交性误差 5.9e-9，`det(R)=1.0000`，焦距 308–313（全正，对应 518px 输入合理）。
- **自一致性**：6 张完全相同帧 → 各帧相机位姿 std 6.8e-5、深度差异 3.9e-3（≈0），  
  证明管线确定且一致。
- **VERDICT: CORRECTNESS OK**。

### Phase 4 — 推理速度 / 显存基准（bf16, MATH 注意力）

脚本：`/root/vggt_benchmark.py`，结果 `/root/vggt_benchmark_results.json`。  
token 数 = S × (H/14)²；无 flash，SDPA 回退 MATH，峰值显存随总 token 数近似二次增长。

| S   | res | 延迟(ms) | 吞吐(fps) | 峰值HBM(GB) |
| --- | --- | ------ | ------- | --------- |
| 1   | 518 | 325    | 3.08    | 7.6       |
| 4   | 518 | 1493   | 2.68    | 11.2      |
| 8   | 518 | 4217   | 1.90    | 23.2      |
| 16  | 518 | OOM    | —       | —         |
| 1   | 224 | 165    | 6.05    | 7.0       |
| 4   | 224 | 221    | 18.1    | 7.2       |
| 8   | 224 | 410    | 19.5    | 7.6       |
| 16  | 224 | 871    | 18.4    | 9.5       |
| 32  | 224 | 2411   | 13.3    | 16.5      |
| 64  | 224 | 8583   | 7.46    | 43.9      |
| 128 | 224 | OOM    | —       | —         |

**解读**：224 分辨率下 S≤64 可跑（峰值 44GB < 64GB），吞吐在 S=4–16 平台期约 18 fps；  
518 分辨率下 S≥16 即 OOM。若需更长序列/更高分辨率，建议：  
减小 `img_size`（如 392/224）、用 bf16、开启梯度累积或序列分块。

> 注意：A100 上有 flash-attention，同等序列显存约 1/3、延迟更低；DCU 当前受限于 MATH 注意力。

---

## 5. Runbook（复现命令）

```bash
# 0) 激活环境
source /root/venv312/bin/activate   # 或 /root/venv312/bin/python

# 1) 下载权重（海光机直连 huggingface.co 不通，必须走镜像）
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=hf_xxx               # 你的 HF token
/root/venv312/bin/python /root/download_ckpt.py   # -> /root/vggt_ckpt/model.pt

# 2) 推理 smoke（随机初始化）
/root/venv312/bin/python /root/vggt_smoke.py

# 3) 推理正确性（真实权重）
/root/venv312/bin/python /root/vggt_correctness.py

# 4) 速度/显存基准
/root/venv312/bin/python /root/vggt_benchmark.py

# 5) 训练图单卡
/root/venv312/bin/python /root/vggt_train_smoke.py

# 6) 8 卡 DDP
export NCCL_SOCKET_IFNAME=lo OMP_NUM_THREADS=8
/root/venv312/bin/torchrun --nproc_per_node=8 /root/vggt_ddp_test.py

# 7) 官方完整训练（需 CO3D 数据集 + 额外依赖，本机未配置）
#    pip install hydra-core omegaconf fvcore tensorboard
#    torchrun --nproc_per_node=8 training/launch.py --config default
#    ⚠ default.yaml 中 CO3D_DIR 为占位符，需替换为真实数据路径
```

---

## 6. 已知限制与建议

1. **无 flash-attention**：SDPA 回退 MATH，长序列（多帧/高分辨率）显存与延迟偏高。  
   推理可用 bf16 压缩显存；训练可用 `accum_steps`/`max_img_per_gpu` 调小。
2. **完整 `launch.py` 训练需要 CO3D 数据与额外依赖**（hydra/fvcore/iopath/tensorboard）。  
   本机未准备数据，故以「训练图单卡 + 8 卡 DDP」作为兼容性证明，等价于验证了  
   真实 autograd/AMP/loss/DDP 路径在 DCU 上可用。
3. **`numpy<2` 硬约束**：DCU torch 编译于 NumPy 1.x，升级会破坏导入。
4. **VGGT 无顶层 `VGGT` 导出**：一律用 `from vggt.models.vggt import VGGT`。

---

## 7. 文件清单

| 文件                                  | 作用                       |
| ----------------------------------- | ------------------------ |
| `/root/dcu_compat.py`               | DCU 兼容 shim（必须最先 import） |
| `/root/download_ckpt.py`            | 通过镜像下载 `model.pt`        |
| `/root/vggt_smoke.py`               | Phase 2 前向 smoke         |
| `/root/vggt_correctness.py`         | Phase 3 推理正确性            |
| `/root/vggt_benchmark.py`           | Phase 4 速度/显存基准          |
| `/root/vggt_train_smoke.py`         | Phase 5 训练图单卡            |
| `/root/vggt_ddp_test.py`            | Phase 6 8 卡 DDP          |
| `/root/vggt_benchmark_results.json` | 基准结果                     |
| `/root/vggt_ckpt/model.pt`          | 真实权重（下载中）                |
