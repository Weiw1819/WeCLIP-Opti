# WeCLIP 训练 GPU 利用率优化记录

## 1. 问题现象

训练设备为 NVIDIA RTX A6000（48 GB）。优化前对正在运行的训练进程进行采样，观察到：

- 显存占用约 4.3 GB，远低于 48 GB 上限；
- GPU SM 利用率在 3%～75% 之间频繁波动，平均约为 30%；
- 训练速度约为 2.5 iter/s；
- GPU 利用率呈间歇性峰值，而不是持续处于高负载状态。

这说明主要问题并非显存不足，而是训练过程中存在大量串行小任务、CPU/GPU 同步和数据往返，导致 GPU 经常等待。

## 2. 主要瓶颈

### 2.1 Grad-CAM 按图像、按类别串行执行

原始实现会依次遍历 batch 中的每张图像，再遍历图像包含的每个前景类别。每个类别都会单独运行一次 CLIP 最后一层的 forward/backward。

假设 batch size 为 4，每张图平均包含 2～3 个类别，则每个训练迭代需要执行约 8～12 次小规模 Grad-CAM 任务。单次计算规模较小，CUDA kernel 启动频繁，GPU 很难保持持续满载。

### 2.2 Grad-CAM 激活和梯度被传到 CPU

原实现通过以下路径计算 CAM：

1. 从 GPU 获取目标层激活和梯度；
2. 将完整的多通道张量传到 CPU；
3. 转成 NumPy 计算通道权重及加权和；
4. 将生成的 CAM 再传回 GPU 进行后处理。

目标层通常包含 768 个通道，而最终 CAM 只有 20×20。传输完整激活和梯度会造成明显同步等待。

### 2.3 PAR 按样本串行细化

原实现逐张图像执行 PAR。PAR 内部包含多组不同 dilation 的卷积，并重复传播 20 次。逐样本执行会产生大量小型 CUDA kernel，增加启动开销。

### 2.4 训练循环存在频繁同步

原训练循环每轮多次调用 `Tensor.item()`，包括损失累计和 tqdm 指标更新。读取 GPU 标量会强制 CPU 等待当前 CUDA 流完成，破坏计算流水线。

### 2.5 数据加载和传输未充分异步化

原 DataLoader 未启用锁页内存，CPU 到 GPU 的输入传输也是阻塞的。此外，分类训练数据集会读取分割 PNG，但读取结果随后被丢弃；CAM 阶段还会再次读取相同标签文件。

## 3. 已实施的优化

### 3.1 批量执行 Grad-CAM

涉及文件：

- `clip/clip_tool.py`
- `WeCLIP_model/model_attn_aff_voc.py`
- `clip/model.py`
- `pytorch_grad_cam/base_cam.py`

新增 `perform_batch_voc_cam()`，将 batch 内的“图像×前景类别”组合成批量任务。

由于不同图像包含的前景类别数量不同，候选文本集合的长度也不同。实现中按照前景类别数量进行分组：

- 同一组中的图像具有相同长度的候选文本集合；
- 每个图像-类别对仍使用该图像原本的“有效前景类别＋背景文本”；
- 每组只运行一次 CLIP 最后一层 forward/backward；
- 避免 padding 类别参与 softmax，保持原计算含义。

这样可将每轮约 8～12 次 Grad-CAM 调用减少为约 2～4 次，具体数量取决于 batch 内类别数的分布。

### 3.2 Grad-CAM 通道聚合保留在 GPU

涉及文件：

- `pytorch_grad_cam/activations_and_gradients.py`
- `pytorch_grad_cam/base_cam.py`
- `pytorch_grad_cam/grad_cam.py`

为 Grad-CAM 增加 `keep_on_device` 路径：

- 目标层激活和梯度不再立即复制到 CPU；
- 梯度全局平均、通道加权和、ReLU 和归一化均在 GPU 上执行；
- 只将最终的小尺寸 CAM 传到 CPU，用于现有边界框处理；
- 其他 CAM 算法默认仍使用原来的 CPU 路径，避免影响未使用的算法。

### 3.3 CAM 归一化和上采样改为 GPU 批处理

涉及文件：`clip/clip_tool.py`。

原实现逐类别执行：

```text
GPU CAM → CPU NumPy → OpenCV resize → Torch Tensor → GPU
```

修改后使用 PyTorch 在 GPU 上一次性完成：

```text
stack → min/max 归一化 → F.interpolate
```

避免了逐类别的 GPU/CPU 往返传输。

### 3.4 PAR 改为按 batch 并行

涉及文件：`WeCLIP_model/model_attn_aff_voc.py`。

不同图像的有效 CAM 通道数先补零到当前 batch 的最大通道数，然后一次性调用 PAR。补齐通道保持为 0，不影响有效通道传播和最终类别映射。

类别索引通过 `valid_key_batch` 按样本映射回 VOC 类别编号。

### 3.5 减少训练循环中的 CUDA 同步

涉及文件：`scripts/dist_clip_voc.py`。

修改内容包括：

- 损失在 GPU 上累计；
- 仅在 `log_iters` 到达时同步一次损失值到 CPU；
- tqdm 的损失和学习率同样只按日志间隔更新；
- 使用 `optimizer.zero_grad(set_to_none=True)`；
- 避免每轮重复调用 `seg_loss.item()` 和 `attn_loss.item()`。

### 3.6 优化 DataLoader 和输入传输

涉及文件：

- `scripts/dist_clip_voc.py`
- `datasets/voc.py`

DataLoader 修改为：

```python
pin_memory=True
prefetch_factor=4
persistent_workers=True
```

输入和验证标签使用异步传输：

```python
inputs.cuda(non_blocking=True)
labels.cuda(non_blocking=True)
```

分类训练数据集直接读取 JPEG 和图像级类别标签，不再读取随后会被丢弃的分割 PNG。训练 CAM 直接使用 `cls_labels_onehot.npy` 中的类别信息，也不再由主进程重复打开分割标签。

### 3.7 启用 A6000 加速选项

涉及文件：`scripts/dist_clip_voc.py`。

启用了：

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

固定 320×320 裁剪尺寸有利于 cuDNN 复用已选择的高性能算法。RTX A6000 支持 TF32，可加速 FP32 卷积和矩阵乘法。

### 3.8 支持从启动脚本覆盖 batch size

涉及文件：

- `scripts/dist_clip_voc.py`
- `run_voc_pipeline.sh`

新增 `--batch_size` 参数，并支持通过环境变量传入：

```bash
BATCH_SIZE=8 ./run_voc_pipeline.sh
```

当前默认配置仍为 batch size 4，不设置 `BATCH_SIZE` 时不会改变原始有效 batch size。

## 4. 验证结果

已完成以下检查：

### 4.1 语法检查

相关 Python 文件均通过 `python -m py_compile`，Shell 脚本通过 `bash -n`。

### 4.2 batch PAR 一致性

随机输入下，新 batch PAR 与原逐样本 PAR 的最大绝对误差为：

```text
sample 0: 0.0
sample 1: 0.0
```

### 4.3 GPU CAM 上采样一致性

PyTorch GPU bilinear 上采样与原 OpenCV 路径的最大绝对误差为：

```text
1.1920929e-07
```

### 4.4 批量 Grad-CAM 数值对照

使用真实 VOC 样本比较原逐类别实现和批量实现，单个 CAM 的最大绝对误差不超过：

```text
0.0028523
```

该差异来自 FP16 批量矩阵运算顺序变化。

### 4.5 完整训练步骤测试

已使用真实 VOC batch 完成：

```text
模型前向 → CAM/PAR → 分割损失 → 注意力损失 → backward
```

检查结果：

```text
seg 输出：有限值
CAM 标签：有限值
attention 输出：有限值
backward：通过
```

batch size 2 的冒烟测试峰值显存约为 1.45 GB。实际训练进程还会包含 DataLoader、优化器状态和日志组件，因此显存占用会更高。

## 5. 启动方式

修改后的代码只会在重新启动训练进程后生效。已经运行的 Python 进程不会自动加载代码变化。

### 保持原 batch size 4

```bash
./run_voc_pipeline.sh
```

### 使用 batch size 8

```bash
BATCH_SIZE=8 ./run_voc_pipeline.sh
```

### 使用 batch size 16

```bash
BATCH_SIZE=16 ./run_voc_pipeline.sh
```

建议先从 8 开始，根据显存、迭代速度和训练指标决定是否增加到 16。

## 6. 监控方法

查看整体状态：

```bash
watch -n 1 nvidia-smi
```

连续采样 GPU SM、显存带宽和功耗：

```bash
nvidia-smi dmon -s pucvmet -d 1
```

评估优化效果时应同时观察：

- 平均 GPU SM 利用率；
- 每秒处理的图像数量；
- `iter/s`；
- 单次迭代耗时；
- 峰值显存；
- loss 和验证指标是否正常。

GPU 利用率本身不是唯一目标。如果 batch size 增加后利用率提高，但每个训练迭代明显变慢，应优先比较 images/s 和完成相同训练量所需的总时间。

## 7. 注意事项

1. 启用 TF32 和 cuDNN benchmark 后，不再保证不同运行之间逐位完全一致。
2. 批量 Grad-CAM 与原逐类别计算存在约 1e-3 量级的 FP16 数值差异。
3. 增大 batch size 会改变梯度统计和训练随机性，不等同于只做工程加速。
4. 如果需要严格复现原始训练配置，应保持 batch size 4。
5. 修改代码后必须停止旧训练进程并重新启动，新优化才会生效。
6. 每次运行会自动创建新的 `output/expN`，不会覆盖之前实验。
