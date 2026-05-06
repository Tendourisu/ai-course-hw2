# STL10-project

本项目使用 `uv` 管理 Python 环境，基于 PyTorch 完成 STL-10 十分类任务。代码包含数据划分、CNN/ResNet-18 模型训练、优化对比实验、最终测试集评估、训练曲线、classification report、confusion matrix 和 Grad-CAM 可解释性可视化。

## 环境配置

```bash
uv sync
```


## 数据结构

默认数据路径为 `STL10`，需要保持如下结构：

```text
STL10/
  train/
    airplane/
    bird/
    ...
  test/
    airplane/
    bird/
    ...
```

训练过程中会从 `STL10/train` 按类别分层划分训练集和验证集，默认 `valid_ratio=0.15`。`STL10/test` 只在模型训练结束后用于最终评估。

## 单次训练

运行一个 ResNet-18 主实验：

```bash
uv run python -m stl10_project.train \
  --model resnet18 \
  --data-root STL10 \
  --epochs 40 \
  --batch-size 64 \
  --optimizer adamw \
  --lr 0.001 \
  --weight-decay 0.0005 \
  --augmentation basic \
  --dropout 0.2 \
  --device cuda:0 \
  --run-name main_resnet18
```

运行一个更轻量的自定义 CNN baseline：

```bash
uv run python -m stl10_project.train \
  --model simple_cnn \
  --data-root STL10 \
  --epochs 30 \
  --batch-size 64 \
  --optimizer sgd \
  --lr 0.03 \
  --augmentation none \
  --no-batchnorm \
  --device cuda:0 \
  --run-name baseline_simple_cnn
```

输出会保存在 `outputs/<run-name>/`，包括：

- `best.pt`：验证集 Accuracy 最优的模型权重。
- `config.json`：本次实验配置。
- `history.csv` 和 `history.json`：每个 epoch 的训练/验证指标。
- `learning_curves.png`：Loss 和 Accuracy 曲线。
- `test_report.txt` / `test_report.json`：最终测试集 Precision、Recall、F1-score。
- `test_confusion_matrix.png`：最终测试集 confusion matrix。
- `gradcam/`：Grad-CAM 可视化图。

## ablation 实验设计

下面脚本会依次运行四组实验：

1. `baseline_simple_cnn`：基础 CNN，无数据增强，SGD。
2. `aug_simple_cnn`：加入 RandomCrop / HorizontalFlip。
3. `regularized_simple_cnn`：加入 BatchNorm 和 Dropout。
4. `resnet18_adamw`：使用 STL-10 适配版 ResNet-18，并更换 optimizer 为 AdamW。

```bash
bash run_experiments.sh --data-root STL10 --epochs 30 --device cuda:0
```

脚本会生成 `outputs/experiment_summary.csv`，用于报告中的对比表格。

## Grad-CAM

训练脚本默认会在最终测试集上生成若干张 Grad-CAM。如果想要仅生成可视化：

```bash
uv run python -m stl10_project.gradcam \
  --checkpoint outputs/main_resnet18/best.pt \
  --data-root STL10 \
  --model resnet18 \
  --device cuda:0 \
  --output-dir outputs/main_resnet18/gradcam
```

## 项目文件

```text
stl10_project/
  data.py             # ImageFolder 数据集、stratified train/valid split、transforms
  engine.py           # train/evaluate/predict loops
  gradcam.py          # Grad-CAM 可视化
  metrics.py          # classification report、confusion matrix、曲线绘制
  models.py           # simple CNN 和 STL-10 适配版 ResNet-18
  train.py            # 主训练入口
  utils.py            # seed、JSON/CSV、输出目录等工具
run_experiments.sh    # 多组优化对比实验
report.md             # 实验报告模板
```

## AI 工具使用说明

本项目代码由 OpenAI Codex(GPT 5.5) 辅助生成,主要在数据处理和训练pipeline管线搭建阶段使用了AI工具。
