# STL-10 实验报告

## 1. 网络结构设计

我在本项目主要实现了两类模型：

- `simple_cnn`：自定义的轻量 CNN，包含 4 个 convolution block，每个 block 由 Conv2d、 BatchNorm(可选)、activation 和 pooling 组成，最后使用 global average pooling 与 fully connected classifier 输出 10 类预测。
- `resnet18`：基于 Kaiming He 的 ResNet-18 的结构，针对 96x96 图像将首层改为 `3x3 stride=1` convolution，并移除原始 ImageNet ResNet 中过早下采样的 max pooling，使模型更适合 STL-10 小尺寸图像。

## 2. 实验设置

评价指标包括 Accuracy、Precision、Recall、F1-score 和 confusion matrix。训练过程中记录训练集与验证集的 Loss 和 Accuracy，并绘制曲线观察收敛过程与过拟合情况。所有实验均训练 30 个 epoch，batch size 为 64。

本实验围绕课程要求的三个优化角度设计对比：

| 实验名 | 模型 | 数据增强 | 正则化/归一化 | optimizer | lr |
| --- | --- | --- | --- | --- | ---: |
| baseline_simple_cnn | simple_cnn | none | 无 BatchNorm / Dropout | SGD | 0.03 |
| aug_simple_cnn | simple_cnn | RandomCrop、HorizontalFlip | 无 BatchNorm / Dropout | SGD | 0.03 |
| regularized_simple_cnn | simple_cnn | RandomCrop、HorizontalFlip | BatchNorm、Dropout=0.3 | AdamW | 0.001 |
| resnet18_adamw | resnet18 | RandomCrop、HorizontalFlip | BatchNorm、Dropout=0.3 | AdamW | 0.001 |

其中，`aug_simple_cnn` 用于验证数据增强效果，`regularized_simple_cnn` 用于验证 BatchNorm、Dropout 和 optimizer 调整的效果，`resnet18_adamw` 用于比较更加现代的模型结构带来的变化。

## 3. 实验结果

| 实验名 | Best Epoch | Best Valid Acc | Test Acc | Test Macro F1 | Test Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_simple_cnn | 29 | 60.29% | 61.60% | 0.6031 | 0.6031 |
| aug_simple_cnn | 28 | 66.29% | 67.10% | 0.6751 | 0.6751 |
| regularized_simple_cnn | 30 | 76.29% | 75.90% | 0.7544 | 0.7544 |
| resnet18_adamw | 26 | 74.57% | 75.00% | 0.7504 | 0.7504 |

从结果看，数据增强使测试 Accuracy 从 61.60% 提升到 67.10%；继续引入 BatchNorm、Dropout，并将 optimizer 从 SGD 换为 AdamW 后，测试 Accuracy 提升到 75.90%。这说明数据增强和正则化对于模型的泛化能力都有提升。
ResNet-18 的测试 Accuracy 为 75.00%，略低于正则化后的轻量 CNN，这说明在本实验的数据规模和训练轮数下，更复杂的结构并不一定直接带来更好的最终泛化效果。

四个实验的训练曲线如下：

![baseline_simple_cnn learning curves](outputs/baseline_simple_cnn/learning_curves.png)

![aug_simple_cnn learning curves](outputs/aug_simple_cnn/learning_curves.png)

![regularized_simple_cnn learning curves](outputs/regularized_simple_cnn/learning_curves.png)

![resnet18_adamw learning curves](outputs/resnet18_adamw/learning_curves.png)

最佳模型为 `regularized_simple_cnn`，其最终测试集分类报告如下：

| 类别 | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| airplane | 0.9255 | 0.8700 | 0.8969 |
| bird | 0.7143 | 0.8500 | 0.7763 |
| car | 0.7638 | 0.9700 | 0.8546 |
| cat | 0.7013 | 0.5400 | 0.6102 |
| deer | 0.6316 | 0.8400 | 0.7210 |
| dog | 0.6479 | 0.4600 | 0.5380 |
| horse | 0.7248 | 0.7900 | 0.7560 |
| monkey | 0.7386 | 0.6500 | 0.6915 |
| ship | 1.0000 | 0.7700 | 0.8701 |
| truck | 0.8095 | 0.8500 | 0.8293 |

该模型在 `car、airplane、ship、truck` 等轮廓较稳定、背景相对清晰的类别上表现较好；其中 `car` 的 Recall 达到 0.9700，`ship` 的 Precision 达到 1.0000。较难类别主要是 `dog、cat、monkey`，这几类动物在姿态、背景和局部纹理上更容易混淆。根据 `regularized_simple_cnn` 的 confusion matrix，`dog` 容易被误分为 `deer、horse、cat、monkey`，`cat` 也常被误分到 `deer、dog、monkey`，说明动物类别之间的细粒度差异仍是主要错误来源。

最佳模型的 confusion matrix 如下：

![regularized_simple_cnn confusion matrix](outputs/regularized_simple_cnn/test_confusion_matrix.png)

## 4. 过拟合分析

基础模型 `baseline_simple_cnn` 在最后一个 epoch 的训练 Accuracy 达到 91.55%，但验证 Accuracy 只有 60.29%，同时训练 Loss 为 0.2656、验证 Loss 为 1.8612，二者差距很大，说明模型已经明显过拟合。该模型没有使用数据增强、BatchNorm 和 Dropout，因此训练集拟合较快，但泛化能力不足。

加入 RandomCrop 和 HorizontalFlip 后，`aug_simple_cnn` 的最终训练 Accuracy 为 73.76%，验证 Accuracy 为 64.57%，训练与验证差距明显缩小。数据增强增加了输入扰动，使模型不能简单记忆训练图像，因此测试 Accuracy 从 61.60% 提升到 67.10%。

`regularized_simple_cnn` 在第 30 个 epoch 取得最佳验证 Accuracy 76.29%，测试 Accuracy 75.90%，是本次实验中表现最好的模型。它的最终训练 Accuracy 为 87.73%，验证 Accuracy 为 76.29%，仍然存在一定过拟合，但相比 baseline 已明显缓解。BatchNorm 改善了训练稳定性，Dropout 降低了 classifier 对局部特征组合的依赖，AdamW 也使收敛过程更加稳定。

`resnet18_adamw` 的最佳验证 Accuracy 出现在第 26 个 epoch，为 74.57%，测试 Accuracy 为 75.00%。虽然 ResNet-18 具有更强表达能力，但在 30 epoch 设置下并未超过正则化后的轻量 CNN。可能原因是 ResNet-18 参数量更大，对训练轮数、学习率和正则化强度更敏感，同时 STL-10 本次训练集规模相对有限。

## 5. Grad-CAM

我们使用 Grad-CAM 对 `resnet18_adamw` 的预测过程进行可视化。每个类别各取一张测试图像进行可视化。

示例：

![Grad-CAM airplane sample](outputs/resnet18_adamw/gradcam_balanced/gradcam_00_airplane.png)

![Grad-CAM car sample](outputs/resnet18_adamw/gradcam_balanced/gradcam_02_car.png)

从可视化结果看，模型在正确识别 `car` 样本时，热力图主要集中在车身和车轮附近，说明模型确实利用了目标主体区域进行判断。对于一张真实类别为 `airplane` 但被预测为 `ship` 的样本，Grad-CAM 热力图同时覆盖了飞机、水面和背景区域，模型可能受港口、水面等上下文影响，将图像误判为 `ship`。这说明模型不仅依赖目标物体本身，也会受到背景线索影响。

## 6. AI 工具使用说明

本项目的代码框架部分由 OpenAI Codex(GPT 5.5) 辅助生成,主要在数据处理和训练pipeline管线搭建阶段使用了AI工具。