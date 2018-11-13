# 迁移学习实验

### 实验目的

1. 验证迁移学习对识别效果的提升
2. 验证样本污染对迁移学习的影响

### 实验设计

> Source dataset

    mnist 数据集 : 共 6w 个样本，10 个标签

> Target dataset

    usps 数据集 : 分为 training 和 test 数据集，10 个标签

- training

  1000 个样本 (每个类别各有100个样本)

- test

  1000 个样本 (每个类别各有100个样本)

> 实验步骤

1. 在 target training 数据集上训练模型A
2. 在 source + target training 数据集上训练模型B
3. 先在 source 上训练模型C'，再将C'迁移到 target training 上进行 fine-tuning 得到模型C
4. 比较模型 A,B,C 在 target test 数据集上的效果
5. 给 source 数据集加入 attack，重复 1-4

> 评估标准

    模型在test集上的准确率，另外也考虑：模型训练时间、轮次

> 实验结果

1. epoch = 10000, learn_rate = 0.0001

> exp result

| l_rate | mnist | noise/poison(%) | usps train | usps test | normal | one-step | two-step |
| ------ | ----- | --------------- | ---------- | --------- | ------ | -------- | -------- |
| 1e-4   | 60000 | 0               | 500        | 800       | 0.791  | 0.884    | 0.924    |
| 1e-4   | 60000 | **20**          | 500        | 800       | 0.791  | 0.806    | 0.890    |

    (说明：one-step是将teacher+student数据集一起训练的方式，two-step是先在teacher上训练得到模型后再在student微调的方式)

> Analysis

分析主要针对攻击效果差的现象

1. 因为采用的是深度模型，label flipping 不足够影响**模型从数据中提取到特征**
2. label flipping 的时候是随机修改的标签，可能不够有针对性地修改的影响大

    <font color=#a8a8a8>(说明：修改一些样本标签的时候，是随机的，比如将标签2随机修改，那么结果可能是改成3、4、5都有可能)</font>

3. 在 teacher 数据集加入噪声后，训练过程中模型一直**无法收敛**，猜测如果模型在训练中可以收敛那么在测试阶段效果应该会比较差，无法收敛的原因可能跟噪声样本占比有关
