## Step


### 0. 处理数据

* 过滤无分类标签的数据

* 处理成pkl文件


### 1. Person 2 分类

* 噪声标签 + 类别不平衡 + Self-Training

* 分类模型预测图像是否包含人

* 处理成pkl文件


### 2. CONTA 多标签分类 + IRNet

* 类别不平衡 + 多标签分类

* CAM + 多尺度

* IR Label

* Ins Seg

* Sem Seg


### 3. Sem Seg 后处理

* 过滤不合格伪标签（Sem Seg）


### 4. 语义分割模型

* 全部训练

