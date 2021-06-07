## Step


### 0. 处理数据

* 过滤无分类标签的数据

    `src_0_deal_data/deal_data_1_for_label_info.py`

* 处理成pkl文件

    `src_0_deal_data/deal_data_2_for_image_info.py`


### 1. Person 2 分类

* 噪声标签 + 类别不平衡 + Self-Training

    `src_1_multi_label/main_0_person.py`

* 分类模型预测图像是否包含人

    `src_1_multi_label/main_0_person.py`

* 处理成pkl文件

    `src_0_deal_data/deal_data_6_for_change_person.py`


### 2. CONTA 多标签分类 + IRNet

* 类别不平衡 + 多标签分类

* CAM + 多尺度

* IR Label

* Ins Seg

* Sem Seg

* Sem Seg 后处理: 过滤不合格伪标签

    `0_filter_mask.py`


### 3. 语义分割模型

* 全部训练

    `src_1_multi_label/main_5_imagenet.py`

    `src_1_multi_label/main_5_imagenet_v3.py`

