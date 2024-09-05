# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PascalVOCDataset_electricity_bill(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('background', 'row', 'col')

    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0]]

    def __init__(self, split, **kwargs):
        super(PascalVOCDataset_electricity_bill, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split,
            reduce_zero_label=False, # 此时 label 里的 0 是背景（上面 CLASSES 里第一个），所以这里是 False
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
