norm_cfg = dict(type='BN', requires_grad=False)
num_classes = 3
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ERFNet',
        in_channels=3,
        enc_downsample_channels=(16, 64, 128),
        enc_stage_non_bottlenecks=(5, 8),
        enc_non_bottleneck_dilations=(2, 4, 8, 16),
        enc_non_bottleneck_channels=(64, 128),
        dec_upsample_channels=(64, 16),
        dec_stages_non_bottleneck=(2, 2),
        dec_non_bottleneck_channels=(64, 16),
        dropout_ratio=0.1,
        init_cfg=None),
    decode_head=dict(
        type='FCNHead',
        in_channels=16,
        channels=128,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=dict(type='BN', requires_grad=False),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0),
            dict(
                type='FocalLoss',
                loss_name='loss_focal',
                use_sigmoid=True,
                loss_weight=5.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    # test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(682, 682))
)
dataset_type = 'PascalVOCDataset_common_table'
data_root = 'data/all'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
img_scale = (2048, 1024)
ratio_range = (0.5, 2.0)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(1024, 1024), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='PascalVOCDataset_common_table',
        data_root='data/all',
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
            dict(
                type='RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(1024, 1024), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='PascalVOCDataset_common_table',
        data_root='data/all',
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='PascalVOCDataset_common_table',
        data_root='data/all',
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    train_dataloader=dict(samples_per_gpu=8, workers_per_gpu=8, shuffle=True),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=4, shuffle=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=4, shuffle=False))
custom_imports = dict(
    imports=['mmseg.datasets.voc_common_table'], allow_failed_imports=False)
log_config = dict(
    interval=10, hooks=[dict(type='TextLoggerHook', by_epoch=True)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'pretrain_models/erfnet_fcn_4x4_512x1024_160k_cityscapes_20211126_082056-03d333ed.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='EpochBasedRunner', max_iters=None, max_epochs=300)
checkpoint_config = dict(by_epoch=True, interval=100)
evaluation = dict(
    interval=1,
    metric=['mIoU', 'mDice', 'mFscore'],
    pre_eval=True,
    save_best='mIoU')
max_epochs = 300
interval = 1
work_dir = './work_dirs/erfnet_fcn_4x4_1024x1024_160k_common_table'
gpu_ids = [0]
auto_resume = False
