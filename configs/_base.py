_base_ = 'mmdet::grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py'

custom_imports = dict(imports=['nllkg'])

dataset_type = 'KeypointGraphDataset'


model = dict(
    type='GroundingPOSE',
    num_queries=900,
    language_model=dict(
        type='MemoizedBertModel',
        use_cache=False,
    ),
    decoder=dict(
        num_layers=4
    ),
    bbox_head=dict(
        type='GroundingPOSEHead',
        relation_branch=dict(
            embed_dims_in=256,
            embed_dims=128,
            relation_decoder=dict(
                num_layers=2,
                layer_cfg=dict(
                    cross_attn_text_cfg=dict(embed_dims=128, num_heads=4, dropout=0.0),
                    cross_attn_cfg=dict(embed_dims=128, num_heads=4, dropout=0.0),
                    ffn_cfg=dict(embed_dims=128, feedforward_channels=512, ffn_drop=0.0))),
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=7.0),
        loss_relation=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5.0
        )
    ),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=7.0, box_format='xywh'),
            ])),
    test_cfg=dict(max_per_img=300)
)


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadKeypointGraphAnnotationsAsBbox'),
    dict(type='Rotate', prob=0.5, level=1, max_mag=30.0),
    dict(type='Brightness', prob=0.5, level=1),
    dict(type='Contrast', prob=0.5, level=1),
    dict(
        type='RandomChoiceResize',
        scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                (736, 1333), (768, 1333), (800, 1333)],
        keep_ratio=True),
    dict(type='ConvertRelationsToMatrix'),
    dict(type='TransformKeypoints'),
    dict(
        type='PackKeypointGraphInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text', 'relation_text',
                   'custom_entities'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadKeypointGraphAnnotationsAsBbox'),
    dict(type='ConvertRelationsToMatrix'),
    dict(
        type='PackKeypointGraphInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'relation_text', 'custom_entities',
                   'crowd_index', 'raw_ann_info'))
]

num_gpus = 1
batch_size_per_gpu = 2
num_workers = 2
gradient_accumulation = 8


auto_scale_lr = dict(base_batch_size=num_gpus * batch_size_per_gpu * gradient_accumulation)


train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/train.json',
        pipeline=train_pipeline,
        data_prefix=dict(img='images/train')))

val_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val')))

test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/test')))


val_evaluator = dict(
    type='KeypointRelationMetric')
test_evaluator = dict(
    type='KeypointRelationMetric')

max_epoch = 200

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=2, save_best='nllkg/summary/overall_score', rule='greater'),
    logger=dict(type='LoggerHook', interval=200))

train_cfg = dict(
    max_epochs=max_epoch, 
    val_interval=10)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[150],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    accumulative_counts=gradient_accumulation,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

visualizer = dict(type='OpenVocPoseVisualizer')