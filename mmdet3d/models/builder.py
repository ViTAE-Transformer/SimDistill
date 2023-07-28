from mmcv.utils import Registry, build_from_cfg

from mmdet.models.builder import BACKBONES, HEADS, LOSSES, NECKS

FUSIONMODELS = Registry("fusion_models")
#TRANSFORMER = Registry('Transformer')
VTRANSFORMS = Registry("vtransforms")
FUSERS = Registry("fusers")


def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_vtransform(cfg):
    return VTRANSFORMS.build(cfg)


def build_fuser(cfg):
    return FUSERS.build(cfg)


def build_head(cfg):
    print("build here: ", type(cfg))
    return HEADS.build(cfg)


def build_loss(cfg):
    return LOSSES.build(cfg)

#def build_transformer(cfg, default_args=None):
 #   """Builder for Transformer."""
  #  return build_from_cfg(cfg, TRANSFORMER, default_args)


def build_fusion_model(cfg, train_cfg=None, test_cfg=None):
    print("build here for fusion model: ", type(cfg))
    return FUSIONMODELS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
    )


def build_model(cfg, train_cfg=None, test_cfg=None):
    return build_fusion_model(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
