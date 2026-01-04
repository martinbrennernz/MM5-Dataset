import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
# from utils.load_utils import load_pretrain
# from functools import partial
from config import config as C

from engine.logger import get_logger

logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        # import backbone and decoder
        if cfg.backbone == 'swin_s':
            logger.info('Using backbone: Swin-Transformer-small')
            from .encoders.dual_swin import swin_s as backbone
            self.channels = [96, 192, 384, 768]
            self.n_modal = len(C.active_modalities())
            self.backbone = backbone(n_modal = self.n_modal,norm_fuse=norm_layer, alignment_method = cfg.alignment_method, fusion_method = cfg.fusion_method)
        elif cfg.backbone == 'swin_b':
            logger.info('Using backbone: Swin-Transformer-Base')
            from .encoders.dual_swin import swin_b as backbone
            self.channels = [128, 256, 512, 1024]
            self.n_modal = len(C.active_modalities())
            self.backbone = backbone(n_modal = self.n_modal,norm_fuse=norm_layer, alignment_method = cfg.alignment_method, fusion_method = cfg.fusion_method)
        elif cfg.backbone == 'mit_b5':
            logger.info('Using backbone: Segformer-B5')
            from .encoders.dual_segformer import mit_b5 as backbone
            self.n_modal = len(C.active_modalities())
            self.backbone = backbone(n_modal = self.n_modal,norm_fuse=norm_layer, alignment_method = cfg.alignment_method, fusion_method = cfg.fusion_method)
        elif cfg.backbone == 'mit_b4':
            logger.info('Using backbone: Segformer-B4')
            from .encoders.dual_segformer import mit_b4 as backbone
            self.n_modal = len(C.active_modalities())
            self.backbone = backbone(n_modal = self.n_modal,norm_fuse=norm_layer, alignment_method = cfg.alignment_method, fusion_method = cfg.fusion_method)
        elif cfg.backbone == 'mit_b2':
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
        
            # how many extra modalities are active
            self.n_modal = len(C.active_modalities())
            # total channels after concatenating them (1 ch for single, 3 for multi)
            extra_ch = [1 if single else 3 for single in C.x_is_single_channel]
        
            # instantiate with image size, RGB channels, and concatenated extras
            self.backbone = backbone(
                img_size         = C.image_height,      # or C.image_size if you have that
                in_chans         = 3,                   # RGB always has 3 channels
                extra_in_chans   = extra_ch,            # total extra‐modal channels
                n_modal          = self.n_modal,
                norm_fuse        = norm_layer,
                alignment_method = cfg.alignment_method,
                fusion_method    = cfg.fusion_method,
                fusion_combination = cfg.fusion_combination,
                use_intensity_enhancement = cfg.use_intensity_enhancement,
                num_classes= cfg.num_classes
            )

        elif cfg.backbone == 'mit_b1':
            logger.info('Using backbone: Segformer-B1')
            from .encoders.dual_segformer import mit_b1 as backbone
            self.n_modal = len(C.active_modalities())
            
            # total channels after concatenating them (1 ch for single, 3 for multi)
            extra_ch = [1 if single else 3 for single in C.x_is_single_channel]
            
            # instantiate with image size, RGB channels, and concatenated extras
            self.backbone = backbone(
                img_size         = C.image_height,      # or C.image_size if you have that
                in_chans         = 3,                   # RGB always has 3 channels
                extra_in_chans   = extra_ch,            # total extra‐modal channels
                n_modal          = self.n_modal,
                norm_fuse        = norm_layer,
                alignment_method = cfg.alignment_method,
                fusion_method    = cfg.fusion_method,
                use_intensity_enhancement = cfg.use_intensity_enhancement,
                num_classes= cfg.num_classes
            )
        elif cfg.backbone == 'mit_b0':
            logger.info('Using backbone: Segformer-B0')
            self.channels = [32, 64, 160, 256]
            from .encoders.dual_segformer import mit_b0 as backbone
            self.n_modal = len(C.active_modalities())
            extra_ch = [1 if single else 3 for single in C.x_is_single_channel]
            
            # Safeguard for fusion_combination
            fusion_combination = getattr(cfg, 'fusion_combination', 'sigmoid_gating')
        
            self.backbone = backbone(
                img_size         = C.image_height,
                in_chans         = 3,
                extra_in_chans   = extra_ch,
                n_modal          = self.n_modal,
                norm_fuse        = norm_layer,
                alignment_method = cfg.alignment_method,
                fusion_method    = cfg.fusion_method,
                fusion_combination = fusion_combination,
                use_intensity_enhancement = cfg.use_intensity_enhancement,
                num_classes= cfg.num_classes
            )
        elif cfg.backbone == 'mit_b0SG': #modified multi generic SGate
            logger.info('Using backbone: Segformer-B0SG')
            self.channels = [32, 64, 160, 256]
            from .encoders.dual_segformer_genSGate import mit_b0 as backbone
            # how many extra modalities are active
            self.n_modal = len(C.active_modalities())
            # total channels after concatenating them (1 ch for single, 3 for multi)
            extra_ch = [1 if single else 3 for single in C.x_is_single_channel]
            
            # Safeguard for fusion_combination
            fusion_combination = getattr(cfg, 'fusion_combination', 'sigmoid_gating')
        
            # instantiate with image size, RGB channels, and concatenated extras
            self.backbone = backbone(
                img_size         = C.image_height,      # or C.image_size if you have that
                in_chans         = 3,                   # RGB always has 3 channels
                extra_in_chans   = extra_ch,            # total extra‐modal channels
                n_modal          = self.n_modal,
                norm_fuse        = norm_layer,
                alignment_method = cfg.alignment_method,
                fusion_method    = cfg.fusion_method,
                fusion_combination = fusion_combination,
                use_intensity_enhancement = cfg.use_intensity_enhancement,
                num_classes= cfg.num_classes
            )
        else:
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.n_modal = len(C.active_modalities())
            self.backbone = backbone(n_modal = self.n_modal,norm_fuse=norm_layer, alignment_method = cfg.alignment_method, fusion_method = cfg.fusion_method)

        self.aux_head = None

        if cfg.decoder == 'MLPDecoder':
            logger.info('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
        
        elif cfg.decoder == 'UPernet':
            logger.info('Using Upernet Decoder')
            from .decoders.UPernet import UPerHead
            self.decode_head = UPerHead(in_channels=self.channels ,num_classes=cfg.num_classes, norm_layer=norm_layer, channels=512)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)
        
        elif cfg.decoder == 'deeplabv3+':
            logger.info('Using Decoder: DeepLabV3+')
            from .decoders.deeplabv3plus import DeepLabV3Plus as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        else:
            logger.info('No decoder(FCN-32s)')
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.num_classes, norm_layer=norm_layer)

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)
    
    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_xs):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape

        # Build a list of per-modality channel counts: 1 if single, else 3
        ch_counts = [1 if single else 3 for single in C.x_is_single_channel]

        # If modal_xs already a list/tuple, use it; otherwise split along C dim
        if isinstance(modal_xs, (list, tuple)):
            xs = list(modal_xs)
        else:
            # modal_xs: Tensor[B, sum(ch_counts), H, W]
            xs = list(modal_xs.split(ch_counts, dim=1))

        # Pass RGB plus list of extra-modality tensors into the backbone
        x = self.backbone(rgb, xs)

        out = self.decode_head.forward(x)
        out = F.interpolate(
            out,
            size=orisize[2:],
            mode='bilinear',
            align_corners=False
        )
        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(
                aux_fm,
                size=orisize[2:],
                mode='bilinear',
                align_corners=False
            )
            return out, aux_fm

        return out


    def forward(self, rgb, modal_x, label=None):
        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out = self.encode_decode(rgb, modal_x)
        if label is not None:
            loss = self.criterion(out, label.long())
            if self.aux_head:
                loss += self.aux_rate * self.criterion(aux_fm, label.long())
            return loss
        return out