import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register, serializable
from ppdet.modeling.ops import get_act_fn
from ppdet.modeling.transformers.detr_transformer import TransformerEncoder

from ..backbones.csp_darknet import BaseConv
from ..backbones.cspresnet import RepVggBlock
from ..initializer import linear_init_
from ..layers import MultiHeadAttention
from ..shape_spec import ShapeSpec

__all__ = [
    'SARCDETRTransformerLayer', 'SARCDETRHybridEncoder',
    'SARCDETRAccelHybridEncoder'
]


class SimilarityAdaptiveConvolution(nn.Layer):
    def __init__(self,
                 channels,
                 kernel_size=3,
                 kernel_type='rbf',
                 gamma=1.0,
                 eps=1e-6):
        super(SimilarityAdaptiveConvolution, self).__init__()
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.eps = eps
        self.prototype_pool = nn.AvgPool2D(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            exclusive=False)
        self.affine = nn.BatchNorm2D(
            channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

    def forward(self, x):
        prototype = self.prototype_pool(x)
        if self.kernel_type == 'linear':
            sim = paddle.sum(x * prototype, axis=1, keepdim=True)
        elif self.kernel_type == 'poly':
            sim = paddle.pow(
                paddle.sum(x * prototype, axis=1, keepdim=True) + 1.0, 2.0)
        else:
            diff = x - prototype
            sim = paddle.exp(
                -self.gamma * paddle.mean(diff * diff, axis=1, keepdim=True))
        gate = F.sigmoid(sim)
        return self.affine(x * gate)


class DirectionalGate(nn.Layer):
    def __init__(self, channels, reduction=4, act='silu'):
        super(DirectionalGate, self).__init__()
        hidden_channels = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = BaseConv(
            channels * 2, hidden_channels, ksize=1, stride=1, act=act)
        self.fc2 = nn.Conv2D(hidden_channels, channels * 2, kernel_size=1)

    def forward(self, horizontal, vertical):
        weights = self.pool(paddle.concat([horizontal, vertical], axis=1))
        weights = self.fc2(self.fc1(weights))
        weights = paddle.reshape(
            weights,
            [weights.shape[0], 2, horizontal.shape[1], 1, 1])
        weights = F.softmax(weights, axis=1)
        return horizontal * weights[:, 0] + vertical * weights[:, 1]


class ReceptiveFieldCrossConvolution(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 span=5,
                 stride=1,
                 act='silu',
                 deploy=False):
        super(ReceptiveFieldCrossConvolution, self).__init__()
        self.deploy = deploy
        if deploy:
            self.cross = nn.Sequential(
                BaseConv(
                    in_channels,
                    in_channels,
                    ksize=3,
                    stride=stride,
                    groups=in_channels,
                    act=act),
                BaseConv(
                    in_channels, out_channels, ksize=1, stride=1, act=act))
        else:
            padding = span // 2
            self.horizontal = nn.Sequential(
                nn.Conv2D(
                    in_channels,
                    out_channels,
                    kernel_size=(1, span),
                    stride=stride,
                    padding=(0, padding),
                    bias_attr=False),
                nn.BatchNorm2D(
                    out_channels,
                    weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
            self.vertical = nn.Sequential(
                nn.Conv2D(
                    in_channels,
                    out_channels,
                    kernel_size=(span, 1),
                    stride=stride,
                    padding=(padding, 0),
                    bias_attr=False),
                nn.BatchNorm2D(
                    out_channels,
                    weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                    bias_attr=ParamAttr(regularizer=L2Decay(0.0))))
            self.directional_gate = DirectionalGate(out_channels, act=act)
            self.isotropic = BaseConv(
                out_channels, out_channels, ksize=3, stride=1, act=act)

    def forward(self, x):
        if self.deploy:
            return self.cross(x)
        horizontal = self.horizontal(x)
        vertical = self.vertical(x)
        fused = self.directional_gate(horizontal, vertical)
        return self.isotropic(fused)


class SARCDETRBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act='silu',
                 rcc_span=5,
                 deploy=False,
                 use_sac=True):
        super(SARCDETRBlock, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.sac = SimilarityAdaptiveConvolution(
            hidden_channels) if use_sac else nn.Identity()
        self.rcc = ReceptiveFieldCrossConvolution(
            hidden_channels,
            hidden_channels,
            span=rcc_span,
            stride=1,
            act=act,
            deploy=deploy)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = BaseConv(
                hidden_channels,
                out_channels,
                ksize=1,
                stride=1,
                bias=bias,
                act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        branch = self.conv1(x)
        sarc = self.rcc(self.sac(branch))
        branch = self.bottlenecks(branch)
        shortcut = self.conv2(x)
        return self.conv3(branch + shortcut + sarc)


@register
class SARCDETRTransformerLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(SARCDETRTransformerLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout, mode='upscale_in_train')
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode='upscale_in_train')
        self.dropout2 = nn.Dropout(dropout, mode='upscale_in_train')
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


@register
@serializable
class SARCDETRHybridEncoder(nn.Layer):
    __shared__ = ['depth_mult', 'act', 'trt', 'eval_size']
    __inject__ = ['encoder_layer']

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 encoder_layer='SARCDETRTransformerLayer',
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 trt=False,
                 eval_size=None,
                 rcc_span=5,
                 deploy=False,
                 use_sac=True):
        super(SARCDETRHybridEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size
        self.rcc_span = rcc_span
        self.deploy = deploy
        self.use_sac = use_sac

        self.input_proj = nn.LayerList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channel, hidden_dim, kernel_size=1, bias_attr=False),
                    nn.BatchNorm2D(
                        hidden_dim,
                        weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0)))))

        self.encoder = nn.LayerList([
            TransformerEncoder(encoder_layer, num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        self.lateral_convs = nn.LayerList()
        self.fpn_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(BaseConv(hidden_dim, hidden_dim, 1, 1,
                                               act=act))
            self.fpn_blocks.append(
                SARCDETRBlock(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion,
                    rcc_span=rcc_span,
                    deploy=deploy,
                    use_sac=use_sac))

        self.downsample_convs = nn.LayerList()
        self.pan_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1):
            self.downsample_convs.append(
                BaseConv(hidden_dim, hidden_dim, 3, stride=2, act=act))
            self.pan_blocks.append(
                SARCDETRBlock(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion,
                    rcc_span=rcc_span,
                    deploy=deploy,
                    use_sac=use_sac))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w,
                                           h,
                                           embed_dim=256,
                                           temperature=10000.):
        grid_w = paddle.arange(int(w), dtype=paddle.float32)
        grid_h = paddle.arange(int(h), dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return paddle.concat(
            [
                paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h),
                paddle.cos(out_h)
            ],
            axis=1)[None, :, :]

    def forward(self, feats, for_mot=False):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).transpose(
                    [0, 2, 1])
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.transpose([0, 2, 1]).reshape(
                    [-1, self.hidden_dim, h, w])

        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(
                feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                paddle.concat([upsample_feat, feat_low], axis=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](
                paddle.concat([downsample_feat, feat_height], axis=1))
            outs.append(out)

        return outs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'feat_strides': [i.stride for i in input_shape]
        }

    @property
    def out_shape(self):
        return [
            ShapeSpec(channels=self.hidden_dim, stride=self.feat_strides[idx])
            for idx in range(len(self.in_channels))
        ]


@register
@serializable
class SARCDETRAccelHybridEncoder(SARCDETRHybridEncoder):
    __shared__ = ['depth_mult', 'act', 'trt', 'eval_size']
    __inject__ = ['encoder_layer']

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 encoder_layer='SARCDETRTransformerLayer',
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 trt=False,
                 eval_size=None,
                 rcc_span=3,
                 deploy=True,
                 use_sac=True):
        super(SARCDETRAccelHybridEncoder, self).__init__(
            in_channels=in_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers,
            encoder_layer=encoder_layer,
            pe_temperature=pe_temperature,
            expansion=expansion,
            depth_mult=depth_mult,
            act=act,
            trt=trt,
            eval_size=eval_size,
            rcc_span=rcc_span,
            deploy=deploy,
            use_sac=use_sac)
