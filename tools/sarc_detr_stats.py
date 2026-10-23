import argparse
import math

import paddle

from ppdet.core.workspace import create, load_config
import ppdet.modeling.architectures
import ppdet.modeling.backbones
import ppdet.modeling.heads
import ppdet.modeling.losses
import ppdet.modeling.transformers


def count_params(model):
    total = sum(math.prod(p.shape) for p in model.parameters())
    trainable = sum(
        math.prod(p.shape) for p in model.parameters() if not p.stop_gradient)
    return total, trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        default='configs/rtdetr/sarc_detr_r50vd_6x_coco.yml')
    parser.add_argument('--height', type=int, default=640)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--params-only', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = create(cfg.architecture)
    model.eval()
    total, trainable = count_params(model)
    print('Config: {}'.format(args.config))
    print('Params: {:.2f} M'.format(total / 1e6))
    print('Trainable params: {:.2f} M'.format(trainable / 1e6))

    if not args.params_only:
        inputs = {
            'image': paddle.randn([1, 3, args.height, args.width]),
            'im_shape': paddle.to_tensor(
                [[args.height, args.width]], dtype='float32'),
            'scale_factor': paddle.ones([1, 2], dtype='float32'),
        }
        with paddle.no_grad():
            model(inputs)
        print('Forward: ok')


if __name__ == '__main__':
    main()
