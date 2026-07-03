import argparse
import sys

from rknn.api import RKNN

DEFAULT_ONNX_MODEL = 'best_nano_111.onnx'
DEFAULT_RKNN_MODEL = 'best_nano_111_rv1126b_i8_keep_output_float.rknn'
DEFAULT_DATASET = 'dataset.txt'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert YOLO11 ONNX to RKNN for rv1126b.'
    )
    parser.add_argument(
        '--onnx',
        default=DEFAULT_ONNX_MODEL,
        help='Path to the ONNX model file.'
    )
    parser.add_argument(
        '--output',
        default=DEFAULT_RKNN_MODEL,
        help='Path to the output RKNN model file.'
    )
    parser.add_argument(
        '--platform',
        default='rv1126b',
        help='RKNN target platform, e.g. rv1126b.'
    )
    parser.add_argument(
        '--dtype',
        choices=['fp', 'i8'],
        default='i8',
        help='Build type: fp for float RKNN, i8 for quantized RKNN.'
    )
    parser.add_argument(
        '--dataset',
        default=DEFAULT_DATASET,
        help='Calibration dataset list used for quantization.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    do_quantization = args.dtype == 'i8'

    rknn = RKNN(verbose=True)

    print('--> Config model')

    # 关键修改：关闭 output_optimize
    # 目标：避免最终 output0 被强制改成 int8
    config_kwargs = {
        'target_platform': args.platform,
        'mean_values': [[0, 0, 0]],
        'std_values': [[255, 255, 255]],
        'output_optimize': False,
    }

    try:
        ret = rknn.config(**config_kwargs)
    except TypeError:
        print('Warning: this rknn-toolkit2 version does not support output_optimize in config().')
        print('Fallback to config without output_optimize.')
        config_kwargs.pop('output_optimize')
        ret = rknn.config(**config_kwargs)

    if ret != 0:
        print('Config failed!')
        return ret

    print('done')

    print('--> Load ONNX model')
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print('Load ONNX failed!')
        return ret
    print('done')

    print('--> Build model')
    build_kwargs = {
        'do_quantization': do_quantization
    }

    if do_quantization:
        build_kwargs['dataset'] = args.dataset

    ret = rknn.build(**build_kwargs)
    if ret != 0:
        print('Build model failed!')
        return ret
    print('done')

    print('--> Export RKNN model')
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        print('Export RKNN failed!')
        return ret
    print('done')

    rknn.release()
    print('RKNN model generated:', args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())