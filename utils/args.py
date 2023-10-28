import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('BYHE training', add_help=False)
    # Main params.
    parser.add_argument('--frame-path', default='X:/vipl-frame/frame_list', type=str,
                        help="""Please specify path to the 'frame_list' as input.""")
    parser.add_argument('--mask-path', default='X:/vipl-frame/mask_list', type=str,
                        help="""Please specify path to the 'mask_list' as GT.""")
    parser.add_argument('--wave-path', default='X:/vipl-frame/wave_gt', type=str,
                        help="""Please specify path to the 'wave' as GT.""")
    parser.add_argument('--length', default=70, type=int, help="""Length of video frames.""")
    parser.add_argument('--test-length', default=300, type=int, help="""Length for video frames testing (HR Calculate).""")
    parser.add_argument('--SSL-length', default=150, type=int, help="""Length for self-supervised learning.""")
    parser.add_argument('--win-length', default=11, type=int, help="""Sliding window length. (default: 11)""")
    parser.add_argument('--output-dir', default='./saved/', type=str, help="""Path to save logs and checkpoints.""")
    parser.add_argument('--GPU-id', default=0, type=int, help="""Index of GPUs.""")
    parser.add_argument('--saveckp-freq', default=10, type=int, help="""Save checkpoint every x epochs.""")
    parser.add_argument('--num-workers', default=0, type=int, help="""Number of data loading workers per GPU. (default: 
                        0)""")
    parser.add_argument('--batch-size', default=1, type=int, help="""batch-size: number of distinct images loaded on GPU.
                        (default: 6)""")
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs of training. (default: 50)')
    parser.add_argument("--lr", default=5e-4, type=float, help="""Learning rate at the end of linear warmup (highest LR 
                        used during training). The learning rate is linearly scaled with the batch size, and specified here
                        for a reference batch size of 256. (default: 1e-3)""")
    parser.add_argument('--min-lr', default=1e-6, type=float, help="""Target LR at the end of optimization. We use a cosine
                        LR schedule with linear warmup. (default: 1e-5)""")
    parser.add_argument('--optimizer', default='adam', type=str, choices=['adamw', 'adam', 'sgd', 'lars'], help="""Type of 
                        optimizer. We recommend using adamw with ViTs. (default: 'adamw')""")
    parser.add_argument("--warmup-epochs", default=0, type=int, help="""Number of epochs for the linear learning-rate warm
                         up. (default: 20)""")
    parser.add_argument('--weight-decay', default=1e-5, type=float, help="""Initial value of the weight decay. With ViT, a 
                        smaller value at the beginning of training works well. (default: 1e-5)""")
    parser.add_argument('--weight-decay-end', default=1e-3, type=float, help="""Final value of the weight decay. We use a 
                        cosine schedule for WD and using a larger decay by the end of training improves performance for 
                        ViTs. (default: 1e-3)""")
    parser.add_argument('--log-enable', default='True', type=str, help="""Whether or not enable tensorboard and logging. 
                       (Default: True).""")
    parser.add_argument('--print-freq', default=1, type=int, help="""Print metrics every x iterations.""")
    parser.add_argument('--log-theme', default='VIPL', type=str, help="""Annotation for tensorboard.""")
    return parser