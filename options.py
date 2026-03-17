import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int, default=5, help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze', 'deblur', 'enhance'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=192, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers.')
parser.add_argument(
    '--persistent_workers',
    action='store_true',
    help='Enable persistent dataloader workers to reduce worker startup overhead.',
)
parser.add_argument(
    '--prefetch_factor',
    type=int,
    default=2,
    help='Prefetch batches per worker for dataloader (only when num_workers > 0).',
)
parser.add_argument(
    '--allow_tf32',
    action='store_true',
    help='Enable TF32 matmul/cuDNN for faster training on Ampere+ GPUs.',
)
parser.add_argument(
    '--matmul_precision',
    type=str,
    default='highest',
    choices=['highest', 'high', 'medium'],
    help='torch.set_float32_matmul_precision value.',
)
parser.add_argument(
    '--cudnn_benchmark',
    action='store_true',
    help='Enable cuDNN benchmark for fixed-size patches to improve throughput.',
)
parser.add_argument(
    '--ddp_find_unused_parameters',
    action='store_true',
    help='Use DDP with unused-parameter search (safer but slower).',
)

# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--gopro_dir', type=str, default='data/Train/Deblur/',
                    help='where clean images of denoising saves.')
parser.add_argument('--enhance_dir', type=str, default='data/Train/Enhance/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/adair5d.ckpt", help='checkpoint save path')
parser.add_argument(
    '--weights_only_ckpt',
    action='store_true',
    help='Load ckpt_path as weights only, even if the filename looks like a training checkpoint.',
)
parser.add_argument(
    '--use_video_prior_train',
    '--use_video_prior',
    dest='use_video_prior_train',
    action='store_true',
    help='Enable prior-assisted training: use original frame + temporal/spatial prior as joint input.',
)
parser.add_argument(
    '--prior_window_size',
    type=int,
    default=9,
    help='Temporal window size for training prior extraction (odd recommended).',
)
parser.add_argument(
    '--prior_align_mode',
    type=str,
    default='ecc',
    choices=['none', 'ecc'],
    help='Alignment mode for temporal prior during training.',
)
parser.add_argument(
    '--prior_warp_mode',
    type=str,
    default='affine',
    choices=['translation', 'euclidean', 'affine', 'homography'],
    help='ECC warp mode when prior_align_mode=ecc.',
)
parser.add_argument(
    '--prior_temporal_semantics',
    type=str,
    default='stable',
    choices=['transient', 'stable'],
    help='Temporal semantics for prior extraction during training.',
)
parser.add_argument(
    '--prior_stable_band_ratio',
    type=float,
    default=0.10,
    help='Low-frequency band ratio for stable temporal prior.',
)
parser.add_argument(
    '--prior_strength',
    type=float,
    default=0.7,
    help='Strength multiplier for prior-guided conditioning in AdaIR.',
)
parser.add_argument(
    '--trust_loss_weight',
    type=float,
    default=0.2,
    help='Supervision weight for trust-map learning based on whether prior is closer to GT than the degraded input.',
)
parser.add_argument(
    '--train_trust_only',
    action='store_true',
    help='Freeze AdaIR backbone and train only the learned trust/fusion branch.',
)
parser.add_argument(
    '--train_prior_modules_only',
    action='store_true',
    help='Freeze most AdaIR weights and train trust_predictor + fre1/fre2/fre3 + output head only.',
)
parser.add_argument(
    '--prior_train_tasks',
    nargs='+',
    default=['derain', 'deblur'],
    help='Tasks to apply temporal prior extraction during training.',
)
parser.add_argument(
    '--prior_min_frames',
    type=int,
    default=3,
    help='Minimum frames in a folder to enable temporal prior extraction.',
)
parser.add_argument(
    '--derain_repeat_factor',
    type=int,
    default=20,
    help='How many times to repeat derain samples per epoch.',
)
parser.add_argument(
    '--deblur_repeat_factor',
    type=int,
    default=10,
    help='How many times to repeat deblur samples per epoch.',
)
parser.add_argument(
    "--wblogger",
    type=str,
    default="AdaIR",
    help="W&B project name.",
)
parser.add_argument("--ckpt_dir",type=str,default="AdaIR/vp_adair_rgb/",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default=2, help = "Number of GPUs to use for training")

options = parser.parse_args()
