import argparse
import os
import logging
import logging.handlers


# DEBUG < INFO < WARNING < ERROR < CRITICAL
def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(levelname)s | %(filename)s:%(lineno)s] %(asctime)s: %(message)s')

    if not os.path.isdir('log'):
        os.mkdir('log')

    file_handler = logging.FileHandler('./log/' + filename + '.log')
    stream_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def get_args():
    parser = argparse.ArgumentParser('parameters')

    #ATTENTION VARS
    parser.add_argument('--all_attention', type=bool, default=False)
    parser.add_argument('--R', type=float, default=3.0, help='Variable R in masking function (controls decay of mask to 0)')
    parser.add_argument('--z_init', type=float, default=0.1, help='mask variable which controls distance of no mask')
    parser.add_argument('--adaptive_span', type=bool, default=False)

    parser.add_argument('--use_adam', type=bool, default=False, help='Whether or not to use Adam optimizer')
    parser.add_argument('--adam_lr', type=float, default=0.001)
    parser.add_argument('--attention_kernel', type=int, default=3)
    parser.add_argument('--test', type=bool, default=False, help='Whether or not on test set')
    parser.add_argument('--small_version', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='CIFAR10, CIFAR100, MNIST')
    parser.add_argument('--model-name', type=str, default='ResNet26', help='ResNet26, ResNet38, ResNet50')
    parser.add_argument('--img-size', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--T_max', type=int, default=50)
    parser.add_argument('--eta_min', type=float, default=0.)
    parser.add_argument('--warmup_epochs', type=int, default=10)

    parser.add_argument('--lr', type=float, default=1.6)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--print-interval', type=int, default=100)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--pretrained-model', type=bool, default=False)
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--gpu-devices', type=int, nargs='+', default=None)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--rank', type=int, default=0, help='current process number')
    parser.add_argument('--world-size', type=int, default=1, help='Total number of processes to be used (number of gpus)')
    parser.add_argument('--dist-backend', type=str, default='nccl')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str)

    args = parser.parse_args()

    # TODO: Remove these comments
    logger = None #get_logger('train')
    #logger.info(vars(args))

    return args, logger
