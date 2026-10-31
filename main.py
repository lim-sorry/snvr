import math
import os
import time
import json
import random
import argparse
import datetime
import numpy as np
import pickle

import torch
import torch.backends.cudnn as cudnn

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from logger import create_logger
from data import build_loader
from models import build_model
from optimizer import build_optimizer
from lr_scheduler import build_scheduler, global_lr_at
from optimizer.sb import apply_spectral_bound_model
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper

from optimizer.tb import Tempbalance
from optimizer.htr import HeavyTailedRegularizer



def parse_option():
    parser = argparse.ArgumentParser('Spectral norm variance regularization training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    # parser.add_argument('--pretrained',
    #                     help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--seed', type=int, default=0, help='random seed for initialization')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--name', type=str, help='model name')
    
    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    # a Subset (DATA.SUBSET_RATIO < 1) keeps the transform on the wrapped dataset
    logger.info(f'train dataset transform: {getattr(dataset_train, "dataset", dataset_train).transform}')
    logger.info(f'valid dataset transform: {dataset_val.transform}')

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()

    if config.MODEL.NAME == 'tb':
        # as in the original TempBalance: the optimizer is built from TB's layer-wise
        # parameter groups so that step() can assign a learning rate per layer
        s1, s2 = config.TRAIN.TB_LR_RATIO
        lr_scheduler = Tempbalance(model, lr_min_ratio=s1, lr_max_ratio=s2)
        # the global LR follows the same schedule as every other method (warmup + cosine + min_lr);
        # TB only distributes it over the layers within [s1, s2] x global LR
        tb_param_groups, n_tuned = lr_scheduler.build_optimizer_param_group(untuned_lr=global_lr_at(config, 0), initialize=True)
        logger.info(f"TempBalance: {n_tuned} layer-wise parameter groups + 1 untuned group, lr ratio [{s1}, {s2}]")
        optimizer = build_optimizer(config, model, param_groups=tb_param_groups)
    else:
        optimizer = build_optimizer(config, model)
        lr_scheduler = build_scheduler(config, optimizer)
    logger.info(f"Optimizer: {optimizer}")
    loss_scaler = NativeScalerWithGradNormCount()

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_acc1 = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        best_epoch, max_acc1, max_acc5 = load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f'accuracy1: {acc1:.3f} / {max_acc1:.3f}%, accuracy5: {max_acc5:.3f}%, loss: {loss:.4f}')
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f'accuracy1: {acc1:.3f} / {max_acc1:.3f}%, accuracy5: {acc5:.3f}%, loss: {loss:.4f}')

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    if config.MODEL.NAME == 'sb':
        # the bound is enforced after every optimizer step; also once before the first forward
        apply_spectral_bound_model(model, coeff=config.TRAIN.SB_COEFF)
        logger.info(f"SB: spectral bound coeff {config.TRAIN.SB_COEFF} on the hidden conv / linear layers")

    htr = None
    if config.MODEL.NAME in ('war', 'srr'):
        # WAR / SRR on the hidden layers; sigma_max by power iteration (warm start), WAR's alpha once per epoch
        htr = HeavyTailedRegularizer(model, n_power=config.TRAIN.OPTIMIZER.N_POWER)
        lam = config.TRAIN.WAR_LAMBDA if config.MODEL.NAME == 'war' else config.TRAIN.SRR_LAMBDA
        logger.info(f"{config.MODEL.NAME.upper()}: lambda {lam}, {len(htr.weights)} hidden layers, "
                    f"n_power {config.TRAIN.OPTIMIZER.N_POWER}")

    logger.info("Start training")
    sv_best = {}
    epoch_grad_norms = []
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if epoch >= int(os.environ.get('COST_EPOCHS', 10 ** 9)):
            # cost measurement (cost_bench.sh): the first epochs of the real schedule only, no results
            logger.info(f'COST_EPOCHS: stopped after {epoch} epochs')
            return
        data_loader_train.sampler.set_epoch(epoch)

        grad_norms = train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler, htr)
        epoch_grad_norms.append(grad_norms)

        if epoch < config.TRAIN.EVAL_DENSE_FROM and (epoch + 1) % config.TRAIN.EVAL_EVERY != 0:
            continue  # sparse evaluation early in training (the best epoch comes late)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        if max_acc1 < acc1:
            best_epoch = epoch
            max_acc1 = acc1
            max_acc5 = acc5
            sv_best = singular_values(model)  # spectra of the best model, kept for sv.npz (every run)
            if config.SAVE_CKPT:
                save_checkpoint(config, epoch, model, acc1, acc5, optimizer, lr_scheduler, loss_scaler, logger, is_best=True)
        logger.info(f'accuracy1: {acc1:.3f} / {max_acc1:.3f}%, accuracy5: {acc5:.3f}%, loss: {loss:.4f}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    results = {
        'epoch': best_epoch,
        'accuracy1': max_acc1,
        'accuracy5': max_acc5,
        'final_accuracy1': acc1,
        'final_accuracy5': acc5,
        'config': config,
    }
    
    epoch_grad_norms = np.stack(epoch_grad_norms)
    np.save(os.path.join(config.OUTPUT, 'grad.npy'), epoch_grad_norms)
    # the layers of grad.npy's axis 1: the weight matrices / kernels (dim > 1) in parameter order,
    # without the last one (see utils.ampscaler_get_grad_norm)
    with open(os.path.join(config.OUTPUT, 'grad_layers.json'), 'w') as f:
        json.dump([n for n, p in model.named_parameters() if p.requires_grad and p.dim() > 1][:-1], f)
    # singular values of every conv / linear weight (as an out x in*k*k matrix) of the best and the
    # final model, for the spectrum plots without a checkpoint
    sv_final = singular_values(model)
    np.savez_compressed(os.path.join(config.OUTPUT, 'sv.npz'), best_epoch=best_epoch,
                        layers=np.array(list(sv_final.keys())),
                        **{f'best/{k}': v for k, v in sv_best.items()}, **{f'final/{k}': v for k, v in sv_final.items()})
    
    results_path = os.path.join(config.OUTPUT, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)


def _svd_flops(a, full_matrices=False, compute_uv=True, *args, out_shape=None, **kwargs):
    """FLOPs of an SVD of (batched) m x n matrices (Golub & Van Loan, Table 5.5.1): singular values only
    4 m n^2 - 4 n^3 / 3, thin SVD with U and V 6 m n^2 + 20 n^3 (m >= n); FlopCounterMode has no formula."""
    *batch, m, n = a
    m, n = max(m, n), min(m, n)
    f = 6 * m * n * n + 20 * n ** 3 if compute_uv else 4 * m * n * n - 4 * n ** 3 / 3
    return int(f * math.prod(batch))


def _mv_flops(a, v, *args, out_shape=None, **kwargs):
    """matrix-vector product (power iterations of SN / SNR / SNVR / WAR / SRR): 2 m n."""
    return 2 * a[0] * a[1]


def _addmv_flops(x, a, v, *args, out_shape=None, **kwargs):
    return 2 * a[0] * a[1]


def _dot_flops(a, b, *args, out_shape=None, **kwargs):
    return 2 * a[0]


class flop_counter:
    """Count the FLOPs of the enclosed code (matmul / conv / SDPA formulas of torch.utils.flop_counter
    plus SVD and matrix-vector products) when enabled, and log them (per training step: amortized per-epoch work is logged
    with the number of steps per epoch)."""
    def __init__(self, enabled, what, steps=None):
        self.enabled, self.what, self.steps = enabled, what, steps

    def __enter__(self):
        if self.enabled:
            from torch.utils.flop_counter import FlopCounterMode
            self.mode = FlopCounterMode(display=False, custom_mapping={
                torch.ops.aten._linalg_svd: _svd_flops, torch.ops.aten.mv: _mv_flops,
                torch.ops.aten.addmv: _addmv_flops, torch.ops.aten.dot: _dot_flops, torch.ops.aten.vdot: _dot_flops})
            self.mode.__enter__()
        return self

    def __exit__(self, *exc):
        if self.enabled:
            self.mode.__exit__(*exc)
            logger.info(f'COST_FLOPS {self.what}: {self.mode.get_total_flops()} FLOPs'
                        + (f', {self.steps} steps per epoch' if self.steps else ''))
        return False


@torch.no_grad()
def singular_values(model):
    """{module name: singular values} of every Conv2d / Linear weight, reshaped to (out, in*k*k); for
    WN / SN the effective (normalized) weight of the last forward pass."""
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            w = m.weight.detach().float()
            out[name] = torch.linalg.svdvals(w.reshape(w.size(0), -1)).cpu().numpy()
    return out


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, htr=None):
    model.train()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    start = time.time()  # the epoch's time includes the per-epoch work of WAR (alpha) and TB (layer LRs)
    # COST_FLOPS=<i>: count the FLOPs of training step i and of the per-epoch work in epoch 0 (cost table)
    flop_step = int(os.environ.get('COST_FLOPS', -1)) if epoch == 0 else -1
    if htr is not None and config.MODEL.NAME == 'war':
        with flop_counter(flop_step >= 0, 'per-epoch (WAR alpha)'):
            htr.refresh_alpha()  # the Hill alpha of every hidden layer, held for the epoch
    htr_watch = {'last_good': None, 'step': None}

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    scaler_meter = AverageMeter()

    end = time.time()
    grad_norms = []
    for idx, (samples, targets) in enumerate(data_loader):
        with flop_counter(idx == flop_step, f'step {idx}', num_steps):
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                outputs = model(samples)
            loss = criterion(outputs, targets)
        
            if config.MODEL.NAME == 'war':
                loss = loss + config.TRAIN.WAR_LAMBDA * htr.weighted_alpha()
            elif config.MODEL.NAME == 'srr':
                loss = loss + config.TRAIN.SRR_LAMBDA * htr.stable_rank()
            if htr is not None and os.environ.get('HTR_WATCHDOG'):
                # diagnostics for the occasional collapse of WAR / SRR runs: keep the last finite state,
                # dump everything when the loss turns non-finite, and stop
                if not torch.isfinite(loss):
                    torch.save({'last_good': htr_watch['last_good'], 'last_good_step': htr_watch['step'],
                                'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'u': {k: v.cpu() for k, v in htr.u.items()}, 'alpha': {k: float(v) for k, v in htr.alpha.items()},
                                'epoch': epoch, 'idx': idx, 'loss_scale': loss_scaler.state_dict()['scale'],
                                'samples': samples.detach().cpu(), 'targets': targets.detach().cpu()},
                               os.path.join(config.OUTPUT, 'collapse.pt'))
                    logger.info(f'HTR_WATCHDOG: non-finite loss at epoch {epoch} iter {idx}; state saved to collapse.pt')
                    raise SystemExit(3)
                if idx % 10 == 0:
                    htr_watch['last_good'] = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    htr_watch['step'] = (epoch, idx)

            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(), create_graph=is_second_order, update_grad=True)
            grad_norms.append(grad_norm.cpu().detach().numpy())
            optimizer.zero_grad()
            loss_scale_value = loss_scaler.state_dict()["scale"]

            torch.cuda.synchronize()

            loss_meter.update(loss.item(), targets.size(0))
            scaler_meter.update(loss_scale_value)
            batch_time.update(time.time() - end)
        
            if config.MODEL.NAME == 'sb':
                apply_spectral_bound_model(model, coeff=config.TRAIN.SB_COEFF)
        
            end = time.time()

            if idx % config.PRINT_FREQ == 0:
                lr = optimizer.param_groups[0]['lr']
                wd = optimizer.param_groups[0]['weight_decay']
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                etas = batch_time.avg * (num_steps - idx)
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
    
    # the LR is updated once per epoch: set here for the next epoch and held for the whole
    # epoch (the first epoch uses the scheduler's initial LR)
    if config.MODEL.NAME == 'tb':
        # TB re-assigns the layer-wise LRs from the next epoch's global LR
        with flop_counter(flop_step >= 0, 'per-epoch (TB)'):
            lr_scheduler.step(optimizer, global_lr_at(config, epoch + 1))
    else:
        lr_scheduler.step(epoch + 1)

    torch.cuda.synchronize()
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))} "
                f"({epoch_time:.3f} s, peak memory {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MB)")

    grad_norms = np.stack(grad_norms).transpose()
    return grad_norms

@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if os.path.exists(os.path.join(config.OUTPUT, 'results.pkl')):
        exit()

    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)