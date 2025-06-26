from config import args as args_config
import time
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = args_config.address
os.environ["MASTER_PORT"] = args_config.port

import json
import numpy as np
from tqdm import tqdm

from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
torch.autograd.set_detect_anomaly(True)

import utility
from model.ognidc import OGNIDC

from summary.gcsummary import OGNIDCSummary
from metric.dcmetric import DCMetric
from data import get as get_data
from loss.sequentialloss import SequentialLoss

# Multi-GPU and Mixed precision supports
# NOTE : Only 1 process per GPU is supported now
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.cuda.amp as amp
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Minimize randomness
def init_seed(seed=None):
    if seed is None:
        seed = args_config.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_args(args):
    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain)

            # new_args = checkpoint['args']
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume
            new_args.start_epoch = checkpoint['epoch'] + 1

    return new_args


def train(gpu, args):
    # Initialize workers
    # NOTE : the worker with gpu=0 will do logging
    if args.multiprocessing:
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=args.num_gpus, rank=gpu)
    else:
        dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{args.tcp_port}',
                              world_size=args.num_gpus, rank=gpu)
    torch.cuda.set_device(gpu)

    # Prepare dataset
    data_train = get_data(args, 'train')
    data_val = get_data(args, 'val')

    # data_train = data(args, 'train')
    # data_val = data(args, 'val')

    sampler_train = DistributedSampler(
        data_train, num_replicas=args.num_gpus, rank=gpu)

    batch_size = args.batch_size

    loader_train = DataLoader(
        dataset=data_train, batch_size=batch_size, shuffle=False,
        num_workers=args.num_threads, pin_memory=True, sampler=sampler_train,
        drop_last=True)
    loader_val = DataLoader(
        dataset=data_val, batch_size=1, shuffle=False,
        num_workers=4, drop_last=False)

    if gpu == 0:
        print(f'Each GPU with training data {len(loader_train)}, validation data {len(loader_val)}!')

    # Network
    if args.model == 'OGNIDC':
        net = OGNIDC(args)
    else:
        raise TypeError(args.model, ['OGNIDC', ])
    net.cuda(gpu)

    # if gpu == 0:
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        # checkpoint = torch.load(args.pretrain, map_location={'cuda:0': 'cuda:%d' % gpu})
        checkpoint = torch.load(args.pretrain, map_location='cpu')
        net.load_state_dict(checkpoint['net'])

        print('Load network parameters from : {}'.format(args.pretrain))

    # Loss
    if args.model == 'OGNIDC':
        loss = SequentialLoss(args)
        summ = OGNIDCSummary
    else:
        raise NotImplementedError

    loss.cuda(gpu)

    # Optimizer
    optimizer, scheduler = utility.make_optimizer_scheduler(args, net, len(loader_train))
    net = apex.parallel.convert_syncbn_model(net)
    net, optimizer = amp.initialize(net, optimizer, opt_level=args.opt_level, verbosity=0)

    # if gpu == 0:
    if args.pretrain is not None:
        if args.resume:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])

                scheduler.load_state_dict(checkpoint['scheduler'])
                scheduler.milestones = Counter(args.milestones)

                amp.load_state_dict(checkpoint['amp'])

                print('Resume optimizer, scheduler and amp '
                      'from : {}'.format(args.pretrain))
            except KeyError:
                print('State dicts for resume are not saved. '
                      'Use --save_full argument')

        del checkpoint

    net = DDP(net)

    metric = DCMetric(args)
    best_val_rmse = 1e10

    if gpu == 0:
        # print('\n' + '='*40 + '\n')
        # print(net)
        # print('\n' + '='*40 + '\n')
        utility.backup_source_code(args.save_dir + '/code')
        try:
            os.makedirs(args.save_dir, exist_ok=True)
            os.makedirs(args.save_dir + '/train', exist_ok=True)
            os.makedirs(args.save_dir + '/val', exist_ok=True)
        except OSError:
            pass

    if gpu == 0:
        writer_train = summ(args.save_dir, 'train', args,
                            loss.loss_name, metric.metric_name)
        writer_val = summ(args.save_dir, 'val', args,
                            loss.loss_name, metric.metric_name)

        with open(args.save_dir + '/args.json', 'w') as args_json:
            json.dump(args.__dict__, args_json, indent=4)

    if args.warm_up:
        warm_up_cnt = 0.0
        warm_up_max_cnt = len(loader_train)+1.0

    for epoch in range(args.start_epoch, args.epochs+1):
        # Train
        net.train()

        sampler_train.set_epoch(epoch)

        if gpu == 0:
            current_time = time.strftime('%y%m%d@%H:%M:%S')

            list_lr = []
            for g in optimizer.param_groups:
                list_lr.append(g['lr'])

            print('=== Epoch {:5d} / {:5d} | Lr : {} | {} | {} ==='.format(
                epoch, args.epochs, list_lr, current_time, args.save_dir
            ))

        num_sample = len(loader_train) * loader_train.batch_size * args.num_gpus

        if gpu == 0:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        init_seed(seed=int(time.time()))
        for batch, sample in enumerate(loader_train):
            sample = {key: val.cuda(gpu) for key, val in sample.items()
                      if val is not None}

            if epoch == 1 and args.warm_up:
                warm_up_cnt += 1

                for param_group in optimizer.param_groups:
                    lr_warm_up = param_group['initial_lr'] \
                                 * warm_up_cnt / warm_up_max_cnt
                    param_group['lr'] = lr_warm_up

            optimizer.zero_grad()

            output = net(sample)

            # visualization
            # writer_train.save(epoch, batch, sample, output)
            # assert False
            
            loss_sum, loss_val = loss(sample, output)

            # Divide by batch size
            loss_sum = loss_sum / loader_train.batch_size
            loss_val = loss_val / loader_train.batch_size

            with amp.scale_loss(loss_sum, optimizer) as scaled_loss:
                scaled_loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()

            if gpu == 0:
                metric_val = metric.evaluate(sample, output, 'train')
                writer_train.add(loss_val, metric_val)

                log_cnt += 1
                log_loss += loss_sum.item()

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss = {:.4f}'.format(
                    'Train', current_time, log_loss / log_cnt)

                if epoch == 1 and args.warm_up:
                    list_lr = []
                    for g in optimizer.param_groups:
                        list_lr.append(round(g['lr'], 6))
                    error_str = '{} | Lr Warm Up : {}'.format(error_str,
                                                              list_lr)
                else:
                    list_lr = []
                    for g in optimizer.param_groups:
                        list_lr.append(round(g['lr'], 6))
                    error_str = '{} | Lr : {}'.format(error_str, list_lr)

                if batch % args.print_freq == 0:
                    pbar.set_description(error_str)
                    pbar.update(loader_train.batch_size * args.num_gpus)

        # update the scheduler state before saving
        scheduler.step()

        if gpu == 0:
            pbar.close()

            writer_train.update(epoch, sample, output)


            if args.save_full or epoch == args.epochs:
                state = {
                    'epoch': epoch,
                    'net': net.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'amp': amp.state_dict(),
                    'args': args
                }
            else:
                state = {
                    'epoch': epoch,
                    'net': net.module.state_dict(),
                    'args': args
                }

            torch.save(state, '{}/model_latest.pt'.format(args.save_dir))
            if epoch % 10 == 1 or epoch == args.epochs:
                torch.save(state, '{}/model_{:05d}.pt'.format(args.save_dir, epoch))

        # only effective for mixed dataset training:
        # re-sample the sample indices after each epoch
        if gpu == 0:
            data_train.refresh_indices()

        # Val
        torch.set_grad_enabled(False)
        net.eval()

        num_sample = len(loader_val) * loader_val.batch_size
        if gpu == 0:
            pbar = tqdm(total=num_sample)
        log_cnt = 0.0
        log_loss = 0.0

        init_seed()
        for batch, sample in enumerate(loader_val):
            sample = {key: val.cuda(gpu) for key, val in sample.items()
                      if val is not None}

            with torch.no_grad():
                output = net(sample)

            loss_sum, loss_val = loss(sample, output)

            # Divide by batch size
            loss_sum = loss_sum / loader_val.batch_size
            loss_val = loss_val / loader_val.batch_size

            if gpu == 0:
                metric_val = metric.evaluate(sample, output, 'val')
                writer_val.add(loss_val, metric_val)

                log_cnt += 1
                log_loss += loss_sum.item()

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss = {:.4f}'.format(
                    'Val', current_time, log_loss / log_cnt)
                if batch % args.print_freq == 0:
                    pbar.set_description(error_str)
                    pbar.update(loader_val.batch_size)

                # don't write samples to save disk space
                # if epoch % 5 == 1:
                #     writer_val.save(epoch, batch, sample, output)

        if gpu == 0:
            pbar.close()

            rmse = writer_val.update(epoch, sample, output)

            is_best = rmse < best_val_rmse
            if is_best:
                best_val_rmse = rmse
                torch.save(state, '{}/model_best.pt'.format(args.save_dir))

        torch.set_grad_enabled(True)


def test(args):
    # Prepare dataset
    # data = get_data(args)
    # data_test = data(args, 'test')

    data_test = get_data(args, 'test')

    loader_test = DataLoader(dataset=data_test, batch_size=1,
                             shuffle=False, num_workers=args.num_threads)

    if args.model == 'OGNIDC':
        loss = SequentialLoss(args)
        summ = OGNIDCSummary
    else:
        raise NotImplementedError

    # Network
    if args.model == 'OGNIDC':
        net = OGNIDC(args)
    else:
        raise TypeError(args.model, ['OGNIDC', ])
    net.cuda()

    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        checkpoint = torch.load(args.pretrain)
        key_m, key_u = net.load_state_dict(checkpoint['net'], strict=False)

        if key_u:
            print('Unexpected keys :')
            print(key_u)

        if key_m:
            print('Missing keys :')
            print(key_m)
            raise KeyError
        print('Checkpoint loaded from {}!'.format(args.pretrain))

    net = nn.DataParallel(net)

    metric = DCMetric(args)

    try:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.save_dir + '/test', exist_ok=True)
    except OSError:
        pass

    writer_test = summ(args.save_dir, 'test', args, None, metric.metric_name)

    net.eval()

    num_sample = len(loader_test)*loader_test.batch_size

    pbar = tqdm(total=num_sample)

    t_total = 0

    init_seed()
    for batch, sample in enumerate(loader_test):
        sample = {key: val.cuda() for key, val in sample.items()
                  if val is not None}

        rgb = sample['rgb']
        dep = sample['dep']

        rgb_raw = torch.clone(rgb)
        dep_raw = torch.clone(dep)

        _, _, H, W = rgb.shape
        diviser = int(4 * 2 ** (args.num_resolution-1))
        if not H % diviser == 0:
            H_new = (H // diviser + 1) * diviser
            H_pad = H_new - H
            rgb = torch.nn.functional.pad(rgb, (0, 0, 0, H_pad))
            dep = torch.nn.functional.pad(dep, (0, 0, 0, H_pad))
        else:
            H_new = H
            H_pad = 0

        if not W % diviser == 0:
            W_new = (W // diviser + 1) * diviser
            W_pad = W_new - W
            rgb = torch.nn.functional.pad(rgb, (0, W_pad, 0, 0))
            dep = torch.nn.functional.pad(dep, (0, W_pad, 0, 0))
        else:
            W_new = W
            W_pad = 0

        sample['rgb'] = rgb
        sample['dep'] = dep

        t0 = time.time()
        with torch.no_grad():
            output = net(sample)
        t1 = time.time()

        output['pred'] = output['pred'][..., :H_new - H_pad, :W_new - W_pad]
        output['pred_inter'] = [pred[..., :H_new - H_pad, :W_new - W_pad] for pred in output['pred_inter']]
        sample['rgb'] = rgb_raw
        sample['dep'] = dep_raw

        t_total += (t1 - t0)

        if args.test_augment:
            sample_fliplr = {key: torch.clone(val) for key, val in sample.items()
                  if val is not None}
            sample_fliplr['rgb'] = torch.flip(sample_fliplr['rgb'], (3,))
            sample_fliplr['dep'] = torch.flip(sample_fliplr['dep'], (3,))

            output_flip = net(sample_fliplr)
            pred_flip_back = torch.flip(output_flip['pred'], (3,))

            output['pred'] = (pred_flip_back + output['pred']) / 2.0

        metric_val = metric.evaluate(sample, output, 'test')

        writer_test.add(None, metric_val)

        # Save data for analysis
        writer_test.save(args.epochs, batch, sample, output)

        current_time = time.strftime('%y%m%d@%H:%M:%S')
        error_str = '{} | Test'.format(current_time)
        if batch % args.print_freq == 0:
            pbar.set_description(error_str)
            pbar.update(loader_test.batch_size)

    pbar.close()

    # writer_test.update(args.epochs, sample, output)
    writer_test.print_loss(args.epochs)

    t_avg = t_total / num_sample
    print('Elapsed time : {} sec, '
          'Average processing time : {} sec'.format(t_total, t_avg))


def main(args):
    init_seed()
    if not args.test_only:
        if not args.multiprocessing:
            train(0, args)
        else:
            assert args.num_gpus > 0

            spawn_context = mp.spawn(train, nprocs=args.num_gpus, args=(args,),
                                     join=False)

            while not spawn_context.join():
                pass

            for process in spawn_context.processes:
                if process.is_alive():
                    process.terminate()
                process.join()

        # args.pretrain = '{}/model_{:05d}.pt'.format(args.save_dir, args.epochs)
        args.pretrain = '{}/model_best.pt'.format(args.save_dir)

    test(args)


if __name__ == '__main__':
    args_main = check_args(args_config)

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args_main)):
        print(key, ':',  getattr(args_main, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')

    main(args_main)