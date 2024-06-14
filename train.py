import argparse
import gc
import os
import time

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from trajectories import data_loader
from utils import gan_g_loss, gan_d_loss, l2_loss, displacement_error, final_displacement_error

from models_gru import TrajectoryGenerator as TrajectoryGeneratorGRU
from models_gru import TrajectoryDiscriminator as TrajectoryDiscriminatorGRU

from models_gru_cnn import TrajectoryGenerator as TrajectoryGeneratorCNN
from models_gru_cnn import TrajectoryDiscriminator as TrajectoryDiscriminatorCNN

from models_gru_cnn_pool import TrajectoryGenerator as TrajectoryGeneratorPooling
from models_gru_cnn_pool import TrajectoryDiscriminator as TrajectoryDiscriminatorPooling

from models_sgan import TrajectoryGenerator as TrajectoryGeneratorSGAN
from models_sgan import TrajectoryDiscriminator as TrajectoryDiscriminatorSGAN


from utils import int_tuple, bool_flag, get_total_norm
from utils import relative_to_abs, get_dset_path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

# Changed Parameters
parser.add_argument('--dataset_name', default='ours', type=str)
parser.add_argument('--model_type', default='sgan', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=5000, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--checkpoint_every', default=20, type=int)

# Dataset options
parser.add_argument('--delim', default='\t')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=1024, type=int)
parser.add_argument('--feature_size', default=1000, type=int)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=64, type=int)
parser.add_argument('--decoder_h_dim_g', default=128, type=int)
parser.add_argument('--noise_dim', default=None, type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='ped')
parser.add_argument('--clipping_threshold_g', default=0, type=float)
parser.add_argument('--g_learning_rate', default=5e-4, type=float)
parser.add_argument('--g_steps', default=1, type=int)

# Pooling Options
parser.add_argument('--pooling_type', default='pool_net')
parser.add_argument('--pool_every_timestep', default=1, type=bool_flag)

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=1024, type=int)

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str)
parser.add_argument('--encoder_h_dim_d', default=128, type=int)
parser.add_argument('--d_learning_rate', default=5e-4, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Loss Options
parser.add_argument('--l2_loss_weight', default=0, type=float)
parser.add_argument('--best_k', default=1, type=int)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')

    long_dtype, float_dtype = get_dtypes(args)

    print("Training Information")
    print("[Dataset] {}".format(args.dataset_name))
    print("[Model] {}".format(args.model_type))

    print("Initializing dataset")
    train_dset, train_loader = data_loader(args, train_path)
    _, val_loader = data_loader(args, val_path)

    iterations_per_epoch = len(train_dset) / args.batch_size / args.d_steps
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)


    if args.model_type == 'sgan':
        generator_model = TrajectoryGeneratorSGAN 
        discriminator_model = TrajectoryDiscriminatorSGAN
    elif args.model_type == 'gru':
        generator_model = TrajectoryGeneratorGRU 
        discriminator_model = TrajectoryDiscriminatorGRU
    elif args.model_type == 'cnn':
        generator_model = TrajectoryGeneratorCNN
        discriminator_model = TrajectoryDiscriminatorCNN
    else:
        generator_model = TrajectoryGeneratorPooling
        discriminator_model = TrajectoryDiscriminatorPooling

    generator = generator_model(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            encoder_h_dim=args.encoder_h_dim_g,
            decoder_h_dim=args.decoder_h_dim_g,
            mlp_dim=args.mlp_dim,
            num_layers=args.num_layers,
            noise_dim=args.noise_dim,
            noise_type=args.noise_type,
            noise_mix_type=args.noise_mix_type,
            pooling_type=args.pooling_type,
            pool_every_timestep=args.pool_every_timestep,
            dropout=args.dropout,
            bottleneck_dim=args.bottleneck_dim,
            neighborhood_size=args.neighborhood_size,
            grid_size=args.grid_size,
            batch_norm=args.batch_norm)

    generator.apply(init_weights)
    generator.type(float_dtype).train()

    discriminator = discriminator_model(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            h_dim=args.encoder_h_dim_d,
            mlp_dim=args.mlp_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_norm=args.batch_norm,
            d_type=args.d_type)

    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=args.d_learning_rate
    )

    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    'data302/%s_%s_%s.pt' % (args.checkpoint_name, args.model_type, args.dataset_name))

    if restore_path is not None and os.path.isfile(restore_path):
        print('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        generator.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        t = 0 #checkpoint['counters']['t']
        epoch = 0 #checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(0) #t 
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'norm_d': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None,
            'd_state': None,
            'd_optim_state': None,
            'g_best_state': None,
            'd_best_state': None,
            'best_t': None,
        }
    t0 = None
    cum, itr = 0, 0
    while t < args.num_iterations:

        gc.collect()
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        epoch += 1
        itr += 1
        print('Starting epoch {}'.format(epoch))
        start = time.time()
        for batch in train_loader:
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            if d_steps_left > 0:
                step_type = 'd'
                losses_d = discriminator_step(args, batch, generator,
                                              discriminator, d_loss_fn,
                                              optimizer_d)
                checkpoint['norm_d'].append(
                    get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
            elif g_steps_left > 0:
                step_type = 'g'
                losses_g = generator_step(args, batch, generator,
                                          discriminator, g_loss_fn,
                                          optimizer_g)
                checkpoint['norm_g'].append(
                    get_total_norm(generator.parameters())
                )
                g_steps_left -= 1

            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                print('{} step took {}'.format(step_type, t2 - t1))

            if d_steps_left > 0 or g_steps_left > 0:
                continue

            if args.timing == 1:
                if t0 is not None:
                    print('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            if t % args.print_every == 0:
                print('t = {} / {}'.format(t + 1, args.num_iterations))
                for k, v in sorted(losses_d.items()):
                    print('  [D] {}: {:.3f}'.format(k, v))
                    checkpoint['D_losses'][k].append(v)
                for k, v in sorted(losses_g.items()):
                    print('  [G] {}: {:.3f}'.format(k, v))
                    checkpoint['G_losses'][k].append(v)
                checkpoint['losses_ts'].append(t)

            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                print('Checking stats ...')
                metrics_val = check_accuracy(
                    args, val_loader, generator, discriminator, d_loss_fn
                )
                
                metrics_train = check_accuracy(
                    args, train_loader, generator, discriminator,
                    d_loss_fn, limit=True
                )

                for k, v in sorted(metrics_val.items()):
                    print('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    print('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_ade = min(checkpoint['metrics_val']['ade'])

                if metrics_val['ade'] == min_ade:
                    checkpoint['best_t'] = t
                    checkpoint['g_best_state'] = generator.state_dict()
                    checkpoint['d_best_state'] = discriminator.state_dict()

                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                checkpoint_path = os.path.join(
                    args.output_dir, 'data302/%s_%s_%s.pt' % (args.checkpoint_name, args.model_type, args.dataset_name))
                print('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                print('Done.')

            t += 1
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            if t >= args.num_iterations:
                break

        end = time.time()
        cum += (end - start)
        print('Average time: %.2f' % (cum / itr))

def discriminator_step(
    args, batch, generator, discriminator, d_loss_fn, optimizer_d
):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, obs_img, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    generator_out = generator(obs_traj, obs_img, obs_traj_rel, seq_start_end)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(),
                                 args.clipping_threshold_d)
    optimizer_d.step()

    return losses


def generator_step(
    args, batch, generator, discriminator, g_loss_fn, optimizer_g
):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, obs_img, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, args.obs_len:]

    for _ in range(args.best_k):
        generator_out = generator(obs_traj, obs_img, obs_traj_rel, seq_start_end)

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel,
                pred_traj_gt_rel,
                loss_mask,
                mode='raw'))

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), args.clipping_threshold_g
        )
    optimizer_g.step()

    return losses


def check_accuracy(
    args, loader, generator, discriminator, d_loss_fn, limit=False
):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error = []
    f_disp_error = []
    total_traj = 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, obs_img, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             loss_mask, seq_start_end) = batch
            loss_mask = loss_mask[:, args.obs_len:]

            pred_traj_fake_rel = generator(
                obs_traj, obs_img, obs_traj_rel, seq_start_end
            )
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade = displacement_error(pred_traj_fake, pred_traj_gt)
            fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            f_disp_error.append(fde.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj

    generator.train()
    return metrics


def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
    loss_mask
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel


if __name__ == '__main__':
    args = parser.parse_args("")
    main(args)