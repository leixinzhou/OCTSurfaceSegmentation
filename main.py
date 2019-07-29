from torch.utils.data import DataLoader
from dataset import *
import yaml
import argparse
import time
from smooth1D import smooth
from tensorboardX import SummaryWriter
from torch import optim
from torch import nn
import shutil
import os
import matplotlib.pyplot as plt
import network
import sys
sys.path.append('../')
from AugSurfSeg import *
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

# Sample training data. The npy starts with AMD and then Control.
TR_AMD_NB = 187
TR_Control_NB = 79
TR_CASE_NB = TR_AMD_NB + TR_Control_NB
TEST_AMD_NB = 41
TEST_Control_NB = 18
SLICE_per_vol = 60


def save_checkpoint(states,  path, filename='model_best.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint_name = os.path.join(path,  filename)
    torch.save(states, checkpoint_name)

# train


def train(model, criterion, optimizer, input_img_gt, hps):
    model.train()
    D = model(input_img_gt['img'])
    
    # print(D.size(), input_img_gt['gt_g'].size())
    if hps['network'] == "UNet" or hps['network'] == "FCN":
        loss = criterion(D, input_img_gt['gt_g'].squeeze(-1))
    elif hps['network']=="PairNet":
        loss =  criterion(D, input_img_gt['gt_d'])
    elif hps['network']=="SurfNet" or hps['network']=="SurfSegNSBNet":
        loss =  criterion(D, input_img_gt['gt'])
    else:
        raise AttributeError('Network not implemented!')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach().cpu().numpy()
# val


def val(model, criterion, input_img_gt, hps):
    model.eval()
    D = model(input_img_gt['img'])
    # print(output.size(), input_img_gt['gaus_gt'].size())
    if hps['network'] == "UNet" or hps['network'] == "FCN":
        loss = criterion(D, input_img_gt['gt_g'].squeeze(-1))
    elif hps['network']=="PairNet":
        loss =  criterion(D, input_img_gt['gt_d'])
    elif hps['network']=="SurfNet" or hps['network']=="SurfSegNSBNet":
        loss =  criterion(D, input_img_gt['gt'])
    else:
        raise AttributeError('Network not implemented!')

    return  loss.detach().cpu().numpy()
# learn


def learn(model, hps):
    since = time.time()
    writer = SummaryWriter(hps['learning']['checkpoint_path'])
    if torch.cuda.device_count() >= 1:
        # os.environ["CUDA_VISIBLE_DEVICES"] = hps['gpu'])
        model.cuda()
        # model = nn.DataParallel(model)

        # model = nn.DataParallel(model, device_ids=hps['gpu'], output_device=hps['gpu'][0])
    else:
        raise NotImplementedError("CPU version is not implemented!")

    np.random.seed(0)
    AMD_vol_list = np.random.choice(range(TR_AMD_NB), 
                            int(TR_AMD_NB*hps['learning']['data']['tr_ratio']), replace=False)
    Control_vol_list = np.random.choice(range(TR_AMD_NB, TR_CASE_NB), 
                            int(TR_Control_NB*hps['learning']['data']['tr_ratio']), replace=False)
    vol_list = np.concatenate((AMD_vol_list, Control_vol_list))
    print(vol_list)
    aug_dict = {"saltpepper": SaltPepperNoise(sp_ratio=0.05), 
                "Gaussian": AddNoiseGaussian(loc=0, scale=0.1),
                "cropresize": RandomCropResize(crop_ratio=0.9), 
                "circulateud": CirculateUD(),
                "mirrorlr":MirrorLR(), 
                "circulatelr": CirculateLR()}
    rand_aug = RandomApplyTrans(trans_seq=[aug_dict[i] for i in hps['learning']['augmentation']],
                                trans_seq_post=[NormalizeSTD()],
                                trans_seq_pre=[NormalizeSTD()])
    val_aug = RandomApplyTrans(trans_seq=[],
                                trans_seq_post=[NormalizeSTD()],
                                trans_seq_pre=[NormalizeSTD()])

    tr_dataset = OCTDataset(surf=hps['surf'], img_np=hps['learning']['data']['tr_img'],
                            label_np=hps['learning']['data']['tr_gt'],
                            vol_list=vol_list, transforms=rand_aug
                            )
    print(tr_dataset.__len__())
    tr_loader = DataLoader(tr_dataset, shuffle=True,
                           batch_size=hps['learning']['batch_size'], num_workers=0)
    val_dataset = OCTDataset(surf=hps['surf'], img_np=hps['learning']['data']['val_img'],
                            label_np=hps['learning']['data']['val_gt'],
                            transforms=val_aug
                            )
    val_loader = DataLoader(val_dataset, shuffle=False,
                            batch_size=hps['learning']['batch_size'], num_workers=0)

    optimizer = getattr(optim, hps['learning']['optimizer'])(
        [{'params': model.parameters(), 'lr': hps['learning']['lr']}
         ])
    # scheduler = getattr(optim.lr_scheduler,
    #                     hps.learning.scheduler)(optimizer, factor=hps.learning.scheduler_params.factor,
    #                                             patience=hps.learning.scheduler_params.patience,
    #                                             threshold=hps.learning.scheduler_params.threshold,
    #                                             threshold_mode=hps.learning.scheduler_params.threshold_mode)
    try:
        loss_func = getattr(nn, hps['learning']['loss'])()
    except AttributeError:
        raise AttributeError(hps['learning']['loss']+" is not implemented!")
    # criterion_KLD = torch.nn.KLDivLoss()

    epoch_start = 0
    best_loss = hps['learning']['best_loss']

    for epoch in range(epoch_start, hps['learning']['total_iterations']):
        # tr_loss_g = 0
        tr_loss_d = 0
        tr_mb = 0
        print("Epoch: " + str(epoch))
        for step, batch in enumerate(tr_loader):
            batch = {key: value.cuda() for (key, value) in batch.items() }
            m_batch_loss = train(model, loss_func, optimizer, batch, hps)
            # tr_loss_g += m_batch_loss[0]
            tr_loss_d += m_batch_loss
            tr_mb += 1
            print("         mini batch train loss: "+ "%.5e" % m_batch_loss)
        # epoch_tr_loss_g = tr_loss_g / tr_mb
        epoch_tr_loss_d = tr_loss_d / tr_mb
        # writer.add_scalar('data/train_loss_g', epoch_tr_loss_g, epoch)
        writer.add_scalar('data/train_loss_d', epoch_tr_loss_d, epoch)
        
        # print("     tr_loss_g: " + "%.5e" % epoch_tr_loss_g)
        print("     tr_loss_d: " + "%.5e" % epoch_tr_loss_d)
        # scheduler.step(epoch_tr_loss)

        # val_loss_g = 0
        val_loss_d = 0
        val_mb = 0
        for step, batch in enumerate(val_loader):
            batch = {key: value.cuda() for (key, value) in batch.items() }
            m_batch_loss = val(model, loss_func, batch, hps)
            # val_loss_g += m_batch_loss[0]
            val_loss_d += m_batch_loss
            val_mb += 1
            print("         mini batch val loss: "+ "%.5e" % m_batch_loss)
        # epoch_val_loss_g = val_loss_g / val_mb
        epoch_val_loss_d = val_loss_d / val_mb
        # writer.add_scalar('data/val_loss_g', epoch_val_loss_g, epoch)
        writer.add_scalar('data/val_loss_d', epoch_val_loss_d, epoch)
        # print("     val_loss_g: " + "%.5e" % epoch_val_loss_g)
        print("     val_loss_d: " + "%.5e" % epoch_val_loss_d)

        if epoch_val_loss_d < best_loss:
            best_loss = epoch_val_loss_d
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict()
                },
                path=hps['learning']['checkpoint_path'],
            )

    writer.export_scalars_to_json(os.path.join(
        hps['learning']['checkpoint_path'], "all_scalars.json"))
    writer.close()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


def infer(model, hps):
    since = time.time()
    if torch.cuda.device_count() >= 1:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(hps.gpu_nb)
        model.cuda()
        # model = nn.DataParallel(model)

    else:
        raise NotImplementedError("CPU version is not implemented!")
        # print("run in cpu.")
        # model = nn.DataParallel(model)
    
    test_aug = RandomApplyTrans(trans_seq=[],
                                trans_seq_post=[NormalizeSTD()],
                                trans_seq_pre=[NormalizeSTD()])
    test_dataset = OCTDataset(surf=hps['surf'], img_np=hps['test']['data']['img'],
                            label_np=hps['test']['data']['gt'], transforms=test_aug
                            )
    test_loader = DataLoader(test_dataset, shuffle=False,
                            batch_size=hps['test']['batch_size'], num_workers=0)
    
    model.eval()
    pred_list = []
    gt_list = []
    # pred_dummy = []
    for step, batch in enumerate(test_loader):
    #     pred = np.zeros(399, dtype=np.float32)
    #     batch_gt_d = batch['gt_d'].squeeze().detach().cpu().numpy()
    #     batch_gt_d_nsm = batch['gt_d_nsm'].squeeze().detach().cpu().numpy()
        batch_gt = batch['gt'].squeeze().detach().cpu().numpy()
    #     # print(batch_gt_d)
    #     # print(batch_gt)
    #     # break
        batch_img = batch['img'].float().cuda()
        pred_tmp = model(batch_img)
        pred = pred_tmp.squeeze().detach().cpu().numpy()
        pred_list.append(pred)
        gt_list.append(batch_gt)
    #     fig, axes = plt.subplots(4,1)
    #     axes[0].imshow(batch_img.squeeze().detach().cpu().numpy().transpose(), cmap="gray", aspect='auto')
    #     axes[0].plot(batch_gt, 'r', label='gt')
    #     axes[0].legend()
    #     axes[1].plot(pred, 'r', label='diff pred')
    #     axes[1].legend()
    #     axes[2].plot(batch_gt_d_nsm, 'b', label='diff gt')
    #     axes[2].legend()
    #     axes[3].plot(batch_gt_d, 'b', label='diff gt smooth')
    #     axes[3].legend()
    #     # pred = cartpolar.gt2cart(pred)
        
    #     fig.savefig(pred_dir)
    #     plt.close()
        # pred_l1.append(np.mean(np.abs(batch_gt-pred)))
    #     pred_dummy.append(np.mean(np.abs(batch_gt_d)))
    pred = np.concatenate(pred_list)
    gt = np.concatenate(gt_list)
    if not os.path.isdir(hps['test']['pred_dir']):
        os.mkdir(hps['test']['pred_dir'])
    pred_dir = os.path.join(hps['test']['pred_dir'],"pred.npy")
    pred_stat_dir = os.path.join(hps['test']['pred_dir'],"pred_stat.txt")
    np.save(pred_dir, pred)
    error = np.abs(pred - gt)
    error_mean = [np.mean(error[i*SLICE_per_vol:(i+1):SLICE_per_vol,]) for i in range(TEST_AMD_NB+TEST_Control_NB)]
    amd_mean = np.mean(error[:TEST_AMD_NB,])
    amd_std = np.std(error[:TEST_AMD_NB, ])
    control_mean = np.mean(error[TEST_AMD_NB:,])
    control_std = np.std(error[TEST_AMD_NB:, ])
    print("AMD", amd_mean, amd_std)
    print("Control", control_mean, control_std)
    np.savetxt(pred_stat_dir, [amd_mean, amd_std, control_mean, control_std])
    #     # np.savetxt(pred_dir, pred, delimiter=',')
    # print("Test done!")
    # pred_l1_mean = np.mean(np.array(pred_l1))
    # pred_l1_std = np.std(np.array(pred_l1))
    # dummy_l1_mean = np.mean(np.array(pred_dummy))
    # dummy_l1_std = np.std(np.array(pred_dummy))
    # print("test L1: ", "%.5e" % pred_l1_mean)
    # print("test L1 std: ", "%.5e" % pred_l1_std)
    # print("test dummy L1: ", "%.5e" % dummy_l1_mean)
    # print("test dummy L1 std: ", "%.5e" % dummy_l1_std)
    # np.savetxt(os.path.join(hps['test']['pred_dir'], "results.txt"), [pred_l1_mean, pred_l1_std, dummy_l1_mean, dummy_l1_std])

    # print("Test done!")
    # time_elapsed = time.time() - since
    # print('Test complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))



def main():
    # read configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hyperparams', default='./para/hparas_unet.json',
                        type=str, metavar='FILE.PATH',
                        help='path to hyperparameters setting file (default: ./para/hparas_unet.json)')

    args = parser.parse_args()
    try:
        with open(args.hyperparams, "r") as config_file:
            hps = yaml.load(config_file)
    except IOError:
        print('Couldn\'t read hyperparameter setting file')
    if hps['network'] == "UNet" or hps['network'] == "FCN":
        model = getattr(network, hps['network'])(num_classes=1, in_channels=1, depth=hps['unary_network']['depth'],
                 start_filts=hps['unary_network']['start_filters'], up_mode=hps['unary_network']['up_mode'])
        if os.path.isfile(hps['unary_network']['resume_path']):
            print('loading unary network checkpoint: {}'.format(hps['unary_network']['resume_path']))
            checkpoint = torch.load(hps['unary_network']['resume_path'])
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded unary checkpoint (epoch {})"
                .format(checkpoint['epoch']))
        else:
            print("=> no unary network checkpoint found at '{}'".format(hps['unary_network']['resume_path']))

    elif hps['network']=="PairNet":
        model = PairNet(num_classes=1, in_channels=1, depth=hps['pair_network']['depth'],
                            start_filts=hps['pair_network']['start_filters'], up_mode=hps['pair_network']['up_mode'], 
                            col_len=hps['pair_network']['col_len'], fc_inter=hps['pair_network']['fc_inter'], 
                            left_nbs=hps['pair_network']['left_nbs'])
        if os.path.isfile(hps['pair_network']['resume_path']):
            print('loading pair network checkpoint: {}'.format(hps['pair_network']['resume_path']))
            checkpoint = torch.load(hps['pair_network']['resume_path'])
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})"
                .format(checkpoint['epoch']))
        else:
            print("=> no pair network checkpoint found at '{}'".format(hps['pair_network']['resume_path']))

    elif hps['network']=="SurfSegNet":
        model_u = getattr(network, hps['network'])(num_classes=1, in_channels=1, depth=hps['unary_network']['depth'],
                 start_filts=hps['unary_network']['start_filters'], up_mode=hps['unary_network']['up_mode'])
        if os.path.isfile(hps['surf_net']["pair_pretrain_path"]):
            model_p = PairNet(num_classes=1, in_channels=1, depth=hps['pair_network']['depth'],
                            start_filts=hps['pair_network']['start_filters'], up_mode=hps['pair_network']['up_mode'], 
                            col_len=hps['pair_network']['col_len'], fc_inter=hps['pair_network']['fc_inter'], 
                            left_nbs=hps['pair_network']['left_nbs'])
        else:
            model_p = None
        model = SurfSegNet(unary_model=model_u, hps=hps, pair_model=model_p)
        model.load_wt()
    elif hps['network']=="SurfSegNSBNet":
        model_u = getattr(network, hps['surf_net']['unary_network'])(num_classes=1, in_channels=1, depth=hps['unary_network']['depth'],
                 start_filts=hps['unary_network']['start_filters'], up_mode=hps['unary_network']['up_mode'])
        model = getattr(network, hps['network'])(unary_model=model_u, hps=hps)
        model.load_wt()
    else:
        raise AttributeError('Network not implemented!')

    
    if hps['test']['mode']:
        infer(model, hps)
    else:
        try:
            learn(model, hps)
        except KeyboardInterrupt:
            torch.save(model.state_dict(), os.path.join(
                hps['learning']['checkpoint_path'], 'INTERRUPTED.pth'))
            print('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)



if __name__ == '__main__':
    main()
