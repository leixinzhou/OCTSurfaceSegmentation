from torch.utils.data import DataLoader
from dataset import *
import unet
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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152


def save_checkpoint(states,  path, filename='model_best.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint_name = os.path.join(path,  filename)
    torch.save(states, checkpoint_name)

# train


def train(model, criterion, optimizer, input_img_gt, hps):
    model.train()
    D = model(input_img_gt['img'])
    D = D.squeeze(1)
    criterion_l1 = nn.L1Loss()
    # print(output.size(), input_img_gt['gaus_gt'].size())
    if hps['learning']['loss'] == "KLDivLoss":
        loss = criterion(output, input_img_gt['gaus_gt'])
    elif hps['learning']['loss'] == "MSELoss" or hps['learning']['loss'] == "L1Loss":
        loss =  criterion(D, input_img_gt['gt_d'])
        # loss_l1_g = criterion_l1(G, input_img_gt['gt'])
        loss_l1_d = criterion_l1(D, input_img_gt['gt_d'])
    else:
        raise AttributeError('Loss not implemented!')
    # watch l1

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss_l1_d.detach().cpu().numpy()
# val


def val(model, criterion, input_img_gt, hps):
    model.eval()
    D = model(input_img_gt['img'])
    D = D.squeeze(1)
    criterion_l1 = nn.L1Loss()

    if hps['learning']['loss'] == "KLDivLoss":
        loss = criterion(output, input_img_gt['gaus_gt'])
    elif hps['learning']['loss'] == "MSELoss" or hps['learning']['loss'] == "L1Loss":
        loss = criterion(D, input_img_gt['gt_d'])
        # loss_l1_g = criterion_l1(G, input_img_gt['gt'])
        loss_l1_d = criterion_l1(D, input_img_gt['gt_d'])
    else:
        raise AttributeError('Loss not implemented!')

    return  loss_l1_d.detach().cpu().numpy()
# learn


def learn(model, hps):
    since = time.time()
    writer = SummaryWriter(hps['learning']['checkpoint_path'])
    if torch.cuda.device_count() >= 1:
        # os.environ["CUDA_VISIBLE_DEVICES"] = hps['gpu'])
        model.cuda()
        model = nn.DataParallel(model)

        # model = nn.DataParallel(model, device_ids=hps['gpu'], output_device=hps['gpu'][0])
    else:
        raise NotImplementedError("CPU version is not implemented!")

    tr_dataset = OCTDataset(surf=hps['surf'], img_np=hps['learning']['data']['tr_img'],
                            label_np=hps['learning']['data']['tr_gt'],
                            g_label_np=None
                            )
    tr_loader = DataLoader(tr_dataset, shuffle=True,
                           batch_size=hps['learning']['batch_size'], num_workers=0)
    val_dataset = OCTDataset(surf=hps['surf'], img_np=hps['learning']['data']['val_img'],
                            label_np=hps['learning']['data']['val_gt'],
                            g_label_np=None
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

    if os.path.isfile(hps['learning']['resume_path']):
        print('loading checkpoint: {}'.format(hps['learning']['resume_path']))
        checkpoint = torch.load(hps['learning']['resume_path'])
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(hps['learning']['resume_path']))

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
        model = nn.DataParallel(model)

    else:
        raise NotImplementedError("CPU version is not implemented!")
        # print("run in cpu.")
        # model = nn.DataParallel(model)
    test_dataset = OCTDataset(surf=hps['surf'], img_np=hps['test']['data']['img'],
                            label_np=hps['test']['data']['gt']
                            )
    test_loader = DataLoader(test_dataset, shuffle=False,
                            batch_size=hps['test']['batch_size'], num_workers=0)
    
    if os.path.isfile(hps['test']['resume_path']):
        print('loading checkpoint: {}'.format(hps['test']['resume_path']))

        checkpoint = torch.load(hps['test']['resume_path'])
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(hps['test']['resume_path']))
    model.eval()
    pred_l1 = []
    pred_dummy = []
    for step, batch in enumerate(test_loader):
        pred = np.zeros(399, dtype=np.float32)
        batch_gt_d = batch['gt_d'].squeeze().detach().cpu().numpy()
        batch_gt_d_nsm = batch['gt_d_nsm'].squeeze().detach().cpu().numpy()
        batch_gt = batch['gt'].squeeze().detach().cpu().numpy()
        # print(batch_gt_d)
        # print(batch_gt)
        # break
        batch_img = batch['img'].float().cuda()
        pred_tmp = model(batch_img)
        pred = pred_tmp.squeeze().detach().cpu().numpy()

        fig, axes = plt.subplots(4,1)
        axes[0].imshow(batch_img.squeeze().detach().cpu().numpy().transpose(), cmap="gray", aspect='auto')
        axes[0].plot(batch_gt, 'r', label='gt')
        axes[0].legend()
        axes[1].plot(pred, 'r', label='diff pred')
        axes[1].legend()
        axes[2].plot(batch_gt_d_nsm, 'b', label='diff gt')
        axes[2].legend()
        axes[3].plot(batch_gt_d, 'b', label='diff gt smooth')
        axes[3].legend()
        # pred = cartpolar.gt2cart(pred)
        if not os.path.isdir(hps['test']['pred_dir']):
            os.mkdir(hps['test']['pred_dir'])
        pred_dir = os.path.join(hps['test']['pred_dir'],"_%d.png" % step)
        fig.savefig(pred_dir)
        plt.close()
        pred_l1.append(np.mean(np.abs(batch_gt_d-pred)))
        pred_dummy.append(np.mean(np.abs(batch_gt_d)))
        # pred = np.transpose(np.stack(pred, axis=0))
       
        # np.savetxt(pred_dir, pred, delimiter=',')
    print("Test done!")
    pred_l1_mean = np.mean(np.array(pred_l1))
    pred_l1_std = np.std(np.array(pred_l1))
    dummy_l1_mean = np.mean(np.array(pred_dummy))
    dummy_l1_std = np.std(np.array(pred_dummy))
    print("test L1: ", "%.5e" % pred_l1_mean)
    print("test L1 std: ", "%.5e" % pred_l1_std)
    print("test dummy L1: ", "%.5e" % dummy_l1_mean)
    print("test dummy L1 std: ", "%.5e" % dummy_l1_std)
    np.savetxt(os.path.join(hps['test']['pred_dir'], "results.txt"), [pred_l1_mean, pred_l1_std, dummy_l1_mean, dummy_l1_std])

    print("Test done!")
    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))



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
    net = getattr(unet, hps['network']['name'])(num_classes=1, in_channels=1,
                 up_mode='transpose', col_len=512, col_nb=400, fc_inter=hps['network']['fc_inter'], depth=hps['network']['depth'],
                  start_filts=hps['network']['start_filts'], left_nbs=hps['left_nbs'])
    if hps['test']['mode']:
        infer(net, hps)
    else:
        try:
            learn(net, hps)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), os.path.join(
                hps['learning']['checkpoint_path'], 'INTERRUPTED.pth'))
            print('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)



if __name__ == '__main__':
    main()
