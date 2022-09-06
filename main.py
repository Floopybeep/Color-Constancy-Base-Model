from network import *
from functions import *
from network import *

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np

import statistics
import time
import datetime
import socket
import matplotlib.pyplot as plt

gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

best_average = 2
new_average = False


def train(epoch):
    epoch_loss = 0
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input1, input2, input3, gt = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])
        # input1, gt = Variable(batch[0]), Variable(batch[1])

        if cuda:
            input1 = input1.cuda(gpus_list[0])
            input2 = input2.cuda(gpus_list[0])
            input3 = input3.cuda(gpus_list[0])
            gt = gt.cuda(gpus_list[0])

        optimizer.zero_grad()
        t0 = time.time()

        prediction = model(input1, input2, input3)[0]
        # prediction = model(input1)[0]
        # prediction = prediction/torch.norm(prediction)

        # torch.autograd.set_detect_anomaly(True)

        loss = criterion(prediction, gt)
        # t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
        t1 = time.time()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration,
                                                                                 len(training_data_loader), loss.data,
                                                                                 (t1 - t0)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


    train_loss[epoch - 1] = epoch_loss / (len(training_data_loader))


def eval(epoch):
    avg_angle = 0
    model.eval()
    errorlist = []
    closed_errorlist = []
    ambient_errorlist = []
    average = 0
    count = 1

    for batch in testing_data_loader:
        with torch.no_grad():
            input1, input2, input3, gt, index = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), batch[4]
            # input1, gt, index = Variable(batch[0]), Variable(batch[1]), batch[2]
        if cuda:
            input1 = input1.cuda(gpus_list[0])
            input2 = input2.cuda(gpus_list[0])
            input3 = input3.cuda(gpus_list[0])
            gt = gt.cuda(gpus_list[0])

        t0 = time.time()

        with torch.no_grad():
            model_output = model(input1, input2, input3)

        prediction = model_output[0]   # illum pred
        confidence = model_output[1]   # confmap
        local = model_output[2]        # RGBmap

        # prediction[0] = prediction[0] / (torch.norm(prediction[0]) + 1e-6)

        angle_error = criterion(prediction, gt)
        avg_angle += angle_error
        name = count

        if opt.evalmode:
            prediction = prediction / torch.norm(prediction)
            (b, _, h, w) = confidence.shape
            out_img = torch.zeros(1, h * 16, w * 16)
            out_rgb = torch.zeros(1, 3, h * 16, w * 16)

            for i in range(h * 16):
                for j in range(w * 16):
                    # out_img[:,0,i,j] = (confidence[:,int(i/16),int(j/16)])*local[:,0,int(i/16),int(j/16)]
                    # out_img[:,1,i,j] = (confidence[:,int(i/16),int(j/16)])*local[:,1,int(i/16),int(j/16)]
                    # out_img[:,2,i,j] = (confidence[:,int(i/16),int(j/16)])*local[:,2,int(i/16),int(j/16)]
                    out_img[0, i, j] = (confidence[0, 0, int(i / 16), int(j / 16)])
                    out_rgb[0, 0, i, j] = (local[0, 0, int(i / 16), int(j / 16)])
                    out_rgb[0, 1, i, j] = (local[0, 1, int(i / 16), int(j / 16)])
                    out_rgb[0, 2, i, j] = (local[0, 2, int(i / 16), int(j / 16)])

            input1 = torch.squeeze(input1)
            input1 = input1.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            prediction = prediction.flatten()

            restoredimg = np.zeros((3, 180, 240))
            restoredimg[0] = input1[0] / prediction[0]
            restoredimg[1] = input1[1] / prediction[1]
            restoredimg[2] = input1[2] / prediction[2]
            restoredimg = torch.from_numpy(restoredimg)

            image_name_str = str(name) + '.png'

            save_img(out_rgb, 'rgb_map', image_name_str)
            save_img(out_img, 'confidence_map', image_name_str)
            save_img(restoredimg, 'restored_image', image_name_str)

            log('Test loss for %d = %.4f \n' % (name, angle_error.item()), logfile2)
            errorlist.append(angle_error.item())
            if name < 26:
                closed_errorlist.append(angle_error.item())
            else:
                ambient_errorlist.append(angle_error.item())
        # save_img((restoredimg_confmap), 'restored_image_confidence', f'{index}.png')

        # print(prediction)

        # print("===> Processing: %d || Timer: %.4f sec. angular error : %.4f" % ( name, (t1 - t0), angle_error))

        average = avg_angle / len(testing_data_loader)
        count += 1

    print("===> Processing Done, Average Angular error : %.4f" % (average))
    log('Epoch[%d] : Test Avg loss = %.4f \n' % (epoch, average), logfile)
    if opt.evalmode:
        log('Test Complete! Test Avg loss = %.4f \n' % average, logfile2)
        log('Median: %.4f / Mean: %.4f / Best-25: %.4f / Worst-25: %.4f / Closed: %.4f / Ambient: %.4f \n'
            % (statistics.median(errorlist), average, statistics.quantiles(errorlist, n=4)[0], statistics.quantiles(errorlist, n=4)[2],
               statistics.mean(closed_errorlist), statistics.mean(ambient_errorlist)), logfile2)
        log('=================================================\n', logfile2)


def outlier_eval(epoch):
    model.eval()
    average = 0

    for batch in outlier_data_loader:
        with torch.no_grad():
            input1, input2, input3, gt, index = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), batch[4]
            # input1, gt, index = Variable(batch[0]), Variable(batch[1]), batch[2]
        if cuda:
            input1 = input1.cuda(gpus_list[0])
            input2 = input2.cuda(gpus_list[0])
            input3 = input3.cuda(gpus_list[0])
            gt = gt.cuda(gpus_list[0])

        with torch.no_grad():
            prediction = model(input1, input2, input3)[0]

        angle_error = criterion(prediction, gt)
        average += angle_error

        print("===> Processing Done, Average Outlier Angular error : %.4f" % (average))
        log('Epoch[%d] Outlier Loss = %.4f \n' % (epoch, average), logfile3)

    global best_average, new_average

    if (epoch_loss/len(outlier_data_loader)) < best_average:
        new_average = True
        best_average = epoch_loss/len(outlier_data_loader)


def checkpoint(epoch):
    model_out_path = opt.save_folder + hostname + "_epoch_{}.pth".format(epoch)
    if not os.path.exists(opt.save_folder):
        os.makedirs(opt.save_folder)
    state = {
        'net': model.state_dict(),
        # 'loss': epoch_loss,
        'epoch': epoch,
    }
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':

    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading training datasets')
    train_set = get_training_set(opt.data_dir, opt.data_int_dir)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)
    print('===> Loading eval datasets')
    test_set = get_eval_set(opt.input_dir, opt.input_int_dir)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    if opt.outliermode:
        print('===> Loading outlier datasets')
        outlier_set = get_outlier_set(opt.input_dir, opt.input_int_dir, opt.outlier_list)
        outlier_data_loader = DataLoader(dataset=outlier_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    print('===> Building model ', opt.model_type)
    if opt.model_type == 'result':
        model = Result()

    model = torch.nn.DataParallel(model, device_ids=gpus_list)
    criterion = Aloss()

    if cuda:
        model = model.cuda(gpus_list[0])
        criterion = criterion.cuda(gpus_list[0])

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    logfile = opt.trainlogfilepath
    logfile2 = opt.evallogfilepath
    if opt.outliermode:
        logfile3 = opt.outlierfilepath

    if opt.resume:  # resume from check point, load once-trained data
        # Load checkpoint
        print('==> Resuming from checkpoint..')
        checkpoint_load = torch.load(checkpoint_name2)
        model.load_state_dict(checkpoint_load['net'])
        # best_loss = checkpoint['loss']
        start_epoch = checkpoint_load['epoch']

    train_loss = torch.zeros(opt.nEpochs)

    if not opt.evalmode:
        for epoch in range(opt.start_iter, opt.nEpochs + 1):
            train(epoch)
            if opt.outliermode:
                outlier_eval(epoch)

            if new_average is True:
                new_average = False
                checkpoint(epoch)
                eval(epoch)

            # learning rate is decayed by a factor of 10 every half of total epochs
            if epoch % (opt.nEpochs / 4) == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10.0
                print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

            if epoch == (opt.nEpochs-1):
                print_graph(train_loss, opt.save_folder)

            if epoch % opt.snapshots == 0 and new_average is False:
                checkpoint(epoch)
                eval(epoch)

        tempstr = 'Experiment finished, Current time: ' + str(datetime.datetime.now()) + '\n'
        log(tempstr, logfile)
        log('=============================================== \n', logfile)
        log('=============================================== \n', logfile3)

    if opt.evalmode:
        checkpoint_load_eval = torch.load(opt.evalweightpath)
        model.load_state_dict(checkpoint_load_eval['net'])
        eval(0)
