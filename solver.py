from __future__ import division
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import numpy as np
import math
import scipy.misc
import progressbar
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pytorch_ssim
import os
import time
import skimage.io as sio
import skimage.color as sc


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss


def quantize(img, rgb_range):
    return img.mul(255 / rgb_range).clamp(0, 255).round()


def rgb2ycbcrT(rgb):
    rgb = rgb.numpy().transpose(1, 2, 0)
    yCbCr = sc.rgb2ycbcr(rgb) / 255

    return torch.Tensor(yCbCr[:, :, 0])


def chop_forward(x, model, scale, shave=10, min_size=8000, n_GPUs=1):
    n_GPUs = min(n_GPUs, 4)
    b, num, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, :, 0:h_size, 0:w_size],
        x[:, :, :, 0:h_size, (w - w_size):w],
        x[:, :, :, (h - h_size):h, 0:w_size],
        x[:, :, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, n_GPUs):
            input_batch = torch.cat(inputlist[i:(i + n_GPUs)], dim=0)
            output_batch, _ = model(input_batch)
            outputlist.extend(output_batch.chunk(n_GPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch, model, scale, shave, min_size, n_GPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = Variable(x.data.new(b, c, h, w), volatile=True)
    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


def forward_x8(lr, forward_function=None):
    def _transform(v, op):
        v = v.float()

        v2np = v.data.cpu().numpy()
        # print(v2np.shape)
        if op == 'v':
            tfnp = v2np[:, :, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 2, 4, 3)).copy()

        ret = Variable(torch.Tensor(tfnp).cuda())
        # ret = ret.half()

        return ret

    def _transform_back(v, op):

        if op == 'v':
            tfnp = v[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v.transpose((0, 1, 3, 2)).copy()

        return tfnp

    x = [lr]
    for tf in 'v', 'h': x.extend([_transform(_x, tf) for _x in x])

    list_r = []
    for k in range(len(x)):
        z = x[k]
        r, _ = forward_function(z)
        r = r.data.cpu().numpy()
        if k % 4 > 1:
            r = _transform_back(r, 'h')
        if (k % 4) % 2 == 1:
            r = _transform_back(r, 'v')
        list_r.append(r)
    y = np.sum(list_r, axis=0) / 4.0

    y = Variable(torch.Tensor(y).cuda())
    if len(y) == 1: y = y[0]
    return y


class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training super resolution
    The Solver accepts both training and validation data label so it can
    periodically check the PSNR on training

    To train a model, you will first construct a Solver instance, pass the model,
    datasets, and various option (optimizer, loss_fn, batch_size, etc) to the
    constructor.

    After train() method is called. The best model is saved into 'check_point' dir, which is used
    for the testing time.

    For statistics, 'loss' history, 'avr_train_psnr' history
    are also saved.
    """

    def __init__(self, model, check_point, model_name, **kwargs):
        """
        Construct a new Solver instance

        Required arguments
        - model: a torch nn module describe the neural network architecture
        - check_point: save trained model for testing for finetuning

        Optional arguments:
        - num_epochs: number of epochs to run during training
        - batch_size: batch size for train phase
        - optimizer: update rule for model parameters
        - loss_fn: loss function for the model
        - fine_tune: fine tune the model in check_point dir instead of training
                     from scratch
        - verbose: print training information
        - print_every: period of statistics printing
        """
        self.model = model
        self.model_name = model_name
        self.check_point = check_point
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 128)
        self.learning_rate = kwargs.pop('learning_rate', 1e-4)
        self.model = nn.DataParallel(self.model)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.learning_rate, weight_decay=1e-6)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.5)
        self.loss_fn = kwargs.pop('loss_fn', nn.MSELoss())
        self.fine_tune = kwargs.pop('fine_tune', False)
        self.verbose = kwargs.pop('verbose', False)
        self.print_every = kwargs.pop('print_every', 10)

        self._reset()

    def _reset(self):
        """ Initialize some book-keeping variable, dont call it manually"""
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model = self.model.cuda()
        self.hist_train_psnr = []
        self.hist_val_psnr = []
        self.hist_loss = []

    def _epoch_step(self, dataset, epoch):
        """ Perform 1 training 'epoch' on the 'dataset'"""
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=64)

        num_batchs = len(dataset) // self.batch_size

        # observe the training progress
        if self.verbose:
            bar = progressbar.ProgressBar(max_value=num_batchs)

        running_loss = 0
        for i, sample in enumerate(dataloader):
            input_batch, label_batch = sample['lr'], sample['hr']
            # Wrap with torch Variable
            input_batch, label_batch = self._wrap_variable(input_batch,
                                                           label_batch,
                                                           self.use_gpu)
            # zero the grad
            self.optimizer.zero_grad()

            # Forward

            if self.model_name in ['TDAN']:
                output_batch, lrs = self.model(input_batch)
                num = input_batch.size(1)
                center = num // 2
                x = input_batch[:, center, :, :, :].unsqueeze(1).repeat(1, num, 1, 1, 1)
                loss = self.loss_fn(output_batch, label_batch) + 0.25 * self.loss_fn(lrs, x)
            else:
                output_batch = self.model(input_batch)
                loss = self.loss_fn(output_batch, label_batch)

            running_loss += loss.data[0]

            # Backward + update
            loss.backward()
            #nn.utils.clip_grad_norm(self.model.parameters(), 0.4)
            self.optimizer.step()

            if self.verbose:
                bar.update(i, force=True)

        average_loss = running_loss / num_batchs
        self.hist_loss.append(average_loss)
        if self.verbose:
            print('Epoch  %5d, loss %.5f' \
                  % (epoch, average_loss))

    def _wrap_variable(self, input_batch, label_batch, use_gpu):
        if use_gpu:
            input_batch, label_batch = (Variable(input_batch.cuda()),
                                        Variable(label_batch.cuda()))
        else:
            input_batch, label_batch = (Variable(input_batch),
                                        Variable(label_batch))
        return input_batch, label_batch

    def _comput_PSNR(self, input, target):
        """Compute PSNR between two image array and return the psnr summation"""
        shave = 4
        ch, h, w = input.size()
        input_Y = rgb2ycbcrT(input.cpu())
        target_Y = rgb2ycbcrT(target.cpu())
        diff = (input_Y - target_Y).view(1, h, w)

        diff = diff[:, shave:(h - shave), shave:(w - shave)]
        mse = diff.pow(2).mean()
        psnr = -10 * np.log10(mse)
        return psnr

    def _check_PSNR(self, dataset, is_test=False):
        """
        Get the output of model with the input being 'dataset' then
        compute the PSNR between output and label.

        if 'is_test' is True, psnr and output of each image is also
        return for statistics and generate output image at test phase
        """

        # process one image per iter for test phase
        if is_test:
            batch_size = 1
        else:
            batch_size = 1  # self.batch_size

        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=1)

        avr_psnr = 0
        avr_ssim = 0

        # book keeping variables for test phase
        psnrs = []  # psnr for each image
        ssims = []  # ssim for each image
        proc_time = []  # processing time
        outputs = []  # output for each image
        names = []

        for batch, sample in enumerate(dataloader):
            input_batch, label_batch, name = sample['lr'], sample['hr'], sample['im_name']

            # Wrap with torch Variable
            input_batch, label_batch = self._wrap_variable(input_batch,
                                                           label_batch,
                                                           self.use_gpu)

            if is_test:
                start = time.time()
                if self.model_name in ['TDAN']:
                    output_batch = chop_forward(input_batch, self.model, 4)
                    #output_batch = chop_forward(input_batch, self.model, 4)
                    #output_batch = forward_x8(input_batch, self.model).unsqueeze(0)
                    #print(output_batch.size())
                    # _, lrs = self.model(input_batch)
                    # output_batch = lrs[:, -1, :, :, :]
                else:
                    output_batch = self.model(input_batch)
                elapsed_time = time.time() - start
            else:
                if self.model_name in ['TDAN']:
                    #output_batch, _ = self.model(input_batch)
                    output_batch = chop_forward(input_batch, self.model, 4)
                else:
                    output_batch = self.model(input_batch)
            # ssim is calculated with the normalize (range [0, 1]) image
            ssim = pytorch_ssim.ssim(output_batch + 0.5, label_batch + 0.5, size_average=False)
            ssim = torch.sum(ssim.data)
            avr_ssim += ssim

            # calculate PSRN
            output = output_batch.data
            label = label_batch.data

            output = (output + 0.5) * 255
            label = (label + 0.5) * 255

            output = quantize(output, 255)
            label = quantize(label, 255)
            # diff = input - target

            output = output.squeeze(dim=0)
            label = label.squeeze(dim=0)

            psnr = self._comput_PSNR(output / 255.0, label / 255.0)
            # print(psnr)
            avr_psnr += psnr

            # save psnrs and outputs for statistics and generate image at test time
            if is_test:
                psnrs.append(psnr)
                ssims.append(ssim)
                proc_time.append(elapsed_time)
                np_output = output.cpu().numpy()
                outputs.append(np_output)
                names.append(name)

        epoch_size = len(dataset)
        avr_psnr /= epoch_size
        avr_ssim /= epoch_size
        stats = (psnrs, ssims, proc_time)

        return avr_psnr, avr_ssim, stats, outputs, names

    def train(self, train_dataset, val_dataset):
        """
        Train the 'train_dataset',
        if 'fine_tune' is True, we finetune the model under 'check_point' dir
        instead of training from scratch.

        The best model is save under checkpoint which is used
        for test phase or finetuning
        """

        # check fine_tuning option
        model_path = os.path.join(self.check_point, 'model_gaussian.pt')
        if self.fine_tune and not os.path.exists(model_path):
            raise Exception('Cannot find %s.' % model_path)
        elif self.fine_tune and os.path.exists(model_path):
            if self.verbose:
                print('Loading %s for finetuning.' % model_path)
            self.model = torch.load(model_path)
            '''
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            model_dict = self.model.state_dict()
            net_dict = net.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in net_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.model.load_state_dict(model_dict)
            '''
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # capture best model
        best_val_psnr = -1
        best_psnr = -1
        best_model_state = self.model.state_dict()

        with open(os.path.join(self.check_point, 'PSNR' + '.txt'), 'w') as f:
            # Train the model
            for epoch in range(self.num_epochs):
                self._epoch_step(train_dataset, epoch)
                self.scheduler.step()

                if epoch % 10 == 0:
                    if self.verbose:
                        print('Computing PSNR...')

                    # capture running PSNR on train and val dataset
                    train_psnr, train_ssim, _, _, _ = self._check_PSNR(val_dataset)
                    self.hist_train_psnr.append(train_psnr)

                    f.write('epoch%d:\t%.3f\n' % (epoch, train_psnr))

                    if self.verbose:
                        print('Average train PSNR:%.3fdB average ssim: %.3f' % (train_psnr, train_ssim))
                        print('')
                    if best_psnr < train_psnr:
                        best_psnr = train_psnr
                        # write the model to hard-disk for testing
                        if not os.path.exists(self.check_point):
                            os.makedirs(self.check_point)
                        model_path = os.path.join(self.check_point, 'model_gaussian.pt')
                        torch.save(self.model, model_path)
                        print(' Best average psnr: %.3f' % (best_psnr))
                        print('')

    def test(self, dataset):
        """
        Load the model stored in train_model.pt from training phase,
        then return the average PNSR on test samples.
        """
        model_path = os.path.join(self.check_point, 'model_gaussian.pt')
        if not os.path.exists(model_path):
            raise Exception('Cannot find %s.' % model_path)

        self.model = torch.load(model_path)
        print(self.model)
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(1.0 * params / (1000 * 1000))
        _, _, stats, outputs, names = self._check_PSNR(dataset, is_test=True)
        return stats, outputs, names

    def align_gen(self, dataset, is_test=False):
        """
        Get the output of model with the input being 'dataset' then
        compute the PSNR between output and label.

        if 'is_test' is True, psnr and output of each image is also
        return for statistics and generate output image at test phase
        """

        # process one image per iter for test phase
        if is_test:
            batch_size = 1
        else:
            batch_size = 1  # self.batch_size

        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=1)

        # book keeping variables for test phase
        psnrs = []  # psnr for each image
        ssims = []  # ssim for each image
        proc_time = []  # processing time
        outputs = []  # output for each image
        names = []

        for batch, sample in enumerate(dataloader):
            input_batch, label_batch, name = sample['lr'], sample['hr'], sample['im_name']

            # Wrap with torch Variable
            input_batch, label_batch = self._wrap_variable(input_batch,
                                                           label_batch,
                                                           self.use_gpu)

            if is_test:
                start = time.time()
                if self.model_name in ['TDAN']:
                    # output_batch, _ = self.model(input_batch)
                    # output_batch = chop_forward(input_batch, self.model, 4)
                    output_batch, _ = self.model(input_batch)
                # output_batch = lrs[:, -1, :, :, :]
                else:
                    output_batch = self.model(input_batch)
                elapsed_time = time.time() - start
            else:
                if self.model_name in ['TDAN']:
                    # output_batch, _ = self.model(input_batch)
                    output_batch = chop_forward(input_batch, self.model, 4)
                else:
                    output_batch = self.model(input_batch)

            # calculate PSRN
            output = output_batch.data
            label = label_batch.data

            output = (output + 0.5) * 255
            label = (label + 0.5) * 255

            output = quantize(output, 255)
            label = quantize(label, 255)
            # diff = input - target

            output = output.squeeze(dim=0)
            label = label.squeeze(dim=0)

            # save psnrs and outputs for statistics and generate image at test time
            if is_test:
                proc_time.append(elapsed_time)
                np_output = output.cpu().numpy()
                outputs.append(np_output)
                names.append(name)

        epoch_size = len(dataset)
        stats = (psnrs, ssims, proc_time)
        avr_psnr = 0
        avr_ssim = 0

        return avr_psnr, avr_ssim, stats, outputs, names

    def ta_gen(self, dataset):
        """
        generate temporal alignment results
        :param dataset:
        :return:
        """

        model_path = os.path.join(self.check_point, 'model_gaussian.pt')
        if not os.path.exists(model_path):
            raise Exception('Cannot find %s.' % model_path)

        self.model = torch.load(model_path)
        _, _, stats, outputs, names = self.align_gen(dataset, is_test=True)
        return stats, outputs, names


