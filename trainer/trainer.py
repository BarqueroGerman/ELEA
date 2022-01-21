import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None, validation_frequency=1):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None or validation_frequency > 0
        self.validation_frequency = 0 if not self.do_validation else validation_frequency
        assert round(validation_frequency) == validation_frequency
        self.lr_scheduler = lr_scheduler
        self.log_step = max(1, int(self.len_epoch / 10))#int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        times_validated = 0
        validation_steps = sorted([round(i * self.len_epoch / self.validation_frequency) for i in range(1, self.validation_frequency + 1)]) # the last on
        log = None

        torch.autograd.set_detect_anomaly(True)
        self.model.train()
        self.train_metrics.reset()
        """
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=1,
                active=1,
                repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/test'),
            with_stack=False
        ) as profiler:
    """
        for batch_idx, batch in enumerate(self.data_loader):
            self.model.train()
            data, target = batch[0].to(self.device), batch[1].to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data, target)

            loss = self.criterion(output, target)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1) # gradient clipping
            self.optimizer.step()
            #profiler.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            # we run validation every X epochs (where X can be decimal)
            if self.do_validation and batch_idx + 1 in validation_steps:
                print(f"Running validation at {batch_idx}...")
                log = {}
                log.update(**{'val_'+k : v for k, v in list(self._valid_epoch(epoch, batch_idx).items())})
                times_validated += 1

            if batch_idx == self.len_epoch:
                break

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            #    self._save_checkpoint(epoch)

        if log is None:
            log = self.train_metrics.result()
        else:
            train_log = self.train_metrics.result()
            log.update(**{k : v for k, v in list(train_log.items())})

        # commented because it is done inside the loop
        #if self.do_validation:
        #    val_log = self._valid_epoch(epoch)
        #    log.update(**{'val_'+k : v for k, v in val_log.items()})
        return log

    def _valid_epoch(self, epoch, step):#epoch_frequency, times_validated):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :param step: Integer, training steps performed within that epoch
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        av_loss = 0
        av_mets = {}
        for met in self.metric_ftns:
            av_mets[met.__name__] = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                data, target = batch[0].to(self.device), batch[1].to(self.device)

                output = self.model(data, target)
                
                loss = self.criterion(output, target)
                av_loss += loss.item() / len(self.valid_data_loader)

                #self.writer.set_step((epoch - 1) * len(self.valid_data_loader) * epoch_frequency + len(self.valid_data_loader) * times_validated + batch_idx, 'valid')
                #self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    #self.valid_metrics.update(met.__name__, met(output, target))
                    av_mets[met.__name__] += met(output, target) / len(self.valid_data_loader)
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True)) # It crashes I don't know why

        self.writer.set_step((epoch - 1) * len(self.data_loader) + step, 'valid') # this way, train_loss and val_loss are sync
        self.valid_metrics.update('loss', av_loss)  
        for met in self.metric_ftns:
            self.valid_metrics.update(met.__name__, av_mets[met.__name__])
            
        # add histogram of model parameters to the tensorboard
        #for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)