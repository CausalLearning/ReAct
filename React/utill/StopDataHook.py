from mmcv.runner import Hook


class StopDataHook(Hook):
    '''
    control the dataloader not provide contrastive data for saving time.
    '''

    def __init__(self, dataloader):
        self.dataloader = dataloader

    def after_train_epoch(self, runner):
        """Called after every training epoch to evaluate the results."""
        current = runner.epoch
        if (current + 1) == self.dataloader.dataset.contrastive_epoch:
            self.dataloader.dataset.provide_contrastive_data = False
