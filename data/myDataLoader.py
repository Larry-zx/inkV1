import torch.utils.data
from data.MyDataset import myDataset


class myDataLoader():
    def __init__(self, opt):
        super(myDataLoader, self).__init__()
        self.opt = opt
        self.dataset = myDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            # num_workers=int(opt.nThreads))
        )
        pass

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
