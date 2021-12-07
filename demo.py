from options.demo_options import DemoOptions
import data as Dataset
from model import create_model
from util import visualizer
from itertools import islice
import numpy as np
import torch

if __name__=='__main__':
    # get demo options
    opt = DemoOptions().parse()
    # creat a dataset
    dataset = Dataset.create_dataloader(opt)

    dataset_size = len(dataset) * opt.batchSize
    print('Running demo pipeline...')
    # create a model
    model = create_model(opt)

    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.test()
