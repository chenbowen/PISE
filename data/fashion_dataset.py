import os
from os import listdir
from os.path import join, exists
from collections import defaultdict
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import pandas as pd
from shutil import copyfile
import random

class FashionDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--test_pairs', type=str, default="fasion-pairs-test.csv")
        parser.add_argument('--demo_video_name', type=str, default="jntm")
        parser.add_argument('--demo_source_image', type=str, default="025_3_2_1.png")
        if is_train:
            parser.set_defaults(load_size=256)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(old_size=(256, 256))
        parser.set_defaults(structure_nc=18)
        parser.set_defaults(image_nc=3)
        parser.set_defaults(display_winsize=256)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = opt.phase
        
        if phase in ["train", "test"]:
            if phase == "train":
                pairLst = join(root, 'fasion-pairs-%s.csv' %phase)
                name_pairs = self.random_sampling_train_pairs()
            elif phase == "test":
                pairLst = join(root, opt.test_pairs)
                print("test using", pairLst)
                name_pairs = self.init_categories(pairLst)
                
            image_dir = join(root, '%s' % phase)
            bonesLst = join(root, 'fasion-annotation-%s.csv' %phase)
            par_dir = join(root, '%sSPL8' %phase)
        else: # demo
            image_dir, bonesLst, name_pairs, par_dir = self.init_demo(opt)
            
        
        return image_dir, bonesLst, name_pairs, par_dir

    def init_demo(self, opt):
        root = opt.dataroot
        sname = opt.demo_source_image
        vname = opt.demo_video_name
        vroot = 'demo_videos'
        image_dir = join(vroot, vname)
        subfolder = "test" if exists(join(root, "test", sname)) else "train"
        
        source_bone_list = join(root, f'fasion-annotation-{subfolder}.csv')
        demo_bone_list = join(vroot, vname, f'{vname}.csv')
        # find and append source image bone to the video bone list
        with open(source_bone_list) as f1:
            for line in f1:
                if line.startswith(sname):
                    source_bone = line                    
                    with open(demo_bone_list, "r+") as f2:
                        if f2.readlines()[-1] != source_bone:
                            f2.write(source_bone)
                    break
        par_dir = join(root, f'{subfolder}SPL8')
        # move source image to demo images root
        old_A = join(root, subfolder, sname)
        A = join(image_dir, sname)
        copyfile(old_A, A)
        name_pairs = [[sname, B] for B in listdir(image_dir) if B.startswith("frame")]
        print(name_pairs[:20])
        
        return image_dir, demo_bone_list, name_pairs, par_dir
        

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        size = len(pairs_file_train)
        pairs = []
        print('Loading data pairs ...')
        for i in range(size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            pairs.append(pair)

        print('Loading data pairs finished ...')  
        return pairs    

    def random_sampling_train_pairs(self):
        size = 10000
        appearances = defaultdict(list)
        train_f = open('./fashion_data/train.lst', 'r')
        for line in train_f:
            line = line.strip()
            if line.endswith('.jpg') or line.endswith('.png'):
                k = "_".join(line.split("_")[:2])
                appearances[k].append(line)
        keys = list(appearances.keys())
        pairs = []
        print('Random sampling data pairs ...')  
        for i in range(size):
            k = random.choice(keys)
            pair = random.sample(appearances[k], 2)
            pairs.append(pair)
            if i < 20:
                print(pair)
        print('Done: Random sampling data pairs...') 
        return pairs
    def name(self):
        return "FashionDataset"

                
