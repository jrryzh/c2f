from data.dataloader_Fishbowl import FishBowl
from data.dataloader_MOViD_A import MOViD_A
from data.dataloader_KINS import Kins_Fusion_dataset, KINS_Aisformer_VRSP_Intersection
from data.dataloader_COCOA import COCOA_Fusion_dataset, COCOA_VRSP
from data.dataloader_UOAIS import Fusion_UOAIS
from data.dataloader_OSD import Fusion_OSD
from data.dataloader_UOAIS_allvm import Fusion_UOAIS_ALLVM
from data.dataloader_OSD_allvm import Fusion_OSD_ALLVM

def load_dataset(config, args, mode):
    if mode=="train":
        if args.dataset=="KINS":
            train_dataset = Kins_Fusion_dataset(config, mode='train')
            test_dataset = Kins_Fusion_dataset(config, mode='test')
        elif args.dataset=="COCOA":
            train_dataset = COCOA_Fusion_dataset(config, mode='train')
            test_dataset = COCOA_Fusion_dataset(config, mode='test')
        elif args.dataset=="Fishbowl":
            train_dataset = FishBowl(config, mode='train')
            test_dataset = FishBowl(config, mode='test')
        elif args.dataset=="MOViD_A":
            train_dataset = MOViD_A(config, mode='train')
            test_dataset = MOViD_A(config, mode='test')
        elif args.dataset=="UOAIS":
            train_dataset = Fusion_UOAIS(config, mode='train')
            test_dataset = Fusion_UOAIS(config, mode='test')
        elif args.dataset=="OSD":
            train_dataset = Fusion_OSD(config, mode='train')
            test_dataset = Fusion_OSD(config, mode='test')
        return train_dataset, test_dataset 
    else:
        if args.dataset=="KINS":
            test_dataset = KINS_Aisformer_VRSP_Intersection(config, mode='test')
        elif args.dataset=="COCOA":
            test_dataset = COCOA_Fusion_dataset(config, mode='test')
        elif args.dataset=="Fishbowl":
            test_dataset = FishBowl(config, mode='test')
        elif args.dataset=="MOViD_A":
            test_dataset = MOViD_A(config, mode='test')
        elif args.dataset=="UOAIS":
            test_dataset = Fusion_UOAIS(config, mode='test')
        elif args.dataset=="OSD":
            test_dataset = Fusion_OSD(config, mode='test')
        elif args.dataset=="UOAIS_ALLVM":
            test_dataset = Fusion_UOAIS_ALLVM(config, mode='test')
        elif args.dataset=="OSD_ALLVM":
            test_dataset = Fusion_OSD_ALLVM(config, mode='test')
        return test_dataset