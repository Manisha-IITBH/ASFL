# from ImageSegmentation_Task.kits19.kits_trainer import PfslKits
# from ImageSegmentation_Task.kits19.kits_nonkv_trainer import PfslKits as PfslKitsNonKV

# from ImageSegmentation_Task.IXI.ixi_trainer import IXITrainer
# from ImageSegmentation_Task.IXI.ixi_nonkv_trainer import IXITrainer as IXITrainerNonKV

# from ImageSegmentation_Task.ISIC2019.isic_trainer import ISICTrainer
from datetime import datetime
#-----------------------------------cifar imports
# from ImageClassification_Task.ic_trainer import ICTrainer #ESL
# from ImageClassification_Task.ic_trainer_individual_personalisation import ICTrainer_ESL_IP #ESL_IP
# from ImageClassification_Task.ic_trainer_combined_loss import ICTrainer_CombinedLoss #RSL
# from ImageClassification_Task.ic_trainer_combined_loss_individual_personalisation import ICTrainer_CombinedLoss_IP #RSL_IP
# from ImageClassification_Task.ic_trainer_serverside_contrastive import ICTrainer_CombinedLoss_RSLE #RSL_E
# #-----------------------------------fmnist imports
# from ImageClassification_Task.ic_fmnist_trainer import IC_FMNISTTrainer #ESL
# from ImageClassification_Task.ic_fmnist_trainer_individual_personalisation import IC_FMNISTTrainer_ESL_IP #ESL_IP
# from ImageClassification_Task.ic_fmnist_trainer_combined_loss import IC_FMNISTTrainer_CombinedLoss #RSL
# from ImageClassification_Task.ic_fmnist_trainer_combined_loss_individual_personalisation import IC_FMNISTTrainer_CombinedLoss_IP
# from ImageClassification_Task.ic_fmnist_trainer_serverside_contrastive import IC_FMNISTTrainer_CombinedLoss_RSLE #RSL_E
# from ImageClassification_Task.ic_fmnist_trainer_serverside_contrastive_individual_personalisation import IC_FMNISTTrainer_CombinedLoss_RSLE_IP #RSL_E_IP

# #-----------------------------------isic2019 imports
# from ImageClassification_Task.ISIC2019.ic_isic_trainer_individual_personalisation import IC_ISICTrainer_ESL_IP #ESL_IP
# from ImageClassification_Task.ISIC2019.ic_isic_trainer_combined_loss_individual_personalisation import IC_ISICTrainer_CombinedLoss_IP #RSL_IP
# from ImageClassification_Task.ISIC2019.ic_isic_trainer_serverside_contrastive_individual_personalisation import IC_ISICTrainer_CombinedLoss_RSLE_IP #RSL_E_IP

#-----------------------------------officehome imports
# from ImageClassification_Task.officehome.ic_officehome_trainer_individual_personalisation import IC_OFFICEHOMETrainer_ESL_IP #ESL_IP
# from ImageClassification_Task.officehome.ic_officehome_trainer_combined_loss_individual_personalisation import IC_OFFICEHOMETrainer_CombinedLoss_IP #RSL_IP
# from ImageClassification_Task.officehome.ic_officehome_trainer_serverside_contrastive_individual_personalisation import IC_OFFICEHOMETrainer_CombinedLoss_RSLE_IP #RSL_E_IP


# -----------------------------------vlcs imports
from ImageClassification_Task.vlcs.ic_vlcs_trainer_individual_personalisation import IC_VLCSTrainer_ESL_IP #ESL_IP
from ImageClassification_Task.vlcs.ic_vlcs_trainer_combined_loss_individual_personalisation import IC_VLCSTrainer_CombinedLoss_IP #RSL_IP
from ImageClassification_Task.vlcs.ic_vlcs_trainer_serverside_contrastive_individual_personalisation import IC_VLCSTrainer_CombinedLoss_RSLE_IP #RSL_E_IP

#-----------------------------------vlcs imports
from ImageClassification_Task.pacs.ic_pacs_trainer_individual_personalisation import IC_PACSTrainer_ESL_IP #ESL_IP
from ImageClassification_Task.pacs.ic_pacs_trainer_combined_loss_individual_personalisation import IC_PACSTrainer_CombinedLoss_IP #RSL_IP
from ImageClassification_Task.pacs.ic_pacs_trainer_serverside_contrastive_individual_personalisation import IC_PACSTrainer_CombinedLoss_RSLE_IP #RSL_E_IP

from utils.argparser import parse_arguments
from pprint import pprint

if __name__ == '__main__':
    args = parse_arguments()

    pprint(vars(args))

    now = datetime.now()
    print("Current Date and Time:", now.strftime("%Y-%m-%d %H:%M:%S"))

    if args.dataset == 'ISIC2019':
        if args.use_key_value_store:
            trainer = ISICTrainer(args)
        else:
            raise NotImplementedError
        
        trainer.fit()
        trainer.inference()
    
    elif args.dataset == 'ISIC2019_ESL_IP':
        if args.use_key_value_store:
            trainer = IC_ISICTrainer_ESL_IP(args)
        else:
            raise NotImplementedError
        
        trainer.fit()
    
    elif args.dataset == 'ISIC2019_RSL_IP':
        if args.use_key_value_store:
            trainer = IC_ISICTrainer_CombinedLoss_IP(args)
        else:
            raise NotImplementedError
        
        trainer.fit()
        
    elif args.dataset == 'ISIC2019_RSLE_IP':
        if args.use_key_value_store:
            trainer = IC_ISICTrainer_CombinedLoss_RSLE_IP(args)
        else:
            raise NotImplementedError
        
        trainer.fit()

    elif args.dataset == 'OFFICEHOME_ESL_IP':
        if args.use_key_value_store:
            trainer = IC_OFFICEHOMETrainer_ESL_IP(args)
        else:
            raise NotImplementedError
        
        trainer.fit()

    elif args.dataset == 'VLCS_ESL_IP':
        if args.use_key_value_store:
            trainer = IC_VLCSTrainer_ESL_IP(args)
        else:
            raise NotImplementedError
        
        trainer.fit()

    elif args.dataset == 'PACS_ESL_IP':
        if args.use_key_value_store:
            trainer = IC_PACSTrainer_ESL_IP(args)
        else:
            raise NotImplementedError
        
        trainer.fit()
    
    elif args.dataset == 'OFFICEHOME_RSL_IP':
        if args.use_key_value_store:
            trainer = IC_OFFICEHOMETrainer_CombinedLoss_IP(args)
        else:
            raise NotImplementedError
        
        trainer.fit()

    elif args.dataset == 'VLCS_RSL_IP':
        if args.use_key_value_store:
            trainer = IC_VLCSTrainer_CombinedLoss_IP(args)
        else:
            raise NotImplementedError
        
        trainer.fit()

    elif args.dataset == 'PACS_RSL_IP':
        if args.use_key_value_store:
            trainer = IC_PACSTrainer_CombinedLoss_IP(args)
        else:
            raise NotImplementedError
        
        trainer.fit()
    
    elif args.dataset == 'VLCS_RSLE_IP':
        if args.use_key_value_store:
            trainer = IC_VLCSTrainer_CombinedLoss_RSLE_IP(args)
        else:
            raise NotImplementedError
        
        trainer.fit()

    elif args.dataset == 'PACS_RSLE_IP':
        if args.use_key_value_store:
            trainer = IC_PACSTrainer_CombinedLoss_RSLE_IP(args)
        else:
            raise NotImplementedError
        
        trainer.fit()
        
    elif args.dataset == 'CIFAR10_ESL':
        if args.use_key_value_store:
            trainer = ICTrainer(args)
        else:
            raise NotImplementedError
        
        trainer.fit()
    
    elif args.dataset == 'FMNIST_ESL':
        if args.use_key_value_store:
            trainer = IC_FMNISTTrainer(args)
        else:
            raise NotImplementedError
        print("Traing is getting Started, be ready to amuse...")
        trainer.fit()

    elif args.dataset == 'CIFAR10_CL':
        if args.use_key_value_store:
            trainer = ICTrainer_CombinedLoss(args)
        else:
            raise NotImplementedError
        
        trainer.fit()
    
    elif args.dataset == 'FMNIST_CL':
        if args.use_key_value_store:
            trainer = IC_FMNISTTrainer_CombinedLoss(args)
        else:
            raise NotImplementedError
        
        trainer.fit()

    elif args.dataset == 'CIFAR10_ESL_IP':
        if args.use_key_value_store:
            trainer = ICTrainer_ESL_IP(args)
        else:
            raise NotImplementedError
        
        trainer.fit()
    
    elif args.dataset == 'FMNIST_ESL_IP':
        if args.use_key_value_store:
            trainer = IC_FMNISTTrainer_ESL_IP(args)
        else:
            raise NotImplementedError
        
        trainer.fit()
    
    elif args.dataset == 'CIFAR10_CL_IP':
        if args.use_key_value_store:
            trainer = ICTrainer_CombinedLoss_IP(args)
        else:
            raise NotImplementedError
        
        trainer.fit()
    
    elif args.dataset == 'FMNIST_CL_IP':
        if args.use_key_value_store:
            trainer = IC_FMNISTTrainer_CombinedLoss_IP(args)
        else:
            raise NotImplementedError
        
        trainer.fit()
    
    elif args.dataset == 'CIFAR10_RSLE':
        if args.use_key_value_store:
            trainer = ICTrainer_CombinedLoss_RSLE(args)
        else:
            raise NotImplementedError
        
        trainer.fit()
    
    elif args.dataset == 'FMNIST_RSLE':
        if args.use_key_value_store:
            trainer = IC_FMNISTTrainer_CombinedLoss_RSLE(args)
        else:
            raise NotImplementedError
        
        trainer.fit()
    
    elif args.dataset == 'FMNIST_RSLE_IP':
        if args.use_key_value_store:
            trainer = IC_FMNISTTrainer_CombinedLoss_RSLE_IP(args)
        else:
            raise NotImplementedError
        
        trainer.fit()
    
    
    else:
        print('invalid dataset -_-')
    
    now = datetime.now()
    print("Current Date and Time:", now.strftime("%Y-%m-%d %H:%M:%S"))
