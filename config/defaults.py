from yacs.config import CfgNode as CN

###########################
# Config definition
###########################

_C = CN()

_C.SEED = 0
_C.NOTE = ''

###########################
# Dataset
###########################
_C.DATASET = CN()
# Directory where datasets are stored
_C.DATASET.ROOT = ''
_C.DATASET.NAME = ''
# List of domains
_C.DATASET.SOURCE_DOMAINS = []
_C.DATASET.TARGET_DOMAINS = []
_C.DATASET.SOURCE_DOMAIN = ''
_C.DATASET.TARGET_DOMAIN = ''
_C.DATASET.SOURCE_VALID_TYPE = 'val'
_C.DATASET.SOURCE_VALID_RATIO = 1.0
_C.DATASET.SOURCE_TRANSFORMS = ('Resize','RandomCrop','Normalize')
_C.DATASET.TARGET_TRANSFORMS = ('Resize','RandomCrop','Normalize')
_C.DATASET.QUERY_TRANSFORMS = ('Resize','CenterCrop','Normalize')
_C.DATASET.TEST_TRANSFORMS = ('Resize','CenterCrop','Normalize')
_C.DATASET.RAND_TRANSFORMS = 'rand_transform'
_C.DATASET.NUM_CLASS = 12

###########################
# Dataloader
###########################
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.BATCH_SIZE = 32
_C.DATALOADER.TGT_UNSUP_BS_MUL = 1

###########################
# Model
###########################
_C.MODEL = CN()
# Path to model weights (for initialization)
_C.MODEL.INIT_WEIGHTS = ''
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = 'ResNet50Fc'
_C.MODEL.BACKBONE.PRETRAINED = True
_C.MODEL.NORMALIZE = False
_C.MODEL.TEMP = 0.05

###########################
# Optimization
###########################
_C.OPTIM = CN()
_C.OPTIM.NAME = 'Adadelta'
_C.OPTIM.SOURCE_NAME = 'Adadelta'
_C.OPTIM.UDA_NAME = 'Adadelta'
_C.OPTIM.SOURCE_LR = 0.1
_C.OPTIM.UDA_LR = 0.1
_C.OPTIM.ADAPT_LR = 0.1
_C.OPTIM.BASE_LR_MULT = 0.1

###########################
# Trainer specifics
###########################
_C.TRAINER = CN()
_C.TRAINER.LOAD_FROM_CHECKPOINT = True
_C.TRAINER.TRAIN_ON_SOURCE = True
_C.TRAINER.MAX_SOURCE_EPOCHS = 20
_C.TRAINER.MAX_UDA_EPOCHS = 20
_C.TRAINER.MAX_EPOCHS = 20
_C.TRAINER.EVAL_ACC = True
_C.TRAINER.ITER_PER_EPOCH = None

###########################
# Active DA
###########################
_C.ADA = CN()
_C.ADA.TASKS = None
_C.ADA.BUDGET = 0.05
_C.ADA.ROUND = 5
_C.ADA.ROUNDS = None
_C.ADA.UDA = 'dann'
_C.ADA.DA = 'ft'
_C.ADA.AL = 'LADA'
_C.ADA.SRC_SUP_WT = 1.0
_C.ADA.TGT_SUP_WT = 1.0
_C.ADA.UNSUP_WT = 0.1
_C.ADA.CEN_WT = 0.1

###########################
# LADA
###########################
_C.LADA = CN()
_C.LADA.S_K = 10
_C.LADA.S_Kc = 10
_C.LADA.S_PROP_ITER = 1
_C.LADA.S_PROP_COEF = 1.0
_C.LADA.A_K = 10
_C.LADA.A_ALPHA = 1.0
_C.LADA.A_TH = 0.9
_C.LADA.A_RAND_NUM = 0
