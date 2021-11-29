from os.path import expanduser

# DATA_DIR = expanduser("~/bg-forecasting/data/")

DIABETES_TYPES = ['DM1', 'DM2', 'GDM', 'MODY', 'other']
GENDERS = ['m', 'f']
THERAPIES = ['CSII', 'MDI', 'basal insulin', 'insulin & nia', 'no insulin antidiabetics [nia]', 'other']
SENSORS = ['DIA', 'dex', 'med']

if expanduser('~') == '/local/home/jholzem':
    GP_MODEL_PATH = expanduser("~/pretrained_models/gps/") + "gp_w12_b8_lr0.01.pth"
elif expanduser('~') == '/cluster/home/jholzem':
    GP_MODEL_PATH = '/cluster/scratch/jholzem/pretrained_models/gp/' + "gp_w12_b8_lr0.01.pth"
else:
    raise FileNotFoundError

DATA_DIR = "/wave/bg-prediction/data"