import warnings
from datetime import datetime

import tensorflow as tf
from ecg_classification import ECGClassification
from ecg_dataset import ECGData

warnings.filterwarnings("ignore", category=DeprecationWarning)
tf.random.set_seed(1234)

if __name__ == "__main__":
    # load data once
    ecg_data = ECGData()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    for run_number in range(5):
        for test_source in ECGData.DOMAINS:
            for method in ["cs", "mmd", "projected"]:
                # End-to-end experiment
                ECGClassification(data=ecg_data, method=method, kernel=None, batch_norm=False, bias=False,
                                  timestamp=timestamp, target_domain=[test_source], save_file=True, save_plot=False,
                                  save_feature=True, batch_size=64, fine_tune=False).run_experiment()
            # Clean experiment (no DA)
            ECGClassification(data=ecg_data, method=None, kernel=None, batch_norm=False, bias=False,
                              timestamp=timestamp, target_domain=[test_source], save_file=True, save_plot=False,
                              save_feature=True, batch_size=64, fine_tune=False).run_experiment()
            # Fine-tuning experiment
            ECGClassification(data=ecg_data, method=None, kernel=None, batch_norm=False, bias=False,
                              timestamp=timestamp, target_domain=[test_source], save_file=True, save_plot=False,
                              save_feature=True, batch_size=64, fine_tune=True).run_experiment()
