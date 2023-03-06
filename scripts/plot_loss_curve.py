import json
import sys
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR='/home/brandon/5tb/final_mlm_models/'
CS_ADAPTER_DIR=BASE_DIR + 's2orc_cs-10000000-processed-output/checkpoint-61000/trainer_state.json'
CS_DAPT_DIR=BASE_DIR + 's2orc_cs-10000000-processed-dapt-output/checkpoint-10000/trainer_state.json'
BIOMED_ADAPTER_DIR=BASE_DIR + 's2orc_biomed-10000000-processed-output/checkpoint-70000/trainer_state.json'

def plot_loss_curve(title, trainer_state_file_name, batch_size, style):
    trainer_state_file = open(trainer_state_file_name, 'rt')
    trainer_state = json.load(trainer_state_file)
    
    x_ = []
    y_ = []
    for row in trainer_state['log_history']:
        num_samples = row['step']*batch_size
        if num_samples < 100000:
            continue
        x_.append(num_samples)
        y_.append(row['loss'])
    
    x = np.array(x_)
    y = np.array(y_)
    
    # https://stackoverflow.com/a/54628145
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    
    SMOOTHING_FACTOR = 50
    y = moving_average(y, SMOOTHING_FACTOR)
    y[y > 3] = 3
    x = moving_average(x, SMOOTHING_FACTOR)
    x /= 100000
    
    #plt.yscale('log')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Samples (100k) - last 50 steps running avg', fontsize=14)
    plt.ylabel('Validation loss', fontsize=14)
    plt.plot(x, y, style, label=title, linewidth=2)

plt.figure(figsize=(6, 2.5))
ax = plt.gca()
plt.title('Validation loss by domain and training method', fontsize=16)
ax.get_xaxis().get_major_formatter().set_useOffset(False)
ax.get_xaxis().get_major_formatter().set_scientific(False)
plot_loss_curve('CS Adapter', CS_ADAPTER_DIR, 163, 'b-')
plot_loss_curve('CS DAPT', CS_DAPT_DIR, 25*40, 'b--')
plot_loss_curve('BIOMED Adapter', BIOMED_ADAPTER_DIR, 142, 'g-')
plt.legend()
plt.savefig(BASE_DIR + 'plots/domain_loss.png',
    bbox_inches='tight', pad_inches=0.1)
plt.close()
