import numpy as np
import matplotlib.pyplot as plt
import re

def parse_log(log_fn):
    with open(log_fn, 'r') as log_f:
        log = log_f.read()
        matches = re.findall(r'===> mAP (\d+(?:\.\d+)?), avg mAP (\d+(?:\.\d+)?)', str(log))
    mAP_list = np.array([float(item[0]) for item in matches])

    return mAP_list
    

def main():
    fn_base = ['hdc_no_attack', 'hdc_rf_attack']

    baseline_thres = 10.
    diff_thres = 5


    drn_baseline = parse_log('log/'+fn_base[0]+'_0.log')
    resnet_baseline = parse_log('log/'+fn_base[1]+'_0.log')

    diff = np.abs(drn_baseline - resnet_baseline)

    filtered_idx = (drn_baseline > baseline_thres) & (diff < 5)

    drn_map_list = []
    resnet_map_list = []

    # step_list = [0,1,20,50,100]
    step_list = [0,20]
    for attack_steps in step_list:

        drn_result = parse_log('log/'+fn_base[0]+'_{0}.log'.format(attack_steps))
        resnet_result = parse_log('log/'+fn_base[1]+'_{0}.log'.format(attack_steps))

        drn_filtered_result =  drn_result[filtered_idx]
        resnet_filtered_result = resnet_result[filtered_idx]

        drn_map_list.append(drn_filtered_result.mean())
        resnet_map_list.append(resnet_filtered_result.mean())
        print(drn_map_list)

    fig = plt.figure()
    plt.plot(['0','20'],drn_map_list,label='hdc_no')
    plt.plot(['0','20'],resnet_map_list,label='hdc_rf')
    plt.legend()
    fig.savefig('figs/attack_step_mAP.png')

if __name__ == "__main__":
    main()