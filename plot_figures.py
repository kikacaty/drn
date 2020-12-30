import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image
import PIL

from pdb import set_trace as st

def parse_log(log_fn):
    with open(log_fn, 'r') as log_f:
        log = log_f.read()
        matches = re.findall(r'===> mAP (\d+(?:\.\d+)?), avg mAP (\d+(?:\.\d+)?)', str(log))
    mAP_list = np.array([float(item[0]) for item in matches])

    return mAP_list
    

def main():
    arch_list = ['no', 'rf', 'bigger']


    baseline_thres = 10.
    diff_thres = 5

    baseline = {}

    for arch in arch_list:
        filename = 'ngc_log/baseline/duc_hdc_{}_step_{}_evalnum_{}.log'.format(arch,0,200)
        baseline[arch] = parse_log(filename)

    filtered_idx = baseline['rf'][:100] > baseline_thres

    drn_map_list = []
    resnet_map_list = []

    step_list = [1,20,50,100]
    
    eval_map_results = {}

    for arch in arch_list:
        eval_map_results[arch] = {}
        eval_map_results[arch]['mean'] = [baseline[arch][:100][filtered_idx].mean()]
        for attack_steps in step_list:

            filename = 'ngc_log/log/duc_hdc_{}_step_{}_evalnum_{}.log'.format(arch,attack_steps,200)

            # drn_result = parse_log('log/'+fn_base[0]+'_{0}.log'.format(attack_steps))
            # resnet_result = parse_log('log/'+fn_base[1]+'_{0}.log'.format(attack_steps))

            eval_result = parse_log(filename)
            eval_result = eval_result[:100][filtered_idx]

            eval_map_results[arch][attack_steps] = eval_result
            eval_map_results[arch]['mean'].append(eval_result.mean())

            print(arch,', ', attack_steps,': ',eval_result.mean(),eval_result.std())


    fig = plt.figure()
    for arch in arch_list:
        plt.plot(['0','1','20','50','100'],eval_map_results[arch]['mean'],label=arch)
    plt.xlabel('PGD steps')
    plt.ylabel('MAP (%)')
    plt.legend()
    fig.savefig('figs/attack_step_mAP.png')

def gen_demo_figures():

    split_pos = 1100

    drn = np.asarray(Image.open('figs/drn.png'))
    drn_adv = np.asarray(Image.open('figs/drn_adv.png'))
    drn_adv_img = np.asarray(Image.open('figs/drn_adv_img.png'))
    gt_img = np.asarray(Image.open('figs/gt_img.png'))
    gt = np.asarray(Image.open('figs/gt.png').convert('RGB'))

    height, width = drn.shape[:2]
    print(drn.shape)
    adv_side_by_side = drn.copy()
    adv_side_by_side[:,split_pos:,:] = drn_adv[:,split_pos:,:]
    adv_img_side_by_side = gt_img.copy()
    adv_img_side_by_side[:,split_pos:,:] = drn_adv_img[:,split_pos:,:]

    Image.fromarray(adv_img_side_by_side).save('figs/adv_img_side_by_side.png')
    Image.fromarray(adv_side_by_side).save('figs/adv_side_by_side.png')


    pass

if __name__ == "__main__":
    main()
    # gen_demo_figures()