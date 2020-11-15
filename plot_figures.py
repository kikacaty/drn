import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image
import PIL

def parse_log(log_fn):
    with open(log_fn, 'r') as log_f:
        log = log_f.read()
        matches = re.findall(r'===> mAP (\d+(?:\.\d+)?), avg mAP (\d+(?:\.\d+)?)', str(log))
    mAP_list = np.array([float(item[0]) for item in matches])

    return mAP_list
    

def main():
    fn_base = ['drn_attack', 'resnet_attack']

    baseline_thres = 10.
    diff_thres = 5

    drn_baseline = parse_log('log/'+fn_base[0]+'_0.log')
    resnet_baseline = parse_log('log/'+fn_base[1]+'_0.log')

    diff = np.abs(drn_baseline - resnet_baseline)

    filtered_idx = (drn_baseline > baseline_thres) & (diff < 5)

    drn_map_list = []
    resnet_map_list = []

    step_list = [0,1,20,50,100]
    for attack_steps in step_list:

        drn_result = parse_log('log/'+fn_base[0]+'_{0}.log'.format(attack_steps))
        resnet_result = parse_log('log/'+fn_base[1]+'_{0}.log'.format(attack_steps))

        drn_filtered_result =  drn_result[filtered_idx]
        resnet_filtered_result = resnet_result[filtered_idx]

        drn_map_list.append(drn_filtered_result.mean())
        resnet_map_list.append(resnet_filtered_result.mean())
        print(drn_map_list)

    fig = plt.figure()
    plt.plot(['0','1','20','50','100'],drn_map_list,label='drn')
    plt.plot(['0','1','20','50','100'],resnet_map_list,label='resnet')
    plt.legend()
    fig.savefig('figs/attack_step_mAP.png')

def gen_demo_figures():

    drn = np.asarray(Image.open('figs/drn.png'))[:,1100-521:1100+512,:]
    drn_adv = np.asarray(Image.open('figs/drn_adv.png'))[:,1100-521:1100+512,:]
    drn_adv_img = np.asarray(Image.open('figs/drn_adv_img.png'))[:,1100-521:1100+512,:]
    gt_img = np.asarray(Image.open('figs/gt_img.png'))[:,1100-521:1100+512,:]
    gt = np.asarray(Image.open('figs/gt.png'))[:,1100-521:1100+512,:]

    height, width = drn.shape[:2]
    print(drn.shape)
    adv_side_by_side = drn.copy()
    adv_side_by_side[:,width//2:,:] = drn_adv[:,width//2:,:]
    adv_img_side_by_side = gt_img.copy()
    adv_img_side_by_side[:,width//2:,:] = drn_adv_img[:,width//2:,:]

    Image.fromarray(adv_img_side_by_side).save('figs/adv_img_side_by_side.png')
    Image.fromarray(adv_side_by_side).save('figs/adv_side_by_side.png')


    pass

if __name__ == "__main__":
    # main()
    gen_demo_figures()