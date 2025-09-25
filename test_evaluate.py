import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
from basicsr.data.flare7k_dataset import Flare_Image_Loader,RandomGammaCorrection
from basicsr.archs.uformer_arch import Uformer
from basicsr.archs.ast_arch import AST
import argparse
import math
from basicsr.archs.restormer_arch import Restormer
from basicsr.archs.mprnet_arch import MPRNet
from basicsr.archs.hinet_arch import HINet
import re
from basicsr.utils.flare_util import blend_light_source,get_args_from_json,save_args_to_json,mkdir,predict_flare_from_6_channel,predict_flare_from_3_channel
from torch.distributions import Normal
import torchvision.transforms as transforms
import os
from thop import profile
import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage import io
from torchvision.transforms import ToTensor
import numpy as np
from glob import glob
import lpips

parser = argparse.ArgumentParser()
parser.add_argument('--input',type=str,default=None)
parser.add_argument('--output',type=str,default=None)
parser.add_argument('--model_type',type=str,default='Uformer')
parser.add_argument('--model_path',type=str,default='checkpoint/flare7kpp/net_g_last.pth')
parser.add_argument('--gt',type=str,default=None)
parser.add_argument('--mask',type=str,default=None)
# parser.add_argument('--flare7kpp', action='store_const', const=False, default=False) #use flare7kpp's inference method and output the light source directly.


def expand2square(timg,factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(1,1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)
    return img, mask
def compare_lpips(img1, img2, loss_fn_alex):
    to_tensor=ToTensor()
    img1_tensor = to_tensor(img1).unsqueeze(0)
    img2_tensor = to_tensor(img2).unsqueeze(0)
    output_lpips = loss_fn_alex(img1_tensor.cuda(), img2_tensor.cuda())
    return output_lpips.cpu().detach().numpy()[0,0,0,0]


def compare_score(img1,img2,img_seg):
    # Return the G-PSNR, S-PSNR, ghost-PSNR and Score
    # This module is for the MIPI 2023 Challange: https://codalab.lisn.upsaclay.fr/competitions/9402
    mask_type_list=['clean']
    metric_dict={'clean':0}
    for mask_type in mask_type_list:
        mask_area,img_mask=extract_mask(img_seg)[mask_type]
        if mask_area>0:
            img_gt_masked=img1*img_mask
            img_input_masked=img2*img_mask
            input_mse=compare_mse(img_gt_masked, img_input_masked)/(255*255*mask_area)
            input_psnr=10 * np.log10((1.0 ** 2) / input_mse)
            metric_dict[mask_type]=input_psnr
        else:
            metric_dict.pop(mask_type)
    return metric_dict

def compare_score_new(img1,img2,img_seg):
    # Return the G-PSNR, S-PSNR, ghost-PSNR and Score
    # This module is for the MIPI 2023 Challange: https://codalab.lisn.upsaclay.fr/competitions/9402
    mask_type_list=['glare','streak','ghost']
    metric_dict={'glare':0,'streak':0,'ghost':0}
    for mask_type in mask_type_list:
        mask_area,img_mask=extract_mask_new(img_seg)[mask_type]
        if mask_area>0:
            img_gt_masked=img1*img_mask
            img_input_masked=img2*img_mask
            input_mse=compare_mse(img_gt_masked, img_input_masked)/(255*255*mask_area)
            input_psnr=10 * np.log10((1.0 ** 2) / input_mse)
            metric_dict[mask_type]=input_psnr
        else:
            metric_dict.pop(mask_type)
    return metric_dict

def extract_mask_new(img_seg):
    # Return a dict with 3 masks including streak,glare,ghost(whole image w/o light source), masks are returned in 3ch. 
    # glare: [255,255,0]
    # streak: [255,0,0]
    # light source: [0,0,255]
    # ghost: [0,255,0]
    # others: [0,0,0]
    mask_dict={}
    streak_mask=(img_seg[:,:,0]-img_seg[:,:,1])/255
    glare_mask=(img_seg[:,:,1])/255
    ghost_mask = (img_seg[:,:,2]) / 255

    
    mask_dict['glare']=[np.sum(glare_mask)/(512*512),np.expand_dims(glare_mask,2).repeat(3,axis=2)] #area, mask
    mask_dict['streak']=[np.sum(streak_mask)/(512*512),np.expand_dims(streak_mask,2).repeat(3,axis=2)] 
    mask_dict['ghost']=[np.sum(ghost_mask)/(512*512),np.expand_dims(ghost_mask,2).repeat(3,axis=2)] 
    return mask_dict

def extract_mask(img_seg):
    # Return a dict with 3 masks including streak,glare,ghost(whole image w/o light source), masks are returned in 3ch. 
    # glare: [255,255,0]
    # streak: [255,0,0]
    # light source: [0,0,255]
    # others: [0,0,0]
    mask_dict={}
    clean_mask = img_seg/255
    # streak_mask=(img_seg[:,:,0]-img_seg[:,:,1])/255
    # glare_mask=(img_seg[:,:,1])/255
    # ghost_mask=(255-img_seg[:,:,2])/255
    mask_dict['clean']=[np.sum(clean_mask)/(512*512),np.expand_dims(clean_mask,2).repeat(3,axis=2)] #area, mask
    # mask_dict['streak']=[np.sum(streak_mask)/(512*512),np.expand_dims(streak_mask,2).repeat(3,axis=2)] 
    # mask_dict['ghost']=[np.sum(ghost_mask)/(512*512),np.expand_dims(ghost_mask,2).repeat(3,axis=2)] 
    return mask_dict
def calculate_metrics_new(args):
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    gt_folder = args['gt'] + '/*'
    input_folder = os.path.join(result_path,"blend") + '/*'
    # input_folder = "/home/ubuntu/qls/FlareX/test_dataset/Ours/input/*"
    gt_list = sorted(glob(gt_folder))
    input_list = sorted(glob(input_folder))
    if args['mask'] is not None:
        mask_folder = args['mask'] + '/*'
        mask_list= sorted(glob(mask_folder))

    assert len(gt_list) == len(input_list)
    n = len(gt_list)

    ssim, psnr, lpips_val = 0, 0, 0
    score_dict={'glare':0,'streak':0,'ghost':0,'glare_num':0,'streak_num':0,'ghost_num':0}
    # pp = 0
    for i in tqdm(range(n)):
        img_gt = io.imread(gt_list[i])
        img_input = io.imread(input_list[i])
        ssim += compare_ssim(img_gt, img_input, multichannel=True)
        psnr += compare_psnr(img_gt, img_input, data_range=255)
        lpips_val += compare_lpips(img_gt, img_input, loss_fn_alex)
        if args['mask'] is not None:
            # psnr_s = 0
            # psnr_g = 0
            img_seg=io.imread(mask_list[i])
            metric_dict=compare_score_new(img_gt,img_input,img_seg)
            for key in metric_dict.keys():
                # if key == "streak":
                #     psnr_s = metric_dict[key]
                # elif key == "glare":
                #     psnr_g = metric_dict[key]
                score_dict[key]+=metric_dict[key]
                score_dict[key+'_num']+=1
            # if psnr_g < psnr_s:
            #     pp += 1
            #     print(pp)
    ssim /= n
    psnr /= n
    lpips_val /= n
    print(f"PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips_val}")
    if args['mask'] is not None:
        for key in ['glare','streak','ghost']:
            if score_dict[key+'_num'] == 0:
                assert False, "Error, No mask in this type!"
            score_dict[key]/= score_dict[key+'_num']
        score_dict['score']=1/3*(score_dict['glare']+score_dict['ghost']+score_dict['streak'])
        print(f"Score: {score_dict['score']}, G-PSNR: {score_dict['glare']}, S-PSNR: {score_dict['streak']}, GO-PSNR: {score_dict['ghost']}")
        
def calculate_metrics(args):
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    gt_folder = args['gt'] + '/*'
    input_folder = os.path.join(result_path,"blend") + '/*'
    # input_folder = "/home/ubuntu/qls/FlareX/test_dataset/Ours/input"+'/*'
    gt_list = sorted(glob(gt_folder))
    input_list = sorted(glob(input_folder))
    if args['mask'] is not None:
        mask_folder = args['mask'] + '/*'
        mask_list= sorted(glob(mask_folder))

    assert len(gt_list) == len(input_list)
    n = len(gt_list)

    ssim, psnr, lpips_val = 0, 0, 0
    score_dict={'clean':0,'clean_num':0}
    file_path = os.path.join(result_path,'ans.txt')
    file = open(file_path,'w')
    for i in tqdm(range(n)):
        img_gt = io.imread(gt_list[i])
        img_input = io.imread(input_list[i])
        ssim0 = compare_ssim(img_gt, img_input, multichannel=True)
        ssim += ssim0
        psnr0 = compare_psnr(img_gt, img_input, data_range=255)
        psnr += psnr0
        lpips_val0 = compare_lpips(img_gt, img_input, loss_fn_alex)
        lpips_val += lpips_val0
        psnr_c = 0
        if args['mask'] is not None:
            img_seg=io.imread(mask_list[i])
            metric_dict=compare_score(img_gt,img_input,img_seg)
            for key in metric_dict.keys():
                score_dict[key]+=metric_dict[key]
                score_dict[key+'_num']+=1
            psnr_c = metric_dict['clean']
        text_content = f"{str(i)}, PSNR: {psnr0}, SSIM: {ssim0}, LPIPS: {lpips_val0},PSNR-Clean: {psnr_c} \n"
        file.write(text_content)
    ssim /= n
    psnr /= n
    lpips_val /= n
    
    if args['mask'] is not None:
        for key in ['clean']:
            if score_dict[key+'_num'] == 0:
                assert False, "Error, No mask in this type!"
            score_dict[key]/= score_dict[key+'_num']
        # print(f"PSNR-C: {score_dict['clean']}")
        
    text_content = f"PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips_val},PSNR-Clean: {score_dict['clean']} \n"
    print(text_content) 
    

    # 写入文本到文件
   
    file.write(text_content)
        
def extract_number(file_path):
    # 从路径末尾提取数字部分
    match = re.search(r'(\d+)', file_path[::-1])
    return int(match.group()[::-1]) if match else float('inf')
def mkdir(path):
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)
def load_params(model_path):
     full_model=torch.load(model_path)
     if 'params_ema' in full_model:
          return full_model['params_ema']
     elif 'params' in full_model:
          return full_model['params']
     else:
          return full_model
def demo(images_path,output_path,model_type,pretrain_dir):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    test_path = glob(images_path)
    test_path.sort()
    # test_path = sorted(test_path, key=extract_number)
    print(test_path)
    result_path=output_path
    torch.cuda.empty_cache()
    if model_type=='Uformer':
        model=Uformer(img_size=512,img_ch=3).cuda()
        model.load_state_dict(load_params(pretrain_dir))
    elif model_type=='Restormer':
        model=Restormer().cuda()
        model.load_state_dict(load_params(pretrain_dir))
    elif model_type == "HINet":
        model = HINet().cuda()
        model.load_state_dict(load_params(pretrain_dir))
    elif model_type == "MPRNet":
        model = MPRNet().cuda()
        model.load_state_dict(load_params(pretrain_dir))
    elif model_type == "AST":
        model = AST().cuda()
        model.load_state_dict(load_params(pretrain_dir))
    else:
        assert False, "This model is not supported!!"
    to_tensor=transforms.ToTensor()
    resize=transforms.Resize((512,512))
    for i,image_path in tqdm(enumerate(test_path)):
        mkdir(os.path.join(result_path,"flare"))
        mkdir(os.path.join(result_path,"input"))
        mkdir(os.path.join(result_path,"blend"))
        
        flare_path = os.path.join(result_path,"flare/"+str(i).zfill(5)+".png")
        merge_path = os.path.join(result_path,"input/"+str(i).zfill(5)+".png")
        blend_path = os.path.join(result_path,"blend/"+str(i).zfill(5)+".png")

        merge_img = Image.open(image_path).convert("RGB")
        resize2org = transforms.Resize((merge_img.size[1], merge_img.size[0]))
        merge_img_ori = to_tensor(merge_img)
        merge_img = resize(merge_img_ori)
        _, h, w =  merge_img.shape
        merge_img = merge_img.cuda().unsqueeze(0)

        model.eval()
        with torch.no_grad():
            rgb_noisy, mask = expand2square(merge_img, factor=16)
            output_img=model(rgb_noisy)
            output_img = torch.masked_select(output_img,mask.bool()).reshape(1,6,h,w)
            gamma=torch.Tensor([2.2])
            
            deflare_img,flare_img_predicted,merge_img_predicted=predict_flare_from_6_channel(output_img,gamma)
            deflare_img = merge_img_ori.cuda().unsqueeze(0) - resize2org(merge_img - deflare_img)
            # torchvision.utils.save_image(merge_img, merge_path)
            if flare7kpp:
                pass
            else:
                deflare_img= blend_light_source(merge_img_ori.cuda().unsqueeze(0), deflare_img, 0.97)
            torchvision.utils.save_image(flare_img_predicted, flare_path)
            torchvision.utils.save_image(deflare_img, blend_path)
            
            
            
if __name__ == "__main__":
    args = parser.parse_args()
    flare7kpp = True
    model_type=args.model_type
    images_path=os.path.join(args.input,"*.*")
    result_path=args.output
    pretrain_dir=args.model_path

    # demo(images_path,result_path,model_type,pretrain_dir)
    # eval the same time
    eval_arg = vars(args)
    
    
    # calculate_metrics_new(eval_arg)
    calculate_metrics(eval_arg)
    