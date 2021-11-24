import cv2
import os
import numpy as np

scale = 4
train_raw_fata_dir = "data/raw_train"
train_save_dir = "data/train/patches_x4/"
list_vids = os.listdir(train_raw_fata_dir)
ret2 = True
patch_w = 32
patch_h = 88
patch_w_hr = 32*scale
patch_h_hr = 88*scale
patch_counter = 0

for vid in list_vids:
    cap = cv2.VideoCapture(train_raw_fata_dir+"/"+vid)
    frame_counter = 0
    lr_temp_frame_l = []
    lr_temp_frame_r = []
    hr_temp_frame_l = []
    hr_temp_frame_r = []
    try:
        while ret2:
            ret, frame = cap.read()
            # ret2, frame2 = cap.read()
            if frame_counter > 600:
                break
            hr_temp_frame_l.append(frame[:,0:1920,:])
            hr_temp_frame_r.append(frame[:,1920:3840,:])    

            frame2 = cv2.resize(frame, (0,0), fx = 1/scale, fy = 1/scale)
            lr_temp_frame_l.append(frame2[:,0:int(1920/scale),:])
            lr_temp_frame_r.append(frame2[:,int(1920/scale):int(3840/scale),:])


            if frame_counter > 3:
                
                for i in range(int(lr_temp_frame_l[0].shape[0] / patch_w)):
                    for j in range(int(lr_temp_frame_l[0].shape[1] / patch_h)):
                        # for k in range(temp_frame_l[t].shape[2]):
                        os.mkdir(train_save_dir+str(patch_counter))
                        for t in range(5):
                            patch_l = lr_temp_frame_l[t][i*patch_w:(i+1)*patch_w, j*patch_h:(j+1)*patch_h,:]
                            patch_r = lr_temp_frame_r[t][i*patch_w:(i+1)*patch_w, j*patch_h:(j+1)*patch_h,:]
                            
                            os.mkdir(train_save_dir +str(patch_counter) + "/" + str(t+1))
                            cv2.imwrite(train_save_dir +str(patch_counter) + "/" + str(t+1)+ "/" +"lr0.png", patch_l)
                            cv2.imwrite(train_save_dir +str(patch_counter) + "/" + str(t+1)+ "/" +"lr1.png", patch_r)

                            if(t==2):
                                hr_patch_l = hr_temp_frame_l[t][i*patch_w_hr:(i+1)*patch_w_hr, j*patch_h_hr:(j+1)*patch_h_hr,:]
                                hr_patch_r = hr_temp_frame_r[t][i*patch_w_hr:(i+1)*patch_w_hr, j*patch_h_hr:(j+1)*patch_h_hr,:]
                                cv2.imwrite(train_save_dir +str(patch_counter) + "/" + str(t+1)+ "/" +"hr0.png", hr_patch_l)
                                cv2.imwrite(train_save_dir+str(patch_counter) + "/" + str(t+1)+ "/" +"hr1.png", hr_patch_r)
                        patch_counter += 1


                lr_temp_frame_l.pop(0)
                lr_temp_frame_r.pop(0)
                hr_temp_frame_l.pop(0)
                hr_temp_frame_r.pop(0)


            frame_counter += 1
    except:
        pass