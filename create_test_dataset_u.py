import cv2
import os
import numpy as np

scale = 6
test_raw_fata_dir = "/media/hassan/My Passport/data/raw_test"
test_save_dir = "/media/hassan/My Passport/data/test/Flickr1024/lr_x6/"
test_hr_save_dir = "/media/hassan/My Passport/data/test/Flickr1024/hr/"
list_vids = os.listdir(test_raw_fata_dir)
ret2 = True

frame_counter = 0
for vid in list_vids:
    cap = cv2.VideoCapture(test_raw_fata_dir+"/"+vid)
    # cap = cv2.VideoCapture(test_raw_fata_dir+"/"+"src02_hrc00_s1920x1080p25n400v0.avi")
    # cap2 = cv2.VideoCapture(test_raw_fata_dir+"/"+"src02_hrc00_s1920x1080p25n400v1.avi")
    
    lr_temp_frame_l = []
    lr_temp_frame_r = []
    hr_temp_frame_l = []
    hr_temp_frame_r = []
    frame_counter2 = 0
    try:
        while ret2:
            ret, frame = cap.read()
            # ret2, frame2 = cap.read()
            if (frame_counter+1)%600==0:
                frame_counter += 1
                break

            hr_temp_frame_l.append(frame[:,0:1920,:])
            hr_temp_frame_r.append(frame[:,1920:3840,:])


            frame2 = cv2.resize(frame, (0,0), fx = 1/scale, fy = 1/scale)
            lr_temp_frame_l.append(frame2[:,0:int(1920/scale),:])
            lr_temp_frame_r.append(frame2[:,int(1920/scale):int(3840/scale),:])


            if frame_counter2 > 3:


                os.mkdir(test_save_dir+str(frame_counter))

                for t in range(5):
                    os.mkdir(test_save_dir +str(frame_counter) + "/" + str(t+1))
                    cv2.imwrite(test_save_dir +str(frame_counter) + "/" + str(t+1)+ "/" +"lr0.png", lr_temp_frame_l[t])
                    cv2.imwrite(test_save_dir +str(frame_counter) + "/" + str(t+1)+ "/" +"lr1.png", lr_temp_frame_r[t])

                    if(t==2):
                        os.mkdir(test_hr_save_dir+str(frame_counter))
                        # os.mkdir(test_hr_save_dir+str(frame_counter)+"/" +str(t+1))
                        cv2.imwrite(test_hr_save_dir +str(frame_counter) + "/"  +"hr0.png", hr_temp_frame_l[t])
                        cv2.imwrite(test_hr_save_dir+str(frame_counter) + "/" + "hr1.png", hr_temp_frame_r[t])


                lr_temp_frame_l.pop(0)
                lr_temp_frame_r.pop(0)
                hr_temp_frame_l.pop(0)
                hr_temp_frame_r.pop(0)


            frame_counter += 1
            frame_counter2 += 1
    except:
        pass