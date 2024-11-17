from fastai.imports import *
from fastai.vision import *
from fastbook import *
import torch, timm

import os
import glob
import cv2

import cv2


model = load_learner('/home/link-lap-24/Downloads/baseball_banner/4IMR4MA2J7RpYwJ.pkl', cpu=True)
video_path = "/home/link-lap-24/Downloads/baseball_banner/dd.mp4"

gr_ou = "/home/link-lap-24/Downloads/baseball_banner/gr_ou/"
pl_ou = "/home/link-lap-24/Downloads/baseball_banner/pl_ou/"
ot_ou = "/home/link-lap-24/Downloads/baseball_banner/ot_ou/"
pit_ou = "/home/link-lap-24/Downloads/baseball_banner/pit_ou/"
ban_out = "/home/link-lap-24/Downloads/baseball_banner/ban_out/"
cel_out= "/home/link-lap-24/Downloads/baseball_banner/cel_out/"

#tags = ["banner","ground","other","pitch","player"]
target_fps=2

cap = cv2.VideoCapture(video_path)
original_fps=cap.get(cv2.CAP_PROP_FPS)
# print(original_fps)
frames_interval=round(original_fps / target_fps)
# print(frames_interval)
frame_count=0

if not cap.isOpened():
    print("error: while open the video")
    exit()    
    
while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_count * frames_interval)
    
    ret, frame = cap.read()
    if ret:
        cls,indx,pred = model.predict(frame)
        cls_name=cls
        idex_value=indx.tolist()
        pred_score=round(pred.tolist()[idex_value], 5)*100
        
        if cls_name=="ground" and pred_score > 25:
            save_path=os.path.join(gr_ou,f"fr_{frame_count:04d}.jpg")
            cv2.imwrite(save_path,frame)
            print("ground frame found & moved")
            
        if cls_name=="pitch" and pred_score >25:
            save_path=os.path.join(pit_ou,f"fr_{frame_count:04d}.jpg")
            cv2.imwrite(save_path,frame)
            print("pitch frame found & moved")
            
        if cls_name=="other" and pred_score >25:
            save_path=os.path.join(ot_ou,f"fr_{frame_count:04d}.jpg")
            cv2.imwrite(save_path,frame)
            print("other frame found & moved")
            
        if cls_name=="player" and pred_score >25:
            save_path=os.path.join(pl_ou,f"fr_{frame_count:04d}.jpg")
            cv2.imwrite(save_path,frame)
            print("other frame found & moved")
            
        if cls_name=="celebration" and pred_score >25:
            save_path=os.path.join(cel_out,f"fr_{frame_count:04d}.jpg")
            cv2.imwrite(save_path,frame)
            print("other frame found & moved")
            
        if cls_name=="banner" and pred_score >25:
            save_path=os.path.join(ban_out,f"fr_{frame_count:04d}.jpg")
            cv2.imwrite(save_path,frame)
            print("other frame found & moved")
                        
            
        frame_count +=1
                         
    else:
        print("video complete")
        break


