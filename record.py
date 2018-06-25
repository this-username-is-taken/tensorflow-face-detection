import cv2
import time
import subprocess

def video(seconds, frameRate):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: cap isn't openned"

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, 0.04)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    name = "video_" + time.strftime("%d-%m-%H%M%S")+".avi"
    out = cv2.VideoWriter(name, fourcc, frameRate, (640, 480))
    program_starts = time.time()
    result = subprocess.Popen(["ffprobe", name], stdout = subprocess.PIPE, stderr = subprocess.STDOUT, shell=True)
    nFrames = 0
    while (nFrames < seconds * frameRate):
        ret, frame = cap.read()
        if ret==True:
            out.write(frame)
            nFrames += 1
        else:
            break
    cap.release()
    return name

video(5, 20)
