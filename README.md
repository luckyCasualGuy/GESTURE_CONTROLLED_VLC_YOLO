# GESTURE_CONTROLLED_VLC_YOLO
Gesture Recognition was Implemented using YOLO V3 and YOLO v3 tiny.

In this project I train YOLO v3 tiny to recognize 2 hand gestures ( you can train more though but collecting and labelling 400 + photos for each class is pain !!)  

This model is than used to controll vlc  

There are two implementations for this script  
`For PC` and `For Raspberry pi with its camera module`  

If you want to run the script on your PC...  
Make sure you have all required modules  
```
# not adding requirements.txt as you just need two things
>>> pip install tensoroflow
>>> pip install python-vlc

## Also make sure you have appropriate vlc installed on your PC/ PI
```

Make sure you have good working camera and proper light conditions and you are good to go :)  

### NOTE:
Model used is just prototype and it works just fine for me...  
If you want better model than mine than train it yourself using my training scripts inside `Training dir`  
( Make sure you have 400 + images of each gesture you want to detect In fact any object more information added in that folder... )  

I use tflite version becauze this script was made to run in **raspberry pi**.  
If you want to use proper yolo model In my script you can replicate `YOLO_V3_Tiny` module that I have created and make suitable changes inside it.  
`I will add this functionality later though !!`  

