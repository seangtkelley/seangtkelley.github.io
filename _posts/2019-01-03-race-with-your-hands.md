---
layout: post
title: "Race with Your Hands - CV-Powered Game Controller"
desc: "In this post, I take an already implemented model for detecting human hands in an image and use the model's output to create a custom steering wheel game controller."
tag: "Computer Vision"
author: "Sean Kelley"
thumb: "/img/blog/2019-01-03-race-with-your-hands/right-turn-light-brake.png"
date: 2019-01-03
---

As the world of computer vision is ever-growing, I find myself struggling to apply the many things I gather from a field of study. Should I implement a new paper or maybe attempt a challenge? In all honesty, as a student, I rarely have time. However, in this project, I set out to show how you can take things you know from a field and apply them to create something totally different. This is something that I think the field of machine learning has trouble with. Although the promise of making models that can classify objects with a higher degree of accuracy makes people excited, the application of such breakthroughs is many times less clear. In this post, I take an already implemented model for detecting human hands in an image and use the model's output to create a custom steering wheel game controller.

## Controller Design

The controller will work by tracking both of your hands. Just like normal, your right will be the gas and your left the brake. Disclaimer: operating the pedals of your actual car with your hands is not recommended. We can determine how far to "depress" the pedals by finding the height of the detected bounding boxes. For example, if the bounding box is short, your hand is likely parallel to the view of the camera and this would be like pressing down the pedal. If the bounding box is tall, your hand is likely perpendicular to the view of the camera and this would be like releasing the pedal. All we have to do is tweak some constants according to the webcam's resolution.

To determine the steering angle, we can take the slope between the two bounding boxes. If we calculate a positive slope, your right hand is above your left which is the motion of a left turn. If we calculate a negative slope, your left hand is above your right which is the motion of a right turn.

I have included some images below to help explain the functionality. Please excuse my HODL T-shirt.

### Idle: both pedals are released
![png](/img/blog/2019-01-03-race-with-your-hands/idle.png)

### Left Turn with Full Gas
![png](/img/blog/2019-01-03-race-with-your-hands/left-turn-full-gas.png)

### Right Turn with Light Brake
![png](/img/blog/2019-01-03-race-with-your-hands/right-turn-light-brake.png)

## Hand Detection

Surprisingly, this was one of the easier parts of the project. Although we could train our own model using a popular architecture like YOLO, Faster R-CNN, SSD, etc, the machine learning community often moves so quickly that someone beat you to the punch. Sure enough, after a quick Google search, I found [this repository](https://github.com/victordibia/handtracking) where [@victordibia](https://github.com/victordibia) used the Egohands Dataset to train an SSD to detect hands. He even included a script to detect hands live from a webcam stream (this will come in handy later). 

If you want to know more about the specifics of the implementation, I highly suggest reading the detailed and informative README for @victordibia's repository.


## Virtual Controller

Normally, creating a virtual game controller would require us to write our own driver so that other programs on the computer understand how to interpret the inputs from our controller. Thankfully, there already exists a project that streamlines this process. [vJoy](http://vjoystick.sourceforge.net/site/) is a device driver that will allow us to programmatically feed values to a virtual game controller that other programs on the computer can recognize. By using [@tidzo](https://github.com/tidzo)'s [pyvjoy](https://github.com/tidzo/pyvjoy), we can feed values to the driver directly from python!

> Note: vJoy and pyvjoy only work on Windows. After extensive searching, I was unable to find any comparably easy solutions for other operating systems.

## Usage

### Prerequisites

You will need to install Tensorflow and OpenCV and make sure they are working with your current python environment.

### Setup

1. Clone my repo

    `git clone --recurse-submodules https://github.com/seangtkelley/race-with-your-hands.git`

2. Install vJoy

    Follow the instructions on the site: http://vjoystick.sourceforge.net/site/

3. Copy .dll

    As mentioned in the README of the pyvjoy repository, we need to copy `vJoyInterface.dll` from the install directory of vJoy to the directory of our pyvjoy submodule. `vJoyInterface.dll` is most likely located in `C:\Program Files\vJoy\x86\` or `C:\Program Files\vJoy\x64\`.

4. Run `start_controller.py`

## Demo (Sorta)

Unfortunately, the best I have to show is the screenshots of the hand detections. At the moment, the pyvjoy library seems to only work the x86 version of the `vJoyInterface.dll`. Normally this would not be a problem as you could simply use a 32bit Python install. However, I was unable to get Tensorflow to work properly with a 32bit install of Python.

So we find ourselves at an impasse. I commented on an issue thread on the pyvjoy's repository and @tidzo expressed that it should most likely work with a 64bit .dll.

If anyone reading this has any luck with the 64bit .dll, please don't hesitate to let me know.

## Conclusion

I hope I was able to show a unique application of machine learning. I implore anyone exploring the territory to try to think of untraditional uses for the well-known models. It can serve as both a test of your knowledge and an exploration of fields you might not normally consider.
