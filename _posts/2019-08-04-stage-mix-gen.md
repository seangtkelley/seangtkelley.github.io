---
layout: post
title: "Finding Interesting Ways to Use Programming: A Kpop Stage Mix Generator"
desc: "Often times I am disappointed when programming as portrayed in the mainstream is relagated to cliché problems. Don't get me wrong.
Programming can solve many compelling problems. But, I believe that many people are made uninterested because they think it's not for them. In this post, 
I try to show a novel application of programming that I haven't seen before."
tag: "Creative"
author: "Sean Kelley"
thumb: "/img/blog/2019-08-04-stage-mix-gen/Girls-Generation-I-Got-A-Boy.jpg"
date: 2019-08-04
---

If you pay any attention to pop music nowadays, you probably have noticed the larger precense of Kpop. Worldwide, Kpop is gaining popularity out of Asia. Most notebly, BTS has had multiple albums in the Billboard World Albums chart since last summer in addition to making the rounds on late night television. Another popular group, Blackpink, performed at Coachella. Although the industry has many issues, I personally enjoy Kpop. I'm not going to try and defend that position as I have failed many times before, but at least now with it's popularity, I know I'm not alone. So where does a programming project fit into all of this? Well, there's a paticular part of Kpop that reliably recurs and any time there's a predictable pattern, there's always a cool project.

Essentially, the mainstream Kpop industry promotes their groups in cycles. I'm no expert but here's how I understand it. Groups will first build hype for a new single or album by releasing teaser images and videos. Then, once the song or album is dropped, the fans will start streaming and the group will already have multiple performances lined up. However, these aren't like traditional tours. The groups have to perform their song at well know events like Inkigayo or M!Countdown, which occur regularly as well. Their performances from each of these shows is now almost guaranteed to be uploaded to YouTube.

Given that the correography for the song does not change, the wealth of footage makes for great montages. This has created a whole new community on YouTube where users will create "stage mixes" of the different performances while a group was promoting. At a certain point, while being a comsumer of this content, I thought to myself, these mixes definitely could be done automatically. All you had to do was align the audio and make the cuts. What could possibly be hard about that? I would soon find the answer to my question as the day after I set out to see if I could do it.

It took a while to work through many bugs and bag logic (I swear I'm good at programming usually), but I eventually succeeded. Meet Stage Mix Generator 1.0.

## Algorithm

The highlevel algorithm is as follows:

1. Retrieve stage videos and audios
2. Retrieve song audio
3. Align stage videos to audios
4. Find cuts in stage videos
4. Create mix using vuts
5. Render mix video
6. Upload to YouTube

The most interesting parts of the algorithm come with aligning the stage videos with the song audio and cutting the mix to match the stage videos.

## Aligning Stage Videos to Song Audio

One thing that I believe I've mentioned before in my blog posts is that one of the most crucial skills a programmer should have is an intuitive sense of when a problem likely has implemented solutions and how to find them. Given the generality of my problem and the likelyhood a solution would be useful in many fields, I started to search. Right away, I found this:

![png](https://github.com/allisonnicoledeal/VideoSync/raw/master/screenshots/screenshot.png)

Well you don't say! This is almost exactly what I'm already trying to do. [VideoSync]() is a project by [Allison Deal]() which "automatically synchronizes and combines personal and crowd-sourced YouTube video clips to recreate a live concert experience from multiple angles."

Here's a description of the program from the repo's [README]():
> * **YouTube Link**: Download YouTube videos as MP4 files with youtube-dl command line program.
> * **WAV File**: Strip audio from video file using the avconv audio/video converter. Read audio data using the Python scipy library.
> * **Fourier Transform of Audio Signal**: Split audio into bins and apply the Fourier transform on each bin using the numpy library. The Fourier transform converts each bin data from the time domain to the frequency domain.
> * **Peak Frequencies**: Identify the frequency with the highest intensity in each bin to create a peak frequency constellation.
> * **Frequency Constellation Alignment**: Determine time offset by aligning frequency constellations of the two audio files.

This is super close to what I was trying to do, however with a few subtle differences:

1. It downloads the videos in MP4 format which with YouTube limits you to 720p
2. It's designed to sync videos of the same performance
3. The performances are meant to be entire concerts, not single songs
4. The final video is not edited

The key difference here is the final product. Deal's goal was to give a live concert experience by showing multiple angles at the same time. I was looking to edit different stage performances into one video, similar to the stage mixes on YouTube. Even with these differences, I knew I could use the underlying code that aligned the videos. Unfortunately, VideoSync has not been updated since 2014 so I struggled to get any of the python to run without errors. After some more googling, I found that the main function in the VideoSync code was adapted to be used in the [cvcalib]() library. Since this library had been updated more recently, the code worked out of the box. If anyone is curious, the function I use can be located [here](https://github.com/Algomorph/cvcalib/blob/4ed638ea523b6d1059556a135576c7afa3a4b07f/audiosync/offset.py#L172).

So now that I could align the videos, I needed to edit them.

## Cutting Mix to Match Video Clip Cuts

### Minimum Viable Product: Cut every 5 seconds

Finding a good library to edit the mixes was actually a lot easier than I expected. Through a quick search, I found [MoviePy](https://github.com/Zulko/moviepy). Here's an excerpt the description from the repository:

> MoviePy (full [documentation](http://zulko.github.io/moviepy/)) is a Python library for video editing: cutting, concatenations, title insertions, video compositing (a.k.a. non-linear editing), video processing, and creation of custom effects.

Using the library, I was easily able to make a video that was cut every 5 seconds. Below is an example video that the algorithm created using the code thus far. Note I had not yet built the upload to YouTube feature.


<iframe src="https://www.youtube.com/embed/bx9MCpnnja8" style="width: 100%; height: 400px" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


Pretty cool, right? I sure thought so. This was definitely my minimum viable product (MVP). I now had a working prototype of the system I envisioned earlier. However, I knew I could do better and so I set out to find a method of detecting the cuts in the stage performance videos.

### Stretch Goal: Detect and match cuts of stage videos used

Although I quickly found a library that could detect cuts in videos, installing OpenCV on Windows turned into a nightmare and delayed my completion of the project rather significantly. Nonetheless, here's how I did it.

The library I used is called [PySceneDetect](https://github.com/Breakthrough/PySceneDetect). It provides two main algorithms for detecting the cuts in videos. The Content Detector "compares the difference in content between adjacent frames against a set threshold/score, which if exceeded, triggers a scene cut." The Threshold Detector "uses a set intensity level to detect scene cuts when the average frame intensity passes the set threshold." The Content Detector algorithm is rather self-explanatory but the Threshold Detector algorithm is a little more nuanced. Essentially, it is mostly good for finding cuts where the video fades to black. Since none of the cuts in the stage videos were fades, I needed to use the Content Detector. More information about these algorithms can be found in [their documentation](https://pyscenedetect-manual.readthedocs.io/en/latest/cli/detectors.html).


To determine the threshold at which the algo would register a cut, I followed the instructions in the documentation:

> The optimal threshold can be determined by generating a stats file (-s), opening it with a spreadsheet editor (e.g. Excel), and examining the content_val column. This value should be very small between similar frames, and grow large when a big change in content is noticed (look at the values near frame numbers/times where you know a scene change occurs). The threshold value should be set so that most scenes fall below the threshold value, and scenes where changes occur should exceed the threshold value (thus triggering a scene change).

For the stage videos, the default value of 30 was a bit too low so I chose to use 40. From this point, my implementation is quite simple. First, I find all the cuts in each video to be used in the mix. Then, I make the cuts by repeatedly chosing a random video and finding next cut from the current timestamp the loop is at. This algorithm is clearly $O(n^2)$ but its efficiency should be fine for now. Below is an example video that was edited using this method.

<iframe src="https://www.youtube.com/embed/YZC8a0AqG4o" style="width: 100%; height: 400px" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Now that the resulting stage mix was satisfactory (at least to me), the final step was to automatically share my creation.

## Uploading to YouTube

This ended up being the easiest part. Although, [Tokland](https://github.com/tokland)'s [youtube-upload](https://github.com/tokland/youtube-upload) doesn't provide an easily importable module, it does have a command line tool that can be run using the [subprocess](https://docs.python.org/3/library/subprocess.html) built-in library. Before using, I followed the instructions in the [setup](https://github.com/tokland/youtube-upload#setup) section of the README. Once the correct configuration files were in place, I created a small wrapper and setup the upload to trigger after the mix is rendered.

## Installation

###  Prerequisites

- [OpenCV](https://opencv.org/)
- [FFmpeg](https://ffmpeg.org/)
- [Python >3.6](https://www.python.org)
- [Anaconda](https://anaconda.org/) (optional)

### Clone Repository

`git clone --recurse-submodules https://github.com/seangtkelley/stage-mix-generator && cd stage-mix-generator`

### Create Python Environment

`pip install requirements.txt` 

**or** 
    
`conda env create -f requirements.yml`

**or**

Import the environment via the Anaconda GUI.

**Note:** You must use Anaconda on Windows because Scipy cannot be installed with `pip`.

## Usage

Similar to VideoSync, I also created a flask app, although mine is much less ornate. Open a terminal window with your environment enabled. I usually do this from the Anaconda GUI by navigating to Environments, clicking on the `stage-mix-generator` environment, clicking the play button, and finally clicking `Open Terminal`.

#### Mac and Linux

```sh
export FLASK_APP=web_ui.py
flask run
```

#### Windows Terminal

```sh
set FLASK_APP=web_ui.py
flask run
```

Here are some screenshots of the app. I think the form should be relatively self-explanatory.

![png](/img/blog/2019-08-04-stage-mix-gen/web_ui_screenshot.png)

## Conclusion

All in all, I wanted to share this project because it gets at the heart of why I love programming. Commonly, programming, and many STEM-related fields, are routinely dismissed by the mainstream creative community. So often have I heard programming, electrical engineering, mathematics, etc. dismissed as only things that nerds do. However, these fields offer enourmous creative potential. Through my blog, I have shown examples of applying programming to topics ranging from personal health to playing video games to predicting the stock market.

As I see it, programming is a tool that can enhance and complement your creative endevours. A huge milestone in bringing more people and more diverse perspectives into the space will be for the current community to dismantle the preconception that programming is for nerds. Artists ~~can be~~ [*are*](https://www.ilm.com/) programmers. Musicians ~~can be~~ [*are*](https://en.wikipedia.org/wiki/MikuMikuDance) programmers. I could go on but I think you get the point.

For all those who think programming is not for them, I sincerely hope you reconsider.