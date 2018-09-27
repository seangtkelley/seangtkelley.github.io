---
layout: post
title: "Visualizing Sexism in Conan O'Brien's YouTube Video Titles Using Wordclouds"
desc: "Late night television video clips dominate my consumption habits on youtube. While watching these hosts, I started to notice O'Brien's videos appearing in the recommended videos. At first, I dismissed them as normal considering I was watching a considerable amount of late night shows. However, I began to notice a difference in the way O'Brien (or most likely his team) titles the interview videos. When the guest is a man, the titles seem normal. It is usually something to do with a book they're releasing or a movie they're in. But when the host is a woman, the titles are noticeably more sexual and provocative."
tag: "Data Analysis"
author: "Sean Kelley"
thumb: "/img/blog/2018-04-07-late-night-sexism/conan-banner.jpg"
date: 2018-04-07
---

I'll admit I a spend considerable amount of time on YouTube. I consume mountains of videos ranging from [eurobeat mixes](https://www.youtube.com/watch?v=cq4s8hrTNT8), to [dash cam montages](https://www.youtube.com/user/DashCamOwnersAustral), to [tensorflow tutorials](https://www.youtube.com/watch?v=2FmcHiLCwTU), to [political video essays](https://www.youtube.com/watch?v=EvXROXiIpvQ). However, some of my favorite content are the clips posted by late night television shows. Colbert, Meyers, Noah, and Oliver always put a funny spin on modern politics and present some compelling interviews.

While watching these hosts, I started to notice O'Brien's videos appearing in the recommended videos. At first, I dismissed them as normal considering I was watching a considerable amount of late night shows. However, I began to notice a difference in the way O'Brien (or most likely his team) titles the interview videos. When the guest is man, the titles seem normal, usually something to do with a book they're releasing or a movie their in. But when the host is woman, the titles are noticeably more sexual and provocative.

Instead of just ignoring it, I thought I would quantify my observations with some visualizations.

Let's first import all the goodies we'll need.


```python
import pafy
import pandas as pd
import numpy as np
from PIL import Image
import os
import gender_guesser.detector as gender
from wordcloud import WordCloud
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from tqdm import tqdm
import string
```

Next, I use the [pafy library](http://pythonhosted.org/Pafy/) to extract the metadata from the playlist of all O'Brien's videos. I believe the pafy library is essentially a wrapper for the [youtube-dl](https://rg3.github.io/youtube-dl/) command line tool. The library's usage doesn't quite match the documentation but it's good enough for what we're doing.

Unfortunately, O'Brien doesn't have a playlist of all the interviews so we'll have to figure out which videos are interviews using the metadata.


```python
plurl = "https://www.youtube.com/playlist?list=UUi7GJNg51C3jgmYTUwqoUXA"
playlist = pafy.get_playlist2(plurl)
```


```python
playlist
```




    Type: Playlist
    Title: Uploads from Team Coco
    Author: Team Coco
    Description: 
    Length: 6658




```python
playlist[21].title
```




    'Stream Coco LIVE: "Attack On Titan 2" & "Oxenfree"'



Next, in order to determine if the video is an interview or not, we will [use a list of names collected from the census](https://stackoverflow.com/questions/32545180/from-of-list-of-strings-identify-which-are-human-names-and-which-are-not). The process will of course not be entirely accurate because of annomalies in the titles. For example, not all interviews start with the name of the guest and "The" is technically a name.

Take note that in the original stackoverflow post, one of the files is in table format and not csv. I was having trouble importing that with pandas so I simply used Google Sheets to convert it to csv.

In addition to using the name check, we will also ignore videos with a forward slash "/" in the title. Those videos are almost always band performances.

To determine the gender of the name, we can use the [gender guesser](https://testpypi.python.org/pypi/gender-guesser) library. In our case, androgynous names will be ignored.

Finally, we can remove common phrases from the titles like " - CONAN on TBS" which is at the end of every one of his titles.


```python
census_surnames = pd.read_csv('app_c.csv')
census_firstnames = pd.read_csv('census-derived-all-first.csv')
```


```python
firsts_lower = [str(name).lower() for name in list(census_firstnames.name)]
lasts_lower = [str(name).lower() for name in list(census_surnames.name)]
```


```python
d = gender.Detector()

male_titles = []
female_titles = []
for i in tqdm(range(6647)):
    try:
        firstname = playlist[i].title.split()[0].lower()
        lastname = playlist[i].title.split()[1].lower()
        
        video_title = ' '.join(playlist[i].title.split()[2:])
        video_title = video_title.replace(" - CONAN on TBS", "")
        video_title = video_title.replace("Conan", "")
        # print(video_title)

        if firstname in firsts_lower and lastname in lasts_lower and "/" not in video_title:
            # print(firstname, d.get_gender(firstname.capitalize()))
            if 'female' in d.get_gender(firstname.capitalize()):
                female_titles.append(video_title)
            elif 'male' in d.get_gender(firstname.capitalize()):
                male_titles.append(video_title)
    except Exception:
        continue
```

    100%|██████████| 6647/6647 [04:27<00:00, 24.86it/s]


Now that we have all the titles of the interviews separated, we can remove the punctuation and subsequently the stopwords.


```python
cachedStopWords = stopwords.words("english")
punct = set(string.punctuation)

female_text = ' '.join(female_titles)
male_text = ' '.join(male_titles)

female_text = ''.join(ch for ch in female_text if ch not in punct)
male_text = ''.join(ch for ch in male_text if ch not in punct)

female_text = ' '.join([word for word in female_text.split() if word not in cachedStopWords])
male_text = ' '.join([word for word in male_text.split() if word not in cachedStopWords])
```

Finally, we can create the wordclouds using [amueller](http://amueller.github.io/)'s [word_cloud library](http://amueller.github.io/word_cloud/index.html).

To enhance the visualization, I masked the word clouds with the gender signs commonly used for bathrooms in the US.


```python
female_mask = np.array(Image.open(os.path.join(os.getcwd(), "female_mask.jpg")))
male_mask = np.array(Image.open(os.path.join(os.getcwd(), "male_mask.jpg")))

female_wordcloud = WordCloud(background_color="white", max_words=1000, mask=female_mask, margin=10,random_state=1).generate(female_text)
male_wordcloud = WordCloud(background_color="white", max_words=1000, mask=male_mask, margin=10,random_state=1).generate(male_text)
```


```python
f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
f.set_size_inches(18, 20.0)
ax1.imshow(male_wordcloud, interpolation="bilinear")
ax1.set_title('Male Wordcloud', fontsize=36)
ax1.axis("off")
ax2.imshow(female_wordcloud, interpolation="bilinear")
ax2.set_title('Female Wordcloud', fontsize=36)
ax2.axis("off")
```




    (-0.5, 699.5, 1463.5, -0.5)




![png](/img/blog/2018-04-07-late-night-sexism/conan_comparison.png)


## Conclusion

I think the wordclouds speak for themselves. While "Sex" and "Sexy" are among the top words for women, they barely come up for men. Perhaps the biggest irony is that "Man" is a top word for women. One of the more off-putting insights is that "Naked", "Boobs", and "Butt" are all common enough to appear in the wordcloud for women, but I cannot find a single body part word in the wordcloud for men.

This idea started a just a hunch but after quantifying it, it's evident that O'Brien's team purposely sexualizes the video titles for interviews with women guests. Although this is not a new practice on YouTube, a mainstream late night host's channel should not be intentionally sexualizing their content that features a woman guest.

Furthermore, this is not a common practice. Here are wordclouds I generated using the same method for Colbert and Fallon.

## Colbert
![Colbert Comparison](/img/blog/2018-04-07-late-night-sexism/colbert_comparison.png "Colbert Comparison")

## Fallon
![Fallon Comparison](/img/blog/2018-04-07-late-night-sexism/fallon_comparison.png "Fallon Comparison")
