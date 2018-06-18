---
layout: post
title: "Quick Exploration of Asana Data"
desc: "Since I've been using it for about three years now, I would say that statement is pretty accurate. Recently, it dawned on me the amount of potential insights data from Asana could provide about my productivity. Sure enough, there is an 'Export to CSV' but in the actions menu (click the down arrow next to the project name in the center of the screen) for every project! Let's see what lies in my (un)productivity data..."
tag: "Data Analysis"
author: "Sean Kelley"
thumb: "/img/blog/2018-06-05-asana-exploration/asanalogo.png"
date: 2018-06-05
---

If you have never heard of it, Asana is a pretty neat tool for keeping track of your personal tasks or managing the Agile workflow of a team. Here's what their landing pages says:

> Asana is the easiest way for teams to 
track their work—and get results.

Since I've been using it for about three years now, I would say that statement is pretty accurate. Recently, it dawned on me the amount of potential insights data from Asana could provide about my productivity. Sure enough, there is an "Export to CSV" but in the actions menu (click the down arrow next to the project name in the center of the screen) for every project!

Let's see what lies in my (un)productivity data...


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly import tools
import cufflinks as cf
from wordcloud import WordCloud
import nltk
import re
import string

nltk.download('stopwords')
cachedStopWords = nltk.corpus.stopwords.words("english")
punct = set(string.punctuation)

init_notebook_mode(connected=True)
cf.set_config_file(world_readable=True, offline=True)
```


    [nltk_data] Downloading package stopwords to /home/sean/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!


<script src="//cdn.plot.ly/plotly-latest.min.js"></script>


For my own sanity, I split my tasks into two projects: `School` for school related tasks and, `Tasks` for everything else. `My Personal Tasks` is the default personal tasks project and is what I used before I created the `Tasks` project.

Let's import the CSVs and concatenate them. I use the `verify_integrity` to make sure there aren't any duplicates from shifting tasks between projects.


```python
df1 = pd.read_csv('School.csv', parse_dates=[1, 2, 3, 7, 8])
df2 = pd.read_csv('Tasks.csv', parse_dates=[1, 2, 3, 7, 8])
df3 = pd.read_csv('My_Personal_Tasks.csv', parse_dates=[1, 2, 3, 7, 8])
frames = [df1, df2, df3]
df = pd.concat(frames, verify_integrity=True, ignore_index=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table class="dataframe table table-striped">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Task ID</th>
      <th>Created At</th>
      <th>Completed At</th>
      <th>Last Modified</th>
      <th>Name</th>
      <th>Assignee</th>
      <th>Assignee Email</th>
      <th>Start Date</th>
      <th>Due Date</th>
      <th>Tags</th>
      <th>Notes</th>
      <th>Projects</th>
      <th>Parent Task</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>148623786710031</td>
      <td>2018-04-15</td>
      <td>2018-04-15</td>
      <td>2018-04-15</td>
      <td>More debt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>148623786710030</td>
      <td>2018-04-06</td>
      <td>2018-04-07</td>
      <td>2018-04-07</td>
      <td>Send one line email to erik to add you to the ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>610060357624798</td>
      <td>2018-03-27</td>
      <td>2018-03-27</td>
      <td>2018-03-27</td>
      <td>withdraw from study abroad</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>588106896688257</td>
      <td>2018-03-09</td>
      <td>2018-03-26</td>
      <td>2018-03-26</td>
      <td>hold</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>570162249229318</td>
      <td>2018-02-22</td>
      <td>2018-02-23</td>
      <td>2018-02-23</td>
      <td>find joydeep</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>2018-02-23</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Projects'].value_counts(dropna=False)
```




    School          766
    Tasks           321
    NaN             160
    Hampstead WX     17
    Name: Projects, dtype: int64



I found interesting that the `My Personal Tasks` pulled in some tasks from an archived project called `Hampstead WX`. That project was [a site](http://hampsteadwx.herokuapp.com/) I created a while ago.

---

Here I plot a histogram of the day of the week the tasks were created on.


```python
dayOfWeek={0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
df['Created At DOW'] = df['Created At'].dt.dayofweek.map(dayOfWeek)
```


```python
x = []
y = []
for key, day in dayOfWeek.items():
    x.append(day)
    y.append(df['Created At DOW'].value_counts()[day])
bar_tracer = [go.Bar(x=x, y=y)]
iplot(bar_tracer)
```


<div id="1232ca98-668a-49f6-8b7f-01bb627bfb7f" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("1232ca98-668a-49f6-8b7f-01bb627bfb7f", [{"type": "bar", "x": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "y": [259, 231, 198, 167, 125, 99, 185]}], {}, {"showLink": true, "linkText": "Export to plot.ly"});</script>


I can't say I didn't expect these results. The bulk of my tasks are created at the start of the week and they slowly fade off until Saturday.

---

Next, let's look at the duration it took for me to complete each task. Because I used the `parse_dates` parameter when importing the CSVs, using the minus operator will return [timedelta objects](https://docs.python.org/2/library/datetime.html#timedelta-objects). Since Asana only provided dates without time, tasks with a duration of 0 days are ones that were created and completed on the same day.


```python
df['Duration'] = (df['Completed At'] - df['Created At'])
```


```python
x = df['Duration'].value_counts().keys().days
y = list(df['Duration'].value_counts().values)
bar_tracer = [go.Bar(x=x, y=y)]
iplot(bar_tracer)
```


<div id="801ac881-f8d2-49a3-bbeb-12d901d7604e" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("801ac881-f8d2-49a3-bbeb-12d901d7604e", [{"type": "bar", "x": [1, 0, 2, 4, 3, 5, 7, 6, 8, 9, 12, 11, 10, 13, 15, 14, 27, 29, 19, 23, 17, 30, 254, 16, 21, 50, 25, 69, 33, 102, 22, 18, 20, 62, 26, 40, 32, 45, 91, 101, 46, 80, 114, 179, 55, 70, 104, 51, 48, 31, 129, 64, 28, 323, 41, 276, 248], "y": [264, 226, 152, 92, 89, 73, 49, 33, 20, 18, 17, 16, 15, 12, 11, 7, 6, 6, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}], {}, {"showLink": true, "linkText": "Export to plot.ly"});</script>


Wow! Seems like we have some outliers. Let's see what some of them are.


```python
df[df['Duration'].astype('timedelta64[D]') > 75]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table class="dataframe table table-striped">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Task ID</th>
      <th>Created At</th>
      <th>Completed At</th>
      <th>Last Modified</th>
      <th>Name</th>
      <th>Assignee</th>
      <th>Assignee Email</th>
      <th>Start Date</th>
      <th>Due Date</th>
      <th>Tags</th>
      <th>Notes</th>
      <th>Projects</th>
      <th>Parent Task</th>
      <th>Created At DOW</th>
      <th>Duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>151</th>
      <td>425397822367084</td>
      <td>2017-09-08</td>
      <td>2017-12-19</td>
      <td>2017-12-19</td>
      <td>Comp Sci 220</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
      <td>Fri</td>
      <td>102 days</td>
    </tr>
    <tr>
      <th>166</th>
      <td>425397822367086</td>
      <td>2017-09-08</td>
      <td>2017-12-21</td>
      <td>2017-12-21</td>
      <td>Comp Sci 240</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
      <td>Fri</td>
      <td>104 days</td>
    </tr>
    <tr>
      <th>184</th>
      <td>425397822367087</td>
      <td>2017-09-08</td>
      <td>2017-12-19</td>
      <td>2017-12-19</td>
      <td>Physics 151</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
      <td>Fri</td>
      <td>102 days</td>
    </tr>
    <tr>
      <th>296</th>
      <td>425397822367083</td>
      <td>2017-09-08</td>
      <td>2017-12-19</td>
      <td>2017-12-19</td>
      <td>Stats 515</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
      <td>Fri</td>
      <td>102 days</td>
    </tr>
    <tr>
      <th>344</th>
      <td>177397329054805</td>
      <td>2016-09-06</td>
      <td>2017-07-26</td>
      <td>2017-07-26</td>
      <td>Misc:</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
      <td>Tue</td>
      <td>323 days</td>
    </tr>
    <tr>
      <th>362</th>
      <td>148623786709961</td>
      <td>2017-01-26</td>
      <td>2017-07-24</td>
      <td>2017-07-24</td>
      <td>al dimeola - egyptian danza LISTEN AND LEARN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
      <td>Thu</td>
      <td>179 days</td>
    </tr>
    <tr>
      <th>419</th>
      <td>200785803637408</td>
      <td>2016-10-23</td>
      <td>2017-07-26</td>
      <td>2017-07-26</td>
      <td>watch shit</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>howls moving castle\nnausicaa\nyeh jawaani hai...</td>
      <td>School</td>
      <td>NaN</td>
      <td>Sun</td>
      <td>276 days</td>
    </tr>
    <tr>
      <th>512</th>
      <td>177397329054770</td>
      <td>2016-09-06</td>
      <td>2017-05-18</td>
      <td>2017-05-18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
      <td>Tue</td>
      <td>254 days</td>
    </tr>
    <tr>
      <th>540</th>
      <td>177397329054778</td>
      <td>2016-09-06</td>
      <td>2017-05-18</td>
      <td>2017-05-18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
      <td>Tue</td>
      <td>254 days</td>
    </tr>
    <tr>
      <th>638</th>
      <td>177397329054780</td>
      <td>2016-09-06</td>
      <td>2017-05-18</td>
      <td>2017-05-18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
      <td>Tue</td>
      <td>254 days</td>
    </tr>
    <tr>
      <th>668</th>
      <td>179665031917793</td>
      <td>2016-09-12</td>
      <td>2017-05-18</td>
      <td>2017-05-18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
      <td>Mon</td>
      <td>248 days</td>
    </tr>
    <tr>
      <th>692</th>
      <td>177397329054776</td>
      <td>2016-09-06</td>
      <td>2017-05-18</td>
      <td>2017-05-18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
      <td>Tue</td>
      <td>254 days</td>
    </tr>
    <tr>
      <th>724</th>
      <td>207879281133328</td>
      <td>2016-11-06</td>
      <td>2017-02-15</td>
      <td>2017-02-15</td>
      <td>figure out piano chords from HATE verses</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
      <td>Sun</td>
      <td>101 days</td>
    </tr>
    <tr>
      <th>733</th>
      <td>201648184747252</td>
      <td>2016-10-24</td>
      <td>2017-01-23</td>
      <td>2017-01-23</td>
      <td>transcribe good voice recordings</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
      <td>Mon</td>
      <td>91 days</td>
    </tr>
    <tr>
      <th>734</th>
      <td>201648184747254</td>
      <td>2016-10-24</td>
      <td>2017-02-15</td>
      <td>2017-02-15</td>
      <td>transcribe dank part from Skin Deep</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
      <td>Mon</td>
      <td>114 days</td>
    </tr>
    <tr>
      <th>1058</th>
      <td>432400137331715</td>
      <td>2017-09-18</td>
      <td>2017-12-07</td>
      <td>2017-12-07</td>
      <td>hit up Ben for the db seeds</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Tasks</td>
      <td>NaN</td>
      <td>Mon</td>
      <td>80 days</td>
    </tr>
    <tr>
      <th>1238</th>
      <td>95110234477969</td>
      <td>2016-02-27</td>
      <td>2016-07-05</td>
      <td>2016-07-05</td>
      <td>Switch to google charts</td>
      <td>Sean Kelley</td>
      <td>seangtkelley@gmail.com</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Hampstead WX</td>
      <td>NaN</td>
      <td>Sat</td>
      <td>129 days</td>
    </tr>
  </tbody>
</table>
</div>



From what I can tell, about half of these are not actually tasks but rather sections. Asana interestingly still allows you to mark a section as complete which is why my sections for classes like `Comp Sci 220` and `Stats 515` show up.

The others are a mixed bag. A lot are music projects I was working that obviously took a while. Another task was a global task I had with a running list of all the movies I need to watch.

---

Let's go deeper into the histogram. Here, I split the bars by their respective project and only look at the tasks that took less than 30 days to complete.


```python
trace1 = go.Bar(
    x=df[(df['Duration'].astype('timedelta64[D]') < 30) & (df['Projects'] == 'School')]['Duration'].value_counts().keys().days,
    y=df[(df['Duration'].astype('timedelta64[D]') < 30) & (df['Projects'] == 'School')]['Duration'].value_counts().values,
    name='School'
)
trace2 = go.Bar(
    x=df[(df['Duration'].astype('timedelta64[D]') < 30) & (df['Projects'] == 'Tasks')]['Duration'].value_counts().keys().days,
    y=df[(df['Duration'].astype('timedelta64[D]') < 30) & (df['Projects'] == 'Tasks')]['Duration'].value_counts().values,
    name='Tasks'
)
data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')
```


<div id="df006ef8-af8f-4b96-86d8-334eef33ba6a" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("df006ef8-af8f-4b96-86d8-334eef33ba6a", [{"type": "bar", "x": [1, 0, 2, 3, 4, 5, 7, 6, 11, 8, 10, 9, 12, 15, 29, 13, 17, 14, 16, 19, 22, 23, 21, 25, 27, 26], "y": [157, 140, 103, 67, 60, 49, 32, 24, 13, 13, 9, 8, 8, 7, 6, 5, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1], "name": "School"}, {"type": "bar", "x": [1, 0, 2, 5, 3, 7, 4, 6, 13, 8, 9, 12, 27, 10, 14, 15, 21, 18, 19, 20, 25, 11, 28, 26, 16, 23, 17], "y": [75, 65, 32, 20, 14, 14, 11, 9, 7, 7, 7, 7, 4, 4, 4, 4, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1], "name": "Tasks"}], {"barmode": "group"}, {"showLink": true, "linkText": "Export to plot.ly"});</script>


It seems between both projects, the distribution of the tasks is very similar. Most tasks are completed within the two weeks after creation. This result definitely highlights the intended purpose of Asana (or at least how I use it). Personally, any deadline that is more than a month away I usually put in Google Calendar, rather than create a task in Asana.

---

Next, let's see if we can figure out what type of tasks usually take longer to complete. I will once again use the fantastic [word_cloud library](http://amueller.github.io/word_cloud/index.html) by [amueller](http://amueller.github.io/).


```python
# concatenate all name fields from tasks separated by duration of 3 days
less_text = ' '.join(list(df[df['Duration'].astype('timedelta64[D]') < 3]['Name'].dropna()))
grtr_text = ' '.join(list(df[df['Duration'].astype('timedelta64[D]') >= 3]['Name'].dropna()))

# remove any punctuation
less_text = ''.join(ch for ch in less_text if ch not in punct)
grtr_text = ''.join(ch for ch in grtr_text if ch not in punct)

# remove stopwords
less_text = ' '.join([word for word in less_text.split() if word not in cachedStopWords])
grtr_text = ' '.join([word for word in grtr_text.split() if word not in cachedStopWords])

# create wordclouds
less_wordcloud = WordCloud(background_color="white", max_words=1000, margin=10,random_state=1).generate(less_text)
grtr_wordcloud = WordCloud(background_color="white", max_words=1000, margin=10,random_state=1).generate(grtr_text)

# display wordclouds using matplotlib
f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
f.set_size_inches(18, 5)
ax1.imshow(less_wordcloud, interpolation="bilinear")
ax1.set_title('<3 days', fontsize=36)
ax1.axis("off")
ax2.imshow(grtr_wordcloud, interpolation="bilinear")
ax2.set_title('>=3 days', fontsize=36)
ax2.axis("off")
```




    (-0.5, 399.5, 199.5, -0.5)




![png](/img/blog/2018-06-05-asana-exploration/output_16_1.png)


This is _by far_ my favorite visualization. Right away we can see the difference between long and short term tasks. 

Some of the top words for the less than three days category are `email`, `note`, `add`, `read`, and `print`. Each of these tasks are relatively simple and usually require at most a few actions.

Some of the top words for the greater than or equal to three days category are `project`, `quiz`, `note`, `create`, `make`, and `research`. In contrast, each one of these tasks evidently require more effort and take a lot longer.

It's interesting that `note` was a top word in both wordclouds. I can only assume that notes are not high priority for me so they are equally likely to be done immediately or postponed.

---

Next, let's take a look at overdue tasks. I haven't started putting due dates on tasks until recently so there isn't as much good data but we'll work with what we've got.

Like the `Duration` field, I again use the `timedelta` functionality for the `datetime` objects. In this case, positive deltas will represent tasks completed after the due date and negative deltas tasks completed before the due date.


```python
df['Overdue'] = df['Completed At'] - df['Due Date']
```


```python
trace1 = go.Bar(
    x=df[(df['Projects'] == 'School')]['Overdue'].value_counts().keys().days,
    y=df[(df['Projects'] == 'School')]['Overdue'].value_counts().values,
    name='School'
)
trace2 = go.Bar(
    x=df[(df['Projects'] == 'Tasks')]['Overdue'].value_counts().keys().days,
    y=df[(df['Projects'] == 'Tasks')]['Overdue'].value_counts().values,
    name='Tasks'
)
data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')
```


<div id="bda7b59d-9ab7-4275-88c9-18205834d2e4" style="height: 525px; width: 100%;" class="plotly-graph-div"></div>
<script type="text/javascript">
  Plotly.newPlot("bda7b59d-9ab7-4275-88c9-18205834d2e4",
                 [{"type": "bar",
                   "x": [0, -1, -2, -3, 1, -5, -4, -6, -7, -9, -8, 2, -10, 6, 4, 5, 11, -12, -28, 21, -11, -15, 23, -14],
                   "y": [131, 51, 29, 25, 24, 18, 17, 13, 7, 5, 5, 4, 3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
                   "name": "School"
                  }, 
                  {"type": "bar",
                   "x": [0, 1, -5, 2, -4, -1, -2, 30],
                   "y": [35, 14, 6, 3, 2, 2, 1, 1], 
                   "name": "Tasks"}
                 ], 
                 {"barmode": "group"}, 
                 {"showLink": true, 
                  "linkText": "Export to plot.ly"}
                );
  </script>


I was happy to see that for the most part, I complete tasks before their set due dates.

---

Let's do a similar wordcloud analysis of the overdue tasks to see what might be causing the delay in completion. Here, I'm going to split the tasks into three sections: completed more than one day before due date, completed on same day as due date, and completed after due date.


```python
# concatenate all name fields from overdue tasks
before_text = ' '.join(list(df[df['Overdue'].astype('timedelta64[D]') < 0]['Name'].dropna()))
sameday_text = ' '.join(list(df[df['Overdue'].astype('timedelta64[D]') == 0]['Name'].dropna()))
overdue_text = ' '.join(list(df[df['Overdue'].astype('timedelta64[D]') > 0]['Name'].dropna()))

# remove any punctuation
before_text = ''.join(ch for ch in before_text if ch not in punct)
sameday_text = ''.join(ch for ch in sameday_text if ch not in punct)
overdue_text = ''.join(ch for ch in overdue_text if ch not in punct)

# remove stopwords
before_text = ' '.join([word for word in before_text.split() if word not in cachedStopWords])
sameday_text = ' '.join([word for word in sameday_text.split() if word not in cachedStopWords])
overdue_text = ' '.join([word for word in overdue_text.split() if word not in cachedStopWords])

# create wordclouds
before_wordcloud = WordCloud(background_color="white", max_words=1000, margin=10,random_state=1).generate(before_text)
sameday_wordcloud = WordCloud(background_color="white", max_words=1000, margin=10,random_state=1).generate(sameday_text)
overdue_wordcloud = WordCloud(background_color="white", max_words=1000, margin=10,random_state=1).generate(overdue_text)

# display wordclouds using matplotlib
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
f.set_size_inches(18, 10)
ax1.imshow(before_wordcloud, interpolation="bilinear")
ax1.set_title('Completed Before', fontsize=36)
ax1.axis("off")
ax2.imshow(sameday_wordcloud, interpolation="bilinear")
ax2.set_title('Completed Same Day', fontsize=36)
ax2.axis("off")
ax3.imshow(overdue_wordcloud, interpolation="bilinear")
ax3.set_title('Overdue', fontsize=36)
ax3.axis("off")
ax4.axis("off")
```




    (-0.5, 399.5, 0.0, 1.0)




![png](/img/blog/2018-06-05-asana-exploration/output_21_1.png)


As expected, we some similar words to those in the Duration analysis but there are also some new ones.

In the completed before wordcloud, the words `tarea` and `Voces` come up which are actually Spanish for `homework` and `Voices` respectively. I study Spanish at University and for some reason, I always write tarea instead of homework just for homework in my Spanish classes. And Voces was the name of a latin american poetry textbook we had one semester in which we often had reading assignments.

Probably the most revealing worldcloud is the overdue one. It's clear that I have a problem with `start`ing tasks.

## Conclusion
Hopefully this post shows just a glimpse of the insights you can gleen from diving into your own metadata. Many productivity applications have options for downloading your data in bulk and I would implore you to give it a shot. Perhaps you can use your discoveries to improve your work habits to become or lifestyle!
