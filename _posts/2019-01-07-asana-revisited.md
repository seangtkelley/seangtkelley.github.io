---
layout: post
title: "Asana Data Revisited: Fall 2018 Semester In Review"
desc: "I previously did a quick analysis of the data from Asana, the website I use to keep track of all my tasks. \n\n Since it's been about 6 months since then, I thought it would be interesting to take another look to see which trends remained and which changed."
tag: "Data Analysis"
author: "Sean Kelley"
thumb: "/img/blog/2018-06-05-asana-exploration/asanalogo.png"
date: 2019-01-07
---

I previously did a [quick analysis](http://seangtkelley.me/blog/2018/06/05/asana-exploration) of the data from Asana, the website I use to keep track of all my tasks.

Since it's been about 6 months, I thought it would be interesting to take another look to see which trends remained and which changed.


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
import datetime
import collections
from lib import custom_utils

init_notebook_mode(connected=True)
cf.set_config_file(world_readable=True, offline=True)
```


    [nltk_data] Downloading package stopwords to /home/sean/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!


<script src="//cdn.plot.ly/plotly-latest.min.js"></script>

Notice I import a module called `custom_utils`. I recently started a project to keep track of my personal health and development and I found the need to reuse functions often. The file can be found [here](https://github.com/seangtkelley/personal-health-tracking/blob/master/lib/custom_utils.py).

### Data

Unlike last time, I'll only be pulling the tasks related to school. Just before the start of last semester (Fall 2018), I created a new project using the [Boards](https://asana.com/guide/help/views/boards) layout. This allows me to better organize my task by class.

First, let's load that data.


```python
f18_df = pd.read_csv('asana-umass-f18.csv', parse_dates=[1, 2, 3, 8, 9])
f18_df.head()
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
      <th>Column</th>
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
      <td>949672124046393</td>
      <td>2018-12-17</td>
      <td>2018-12-19</td>
      <td>2018-12-19</td>
      <td>training script to take only outputs for annot...</td>
      <td>Research</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>UMass</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>949607828976735</td>
      <td>2018-12-17</td>
      <td>2018-12-18</td>
      <td>2018-12-18</td>
      <td>make generate preds output .npy file with pred...</td>
      <td>Research</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>UMass</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>949607828976733</td>
      <td>2018-12-17</td>
      <td>2018-12-18</td>
      <td>2018-12-18</td>
      <td>option for generate preds script for not rotat...</td>
      <td>Research</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>UMass</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>949607828976731</td>
      <td>2018-12-17</td>
      <td>2018-12-18</td>
      <td>2018-12-18</td>
      <td>make generate preds scripts output rboxes inst...</td>
      <td>Research</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>UMass</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>949607828976727</td>
      <td>2018-12-17</td>
      <td>2018-12-18</td>
      <td>2018-12-18</td>
      <td>nms</td>
      <td>Research</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>UMass</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Next, we can load the previous data to get some sweet comparison visualizations.


```python
old_df = pd.read_csv('School.csv', parse_dates=[1, 2, 3, 7, 8])
old_df.tail()
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
      <th>804</th>
      <td>351764138393835</td>
      <td>2017-05-29</td>
      <td>2017-05-30</td>
      <td>2017-05-30</td>
      <td>brand new congress questions</td>
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
      <th>805</th>
      <td>351764138393836</td>
      <td>2017-05-29</td>
      <td>2017-05-29</td>
      <td>2017-05-29</td>
      <td>sign up for learnupon brand new congress</td>
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
      <th>806</th>
      <td>351764138393838</td>
      <td>2017-05-29</td>
      <td>2017-06-28</td>
      <td>2017-06-28</td>
      <td>create base for Hodler app</td>
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
      <th>807</th>
      <td>351779687262554</td>
      <td>2017-05-29</td>
      <td>2017-06-28</td>
      <td>2017-06-28</td>
      <td>battery life on lappy ubuntu</td>
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
      <th>808</th>
      <td>356635261007682</td>
      <td>2017-06-05</td>
      <td>2017-06-28</td>
      <td>2017-06-28</td>
      <td>bnc volunteer training</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_df = pd.concat([old_df, f18_df], verify_integrity=True, ignore_index=True, sort=True)
all_df.head()
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
      <th>Assignee</th>
      <th>Assignee Email</th>
      <th>Column</th>
      <th>Completed At</th>
      <th>Created At</th>
      <th>Due Date</th>
      <th>Last Modified</th>
      <th>Name</th>
      <th>Notes</th>
      <th>Parent Task</th>
      <th>Projects</th>
      <th>Start Date</th>
      <th>Tags</th>
      <th>Task ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-04-15</td>
      <td>2018-04-15</td>
      <td>NaT</td>
      <td>2018-04-15</td>
      <td>More debt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>148623786710031</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-04-07</td>
      <td>2018-04-06</td>
      <td>NaT</td>
      <td>2018-04-07</td>
      <td>Send one line email to erik to add you to the ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>148623786710030</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-03-27</td>
      <td>2018-03-27</td>
      <td>NaT</td>
      <td>2018-03-27</td>
      <td>withdraw from study abroad</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>610060357624798</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-03-26</td>
      <td>2018-03-09</td>
      <td>NaT</td>
      <td>2018-03-26</td>
      <td>hold</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>588106896688257</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-02-23</td>
      <td>2018-02-22</td>
      <td>2018-02-23</td>
      <td>2018-02-23</td>
      <td>find joydeep</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>School</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>570162249229318</td>
    </tr>
  </tbody>
</table>
</div>



To help retain our sanity, let's define colors for each semester.


```python
all_color = 'rgba(219, 64, 82, 0.7)'
old_color = 'rgba(63, 81, 191, 1.0)'
f18_color = 'rgba(33, 150, 255, 1.0)'
```

### Task Creation Day of Week Comparison

Let's see if tasks where still created with the same daily frequencies. Since there are much more tasks in the old data, we can normalize the value counts for a fair comparison. For the sake of keeping the code clean and the x-axis in order, I decided keep the days of the week as numbers. For reference, 0 is Monday.


```python
old_df['Created At DOW'] = old_df['Created At'].dt.dayofweek
f18_df['Created At DOW'] = f18_df['Created At'].dt.dayofweek
```


```python
trace1 = go.Bar(
    x=old_df['Created At DOW'].value_counts(normalize=True).keys(),
    y=old_df['Created At DOW'].value_counts(normalize=True).values,
    name='Old Data',
    marker={
        'color': old_color
    }
)
trace2 = go.Bar(
    x=f18_df['Created At DOW'].value_counts(normalize=True).keys(),
    y=f18_df['Created At DOW'].value_counts(normalize=True).values,
    name='Fall 18',
    marker={
        'color': f18_color
    }
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='DOW Comparison')
```


<div id="69cb7eeb-535d-4882-85ca-bee917f72e78" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("69cb7eeb-535d-4882-85ca-bee917f72e78", [{"marker": {"color": "rgba(63, 81, 191, 1.0)"}, "name": "Old Data", "x": [1, 0, 2, 6, 3, 4, 5], "y": [0.19283065512978986, 0.18294190358467244, 0.14709517923362175, 0.14091470951792337, 0.1273176761433869, 0.11372064276885044, 0.09517923362175525], "type": "bar", "uid": "632d1ee7-3a78-4624-ae5b-556e3c88b446"}, {"marker": {"color": "rgba(33, 150, 255, 1.0)"}, "name": "Fall 18", "x": [2, 4, 0, 5, 1, 3, 6], "y": [0.24166666666666667, 0.18333333333333332, 0.16666666666666666, 0.15, 0.13333333333333333, 0.075, 0.05], "type": "bar", "uid": "481bdfb2-4ad0-4939-a287-5f4aaa296135"}], {"barmode": "group"}, {"showLink": true, "linkText": "Export to plot.ly"});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("69cb7eeb-535d-4882-85ca-bee917f72e78"));});</script>


This was quite a surprise. The days when I created tasks seems to have changed somewhat this semester. 

Let's check if the overall trend has remained the same.


```python
all_df['Created At DOW'] = all_df['Created At'].dt.dayofweek

trace1 = go.Bar(
    x=all_df['Created At DOW'].value_counts(normalize=True).keys(),
    y=all_df['Created At DOW'].value_counts(normalize=True).values,
    name='All Data',
    marker={
        'color': all_color
    }
)

data = [trace1]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='DOW Comparison')
```


<div id="6a090126-8dc9-4d89-bf3a-414d8a2649f3" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("6a090126-8dc9-4d89-bf3a-414d8a2649f3", [{"marker": {"color": "rgba(219, 64, 82, 0.7)"}, "name": "All Data", "x": [1, 0, 2, 6, 4, 3, 5], "y": [0.18514531754574812, 0.18083961248654468, 0.15931108719052745, 0.12917115177610333, 0.12271259418729817, 0.12055974165769645, 0.10226049515608181], "type": "bar", "uid": "3d1ab8d5-94ec-4b79-acf0-d39f9feab79a"}], {"barmode": "group"}, {"showLink": true, "linkText": "Export to plot.ly"});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("6a090126-8dc9-4d89-bf3a-414d8a2649f3"));});</script>


There are definitely some small changes. Thursday caught up to Wednesday and Monday is catching up to Tuesday. However, the overall trend that I create the majority of my tasks at the beginning of the week remains strong.

### Completion Time

Next, let's look at the duration it took for me to complete each task. Because I used the `parse_dates` parameter when importing the CSVs, using the minus operator will return [timedelta objects](https://docs.python.org/2/library/datetime.html#timedelta-objects). Since Asana only provided dates without time, tasks with a duration of 0 days are ones that were created and completed on the same day.

Having already found outliers in the last analysis, let's only consider only tasks that took less than 30 days to complete. Again, we normalize for a better comparison.


```python
old_df['Duration'] = (old_df['Completed At'] - old_df['Created At'])
f18_df['Duration'] = (f18_df['Completed At'] - f18_df['Created At'])
```


```python
trace1 = go.Bar(
    x=old_df[(old_df['Duration'].astype('timedelta64[D]') < 30)]['Duration'].value_counts(normalize=True).keys().days,
    y=old_df[(old_df['Duration'].astype('timedelta64[D]') < 30)]['Duration'].value_counts(normalize=True).values,
    name='Old Data',
    marker={
        'color': old_color
    }
)
trace2 = go.Bar(
    x=f18_df[(f18_df['Duration'].astype('timedelta64[D]') < 30)]['Duration'].value_counts(normalize=True).keys().days,
    y=f18_df[(f18_df['Duration'].astype('timedelta64[D]') < 30)]['Duration'].value_counts(normalize=True).values,
    name='Fall 18',
    marker={
        'color': f18_color
    }
)
data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')
```


<div id="525466ec-901c-4a1e-a061-851ffdf75b9f" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("525466ec-901c-4a1e-a061-851ffdf75b9f", [{"marker": {"color": "rgba(63, 81, 191, 1.0)"}, "name": "Old Data", "x": [1, 0, 2, 4, 3, 5, 7, 6, 11, 8, 10, 12, 9, 15, 29, 13, 17, 14, 23, 16, 22, 19, 21, 27, 25, 26], "y": [0.22371967654986524, 0.1940700808625337, 0.13881401617250674, 0.09164420485175202, 0.09164420485175202, 0.0660377358490566, 0.0431266846361186, 0.03234501347708895, 0.018867924528301886, 0.01752021563342318, 0.012129380053908356, 0.01078167115902965, 0.01078167115902965, 0.009433962264150943, 0.008086253369272238, 0.006738544474393531, 0.004043126684636119, 0.004043126684636119, 0.0026954177897574125, 0.0026954177897574125, 0.0026954177897574125, 0.0026954177897574125, 0.0013477088948787063, 0.0013477088948787063, 0.0013477088948787063, 0.0013477088948787063], "type": "bar", "uid": "c179531e-5497-437a-9179-478a8508f976"}, {"marker": {"color": "rgba(33, 150, 255, 1.0)"}, "name": "Fall 18", "x": [3, 1, 0, 2, 7, 4, 5, 9, 6, 12, 13, 11, 8, 14, 17, 19, 20, 27, 15, 25], "y": [0.16666666666666666, 0.15833333333333333, 0.125, 0.1, 0.08333333333333333, 0.075, 0.058333333333333334, 0.03333333333333333, 0.03333333333333333, 0.03333333333333333, 0.016666666666666666, 0.016666666666666666, 0.016666666666666666, 0.016666666666666666, 0.016666666666666666, 0.016666666666666666, 0.008333333333333333, 0.008333333333333333, 0.008333333333333333, 0.008333333333333333], "type": "bar", "uid": "0370f679-0f2c-45b5-86bf-57513ee42599"}], {"barmode": "group"}, {"showLink": true, "linkText": "Export to plot.ly"});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("525466ec-901c-4a1e-a061-851ffdf75b9f"));});</script>


Now this is interesting! For the most part, the time it takes me to complete tasks seems to have remained relatively the same. However, this semester it seems I create more tasks that take around a week to complete. In addition, I made less tasks that were completed on the same day.

Next, like last time, let's see if we can figure out what type of tasks usually take longer to complete. I will once again use the fantastic [word_cloud library](http://amueller.github.io/word_cloud/index.html) by [amueller](http://amueller.github.io/).


```python
# concatenate all name fields from tasks separated by duration of 3 days
old_less_text = ' '.join(list(old_df[old_df['Duration'].astype('timedelta64[D]') < 3]['Name'].dropna()))
old_grtr_text = ' '.join(list(old_df[old_df['Duration'].astype('timedelta64[D]') >= 3]['Name'].dropna()))

f18_less_text = ' '.join(list(f18_df[f18_df['Duration'].astype('timedelta64[D]') < 3]['Name'].dropna()))
f18_grtr_text = ' '.join(list(f18_df[f18_df['Duration'].astype('timedelta64[D]') >= 3]['Name'].dropna()))

# prep text
old_less_text = custom_utils.prep_text_for_wordcloud(old_less_text)
old_grtr_text = custom_utils.prep_text_for_wordcloud(old_grtr_text)

f18_less_text = custom_utils.prep_text_for_wordcloud(f18_less_text)
f18_grtr_text = custom_utils.prep_text_for_wordcloud(f18_grtr_text)

# get word frequencies
old_less_counts = dict(collections.Counter(old_less_text.split()))
old_grtr_counts = dict(collections.Counter(old_grtr_text.split()))

f18_less_counts = dict(collections.Counter(f18_less_text.split()))
f18_grtr_counts = dict(collections.Counter(f18_grtr_text.split()))

# create wordclouds
old_less_wordcloud = WordCloud(background_color="white", max_words=1000, margin=10,random_state=1).generate_from_frequencies(old_less_counts)
old_grtr_wordcloud = WordCloud(background_color="white", max_words=1000, margin=10,random_state=1).generate_from_frequencies(old_grtr_counts)

f18_less_wordcloud = WordCloud(background_color="white", max_words=1000, margin=10,random_state=1).generate_from_frequencies(f18_less_counts)
f18_grtr_wordcloud = WordCloud(background_color="white", max_words=1000, margin=10,random_state=1).generate_from_frequencies(f18_grtr_counts)

# display wordclouds using matplotlib
f, axes = plt.subplots(2, 2, sharex=True)
f.set_size_inches(18, 10)
axes[0, 0].imshow(old_less_wordcloud, interpolation="bilinear")
axes[0, 0].set_title('Old <3 days', fontsize=36)
axes[0, 0].axis("off")
axes[0, 1].imshow(old_grtr_wordcloud, interpolation="bilinear")
axes[0, 1].set_title('Old >=3 days', fontsize=36)
axes[0, 1].axis("off")

axes[1, 0].imshow(f18_less_wordcloud, interpolation="bilinear")
axes[1, 0].set_title('F18 <3 days', fontsize=36)
axes[1, 0].axis("off")
axes[1, 1].imshow(f18_grtr_wordcloud, interpolation="bilinear")
axes[1, 1].set_title('F18 >=3 days', fontsize=36)
axes[1, 1].axis("off")
```




    (-0.5, 399.5, 199.5, -0.5)




![png](/img/blog/2019-01-07-asana-revisited/output_20_1.png)


A few things changed this semester. The research project I was on during the semester was pretty demanding so a lot of those tasks show up like `image`, `preds`, or `synthtext`. Also, since none of my classes had projects this semester, that doesn't show up.

However, some things remained the same. Homework usually takes more than 3 days and lecture notes are either done quickly or they get put off because other tasks are more important.

### Overdue Tasks

Next, let's take a look at overdue tasks.


```python
old_df['Overdue'] = old_df['Completed At'] - old_df['Due Date']
f18_df['Overdue'] = f18_df['Completed At'] - f18_df['Due Date']
```


```python
trace1 = go.Bar(
    x=old_df['Overdue'].value_counts(normalize=True).keys().days,
    y=old_df['Overdue'].value_counts(normalize=True).values,
    name='Old Data',
    marker={
        'color': old_color
    }
)
trace2 = go.Bar(
    x=f18_df['Overdue'].value_counts(normalize=True).keys().days,
    y=f18_df['Overdue'].value_counts(normalize=True).values,
    name='Fall 18',
    marker={
        'color': f18_color
    }
)
data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')
```


<div id="ea5e783c-fdc4-41a3-a716-69ade721e9b9" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("ea5e783c-fdc4-41a3-a716-69ade721e9b9", [{"marker": {"color": "rgba(63, 81, 191, 1.0)"}, "name": "Old Data", "x": [0, -1, -2, -3, 1, -5, -4, -6, -7, -9, -8, 2, -10, 6, 4, 5, 11, -12, -28, 21, -11, -15, 23, -14], "y": [0.3764367816091954, 0.14655172413793102, 0.08333333333333333, 0.07183908045977011, 0.06896551724137931, 0.05172413793103448, 0.04885057471264368, 0.03735632183908046, 0.020114942528735632, 0.014367816091954023, 0.014367816091954023, 0.011494252873563218, 0.008620689655172414, 0.008620689655172414, 0.008620689655172414, 0.005747126436781609, 0.0028735632183908046, 0.0028735632183908046, 0.0028735632183908046, 0.0028735632183908046, 0.0028735632183908046, 0.0028735632183908046, 0.0028735632183908046, 0.0028735632183908046], "type": "bar", "uid": "ba6ffd60-df7e-4ad7-bd1b-8a464cabece4"}, {"marker": {"color": "rgba(33, 150, 255, 1.0)"}, "name": "Fall 18", "x": [0, -1, 1, 5, -5, -2, -4, -6, -8, -3, -10, 7], "y": [0.5283018867924528, 0.18867924528301888, 0.07547169811320754, 0.03773584905660377, 0.03773584905660377, 0.018867924528301886, 0.018867924528301886, 0.018867924528301886, 0.018867924528301886, 0.018867924528301886, 0.018867924528301886, 0.018867924528301886], "type": "bar", "uid": "d0c86972-32b1-4aaa-b2cd-8d60ade2d9ae"}], {"barmode": "group"}, {"showLink": true, "linkText": "Export to plot.ly"});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("ea5e783c-fdc4-41a3-a716-69ade721e9b9"));});</script>


Seems like I did alright staying on top of things this semester.

---

Again, let's use wordclouds to check out what might be causing me to miss due dates.


```python
# concatenate all name fields from overdue tasks
old_before_text = ' '.join(list(old_df[old_df['Overdue'].astype('timedelta64[D]') < 0]['Name'].dropna()))
old_sameday_text = ' '.join(list(old_df[old_df['Overdue'].astype('timedelta64[D]') == 0]['Name'].dropna()))
old_overdue_text = ' '.join(list(old_df[old_df['Overdue'].astype('timedelta64[D]') > 0]['Name'].dropna()))

f18_before_text = ' '.join(list(f18_df[f18_df['Overdue'].astype('timedelta64[D]') < 0]['Name'].dropna()))
f18_sameday_text = ' '.join(list(f18_df[f18_df['Overdue'].astype('timedelta64[D]') == 0]['Name'].dropna()))
f18_overdue_text = ' '.join(list(f18_df[f18_df['Overdue'].astype('timedelta64[D]') > 0]['Name'].dropna()))

# prep text
old_before_text = custom_utils.prep_text_for_wordcloud(old_before_text)
old_sameday_text = custom_utils.prep_text_for_wordcloud(old_sameday_text)
old_overdue_text = custom_utils.prep_text_for_wordcloud(old_overdue_text)

f18_before_text = custom_utils.prep_text_for_wordcloud(f18_before_text)
f18_sameday_text = custom_utils.prep_text_for_wordcloud(f18_sameday_text)
f18_overdue_text = custom_utils.prep_text_for_wordcloud(f18_overdue_text)

# get word frequencies
old_before_counts = dict(collections.Counter(old_before_text.split()))
old_sameday_counts = dict(collections.Counter(old_sameday_text.split()))
old_overdue_counts = dict(collections.Counter(old_overdue_text.split()))

f18_before_counts = dict(collections.Counter(f18_before_text.split()))
f18_sameday_counts = dict(collections.Counter(f18_sameday_text.split()))
f18_overdue_counts = dict(collections.Counter(f18_overdue_text.split()))

# create wordclouds
old_before_wordcloud = WordCloud(background_color="white", max_words=1000, margin=10,random_state=1).generate_from_frequencies(old_before_counts)
old_sameday_wordcloud = WordCloud(background_color="white", max_words=1000, margin=10,random_state=1).generate_from_frequencies(old_sameday_counts)
old_overdue_wordcloud = WordCloud(background_color="white", max_words=1000, margin=10,random_state=1).generate_from_frequencies(old_overdue_counts)

f18_before_wordcloud = WordCloud(background_color="white", max_words=1000, margin=10,random_state=1).generate_from_frequencies(f18_before_counts)
f18_sameday_wordcloud = WordCloud(background_color="white", max_words=1000, margin=10,random_state=1).generate_from_frequencies(f18_sameday_counts)
f18_overdue_wordcloud = WordCloud(background_color="white", max_words=1000, margin=10,random_state=1).generate_from_frequencies(f18_overdue_counts)

# display wordclouds using matplotlib
f, axes = plt.subplots(4, 2, sharex=True)
f.set_size_inches(18, 20)
axes[0, 0].imshow(old_before_wordcloud, interpolation="bilinear")
axes[0, 0].set_title('Old Completed Before', fontsize=36)
axes[0, 0].axis("off")
axes[0, 1].imshow(old_sameday_wordcloud, interpolation="bilinear")
axes[0, 1].set_title('Old Completed Same Day', fontsize=36)
axes[0, 1].axis("off")
axes[1, 0].imshow(old_overdue_wordcloud, interpolation="bilinear")
axes[1, 0].set_title('Old Overdue', fontsize=36)
axes[1, 0].axis("off")
axes[1, 1].axis("off")

axes[2, 0].imshow(f18_before_wordcloud, interpolation="bilinear")
axes[2, 0].set_title('F18 Completed Before', fontsize=36)
axes[2, 0].axis("off")
axes[2, 1].imshow(f18_sameday_wordcloud, interpolation="bilinear")
axes[2, 1].set_title('F18 Completed Same Day', fontsize=36)
axes[2, 1].axis("off")
axes[3, 0].imshow(f18_overdue_wordcloud, interpolation="bilinear")
axes[3, 0].set_title('F18 Overdue', fontsize=36)
axes[3, 0].axis("off")
axes[3, 1].axis("off")
```




    (-0.5, 399.5, 0.0, 1.0)




![png](/img/blog/2019-01-07-asana-revisited/output_26_1.png)


## Busiest Class this Semester

Since I began to track the class for each task, we can check out which class was my busiest this semester.


```python
# https://community.plot.ly/t/setting-up-pie-charts-subplots-with-an-appropriate-size-and-spacing/5066
domain1={'x': [0, 1], 'y': [0, 1]}#cell (1,1)

fig = {
  "data": [
    {
      "values": f18_df['Column'].value_counts().values,
      "labels": f18_df['Column'].value_counts().keys(),
      'domain': domain1,
      "name": "Fall 18",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "annotations": [
            {
                "font": {
                    "size": 15
                },
                "showarrow": False,
                "text": "Fall 2018",
                "x": 0.5,
                "y": 0.5
            }
        ]
    }
}

iplot(fig, filename='donut')
```


<div id="fe532444-711f-4ee1-bd83-bd339b2ccb2c" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("fe532444-711f-4ee1-bd83-bd339b2ccb2c", [{"domain": {"x": [0, 1], "y": [0, 1]}, "hole": 0.4, "hoverinfo": "label+percent+name", "labels": ["Research", "CS 311", "Math 551", "Spanish 322", "CS 383"], "name": "Fall 18", "values": [39, 26, 22, 19, 14], "type": "pie", "uid": "f66a0ae9-9c03-42f9-adcc-b4978981fca6"}], {"annotations": [{"font": {"size": 15}, "showarrow": false, "text": "Fall 2018", "x": 0.5, "y": 0.5}]}, {"showLink": true, "linkText": "Export to plot.ly"});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("fe532444-711f-4ee1-bd83-bd339b2ccb2c"));});</script>


## Due Date Frequency this Semester

We can also see when my tasks were due.


```python
trace1 = go.Bar(
    x=f18_df['Due Date'].dropna().value_counts().keys(),
    y=f18_df['Due Date'].dropna().value_counts().values,
    name='Fall 18',
    marker={
        'color': f18_color
    }
)

data = [trace1]
iplot(data, filename='due date freq')
```


<div id="5e233523-b948-4646-863a-6f925c07121f" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("5e233523-b948-4646-863a-6f925c07121f", [{"marker": {"color": "rgba(33, 150, 255, 1.0)"}, "name": "Fall 18", "x": ["2018-09-07", "2018-11-30", "2018-09-17", "2018-10-22", "2018-09-10", "2018-12-12", "2018-11-09", "2018-09-20", "2018-09-24", "2018-10-09", "2018-11-05", "2018-10-01", "2018-11-16", "2018-10-02", "2018-10-05", "2018-11-02", "2018-11-28", "2018-12-03", "2018-09-14", "2018-10-29", "2018-10-10", "2018-09-21", "2018-12-06", "2018-10-24", "2018-10-18", "2018-10-06", "2018-10-19", "2018-11-13", "2018-10-07", "2018-11-01", "2018-12-15"], "y": [5, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "type": "bar", "uid": "4582541b-1a16-476b-9a93-1eb6273140b6"}], {}, {"showLink": true, "linkText": "Export to plot.ly"});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("5e233523-b948-4646-863a-6f925c07121f"));});</script>


## Conclusion
Hopefully this post show the motivation and potential benefit of revisiting a previous analysis in an attempt to find any significant changes. In the context of personal development, doing so can help you track your progress and achieve your goals. I still definitely need to put more effort into taking notes on time!
