---
layout: post
title: "Tracking Stress at University Part 2 - A Visual Summary"
desc: "During my winter break, I contemplated a lot about my previous two semesters at university. I had been stressed constantly and I frequently overworked myself leading to more stress..."
tag: "Data Visualization"
author: "Sean Kelley"
thumb: "/img/blog/2019-06-12-stress-tracking-pt2/sean_in_paris.jpg"
date: 2019-06-12
---

During my winter break, I contemplated a lot about my previous two semesters at university. I had been stressed constantly and I frequently overworked myself leading to more stress. Before I started Spring 2019, I devised a project that I could use to track and hopefully mitigate my stress while also flexing my data science muscles a bit. In [this blog post](http://seangtkelley.me/blog/2019/01/22/stress-tracking-pt1) I made on the first day of the semester, I detailed the variables that I would be tracking and the analyses I thought might be interesting.

However, I believe I might have inadvertantly become a victim of Goodhart's law, best phrased by Marilyn Strathern: "When a measure becomes a target, it ceases to be a good measure." Throughout the semester, my stress levels were significantly lower, my classes required less work, and overall, I felt as though I was in a much better place. Now, make no mistake, this is great news for me personally. However, the visualizations aren't as cool as had hoped.

Either way, here I present the visualizations that I tracked during my semester in order to keep my mental in check.


```python
import datetime as dt
import pytz

import numpy as np
import pandas as pd
import json
from ics import Calendar
from lib.python_fitbit import fitbit
from lib.python_fitbit import gather_keys_oauth2 as Oauth2

import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly import tools
import cufflinks as cf
from wordcloud import WordCloud

from lib import custom_utils

init_notebook_mode(connected=True)
cf.set_config_file(world_readable=True, offline=True)
```

<script src="//cdn.plot.ly/plotly-latest.min.js"></script>

### Constants


```python
START_DATE = dt.datetime.strptime('2019-01-22', '%Y-%m-%d')
END_DATE = dt.datetime.strptime('2019-05-10', '%Y-%m-%d')

local = pytz.timezone ("America/New_York")

START_DATE_UTC = local.localize(START_DATE, is_dst=None).astimezone(pytz.utc)
END_DATE_UTC = local.localize(END_DATE, is_dst=None).astimezone(pytz.utc)
```

### Load Daily Log Data


```python
daily_log_df = pd.read_csv('data/Daily Log (Responses) - Form Responses 1.csv', parse_dates=[0, 6])

# get relevant dates
daily_log_df = daily_log_df[(daily_log_df['Timestamp'] > START_DATE) & (daily_log_df['Timestamp'] < END_DATE)]

daily_log_df.head()
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
      <th>Timestamp</th>
      <th>Stress</th>
      <th>Happiness</th>
      <th>Energy</th>
      <th>Motivation</th>
      <th>Notes</th>
      <th>Date</th>
      <th>Food Coma</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>2019-01-22 20:07:08</td>
      <td>2</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>First day of classes. I think it's gonna be go...</td>
      <td>2019-01-22</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2019-01-23 21:51:14</td>
      <td>3</td>
      <td>7</td>
      <td>8</td>
      <td>8</td>
      <td>Had all classes now. I'm going to have to work...</td>
      <td>2019-01-23</td>
      <td>2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2019-01-24 20:13:42</td>
      <td>3</td>
      <td>7</td>
      <td>8</td>
      <td>7</td>
      <td>Everything is still going well for keeping up ...</td>
      <td>2019-01-24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2019-01-25 20:49:29</td>
      <td>2</td>
      <td>8</td>
      <td>7</td>
      <td>8</td>
      <td>Chilling at the SASA party right now. It's pre...</td>
      <td>2019-01-25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2019-01-26 20:07:14</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>6</td>
      <td>Chilling all day. Went to the gym to start the...</td>
      <td>2019-01-26</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Simple Stress Timeseries


```python
stress_trace = go.Scatter(
    x=daily_log_df.Date,
    y=daily_log_df.Stress,
    name='Stress Level',
    fill='tozeroy',
)
data = [stress_trace]
layout = go.Layout(
    title='Stress Level',
    yaxis={ 
        'title': 'Stress Level',
        'range': [1, 8] 
        
        
    }
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='stress')
```


<div id="b7d86760-d0c9-47c5-bcb5-5780bf54ca44" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("b7d86760-d0c9-47c5-bcb5-5780bf54ca44", [{"fill": "tozeroy", "name": "Stress Level", "x": ["2019-01-22", "2019-01-23", "2019-01-24", "2019-01-25", "2019-01-26", "2019-01-27", "2019-01-28", "2019-01-29", "2019-01-30", "2019-02-01", "2019-02-03", "2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08", "2019-02-09", "2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15", "2019-02-16", "2019-02-17", "2019-02-17", "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22", "2019-02-23", "2019-02-24", "2019-02-25", "2019-02-26", "2019-02-27", "2019-02-28", "2019-03-01", "2019-03-02", "2019-03-03", "2019-03-04", "2019-03-05", "2019-03-06", "2019-03-09", "2019-03-10", "2019-03-11", "2019-03-12", "2019-03-13", "2019-03-14", "2019-03-15", "2019-03-16", "2019-03-17", "2019-03-18", "2019-03-19", "2019-03-20", "2019-03-21", "2019-03-22", "2019-03-23", "2019-03-24", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28", "2019-03-29", "2019-03-30", "2019-03-31", "2019-04-01", "2019-04-02", "2019-04-03", "2019-04-04", "2019-04-05", "2019-04-06", "2019-04-07", "2019-04-08", "2019-04-09", "2019-04-10", "2019-04-11", "2019-04-12", "2019-04-13", "2019-04-14", "2019-04-15", "2019-04-16", "2019-04-18", "2019-04-20", "2019-04-21", "2019-04-22", "2019-04-23", "2019-04-24", "2019-04-25", "2019-04-26", "2019-04-27", "2019-04-29", "2019-04-30", "2019-05-01", "2019-05-02", "2019-05-03", "2019-05-04", "2019-05-05", "2019-05-06", "2019-05-07", "2019-05-09"], "y": [2, 3, 3, 2, 2, 3, 2, 2, 2, 2, 2, 3, 4, 2, 2, 2, 2, 4, 3, 3, 2, 3, 3, 3, 2, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 4, 3, 3, 4, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 2, 3, 3, 2, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3], "type": "scatter", "uid": "1c2cdb12-a829-4c0b-8a56-b233d550e0bb"}], {"title": "Stress Level", "yaxis": {"range": [1, 8], "title": "Stress Level"}}, {"showLink": true, "linkText": "Export to plot.ly"});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("b7d86760-d0c9-47c5-bcb5-5780bf54ca44"));});</script>


# Work

How much does what I need to complete correlate with my stress?

### Load Asana Data


```python
asana_df = pd.read_csv('data/asana-umass-s19.csv', parse_dates=[1, 2, 3, 8, 9])
asana_df.head()
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
      <td>1120800788569533</td>
      <td>2019-05-01</td>
      <td>2019-05-04</td>
      <td>2019-05-04</td>
      <td>final report</td>
      <td>Spanish 306</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>2019-05-04</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>UMass</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1120292396473715</td>
      <td>2019-04-28</td>
      <td>2019-04-30</td>
      <td>2019-04-30</td>
      <td>final submission</td>
      <td>CS 326</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>2019-04-30</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>UMass</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1120002043175716</td>
      <td>2019-04-25</td>
      <td>2019-04-25</td>
      <td>2019-04-25</td>
      <td>go to wikipedia page button</td>
      <td>CS 326</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>2019-04-25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>UMass</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>884979504873319</td>
      <td>2019-04-24</td>
      <td>2019-05-02</td>
      <td>2019-05-02</td>
      <td>Final cheat sheet</td>
      <td>Stats 516</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>2019-05-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>UMass</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1119773356459372</td>
      <td>2019-04-23</td>
      <td>2019-04-30</td>
      <td>2019-04-30</td>
      <td>hw 5</td>
      <td>CS 589</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>2019-05-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>UMass</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Timeseries Due Dates vs Stress


```python
due_date_val_counts = asana_df['Due Date'].value_counts()

due_date_freqs = []
for i in range((END_DATE - START_DATE).days + 1):
    date_str = (START_DATE + dt.timedelta(days=i)).strftime('%Y-%m-%d')
    date_val = due_date_val_counts.get(date_str, 0)
    num = date_val.values[0] if len(date_val)>0 else 0
    due_date_freqs.append({
        'date': date_str,
        'num': num
    })
due_date_freqs_df = pd.DataFrame(due_date_freqs)
```


```python
due_date_trace = go.Bar(
    x=due_date_freqs_df.date,
    y=due_date_freqs_df.num,
    name='Due Dates',
    yaxis='y2'
)

data = [stress_trace, due_date_trace]
layout = go.Layout(
    title='Stress Level vs Number of Due Dates',
    yaxis1=dict(
        title='Stress Level',
        overlaying='y2',
        range=[0, 8]
    ),
    yaxis2=dict(
        title='Due Dates',
        side='right',
        range=[0, 8]
    ),
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='stress-vs-due-dates')
```


<div id="d6dc48d1-be77-4873-a8f7-253902846b1c" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("d6dc48d1-be77-4873-a8f7-253902846b1c", [{"fill": "tozeroy", "name": "Stress Level", "x": ["2019-01-22", "2019-01-23", "2019-01-24", "2019-01-25", "2019-01-26", "2019-01-27", "2019-01-28", "2019-01-29", "2019-01-30", "2019-02-01", "2019-02-03", "2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08", "2019-02-09", "2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15", "2019-02-16", "2019-02-17", "2019-02-17", "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22", "2019-02-23", "2019-02-24", "2019-02-25", "2019-02-26", "2019-02-27", "2019-02-28", "2019-03-01", "2019-03-02", "2019-03-03", "2019-03-04", "2019-03-05", "2019-03-06", "2019-03-09", "2019-03-10", "2019-03-11", "2019-03-12", "2019-03-13", "2019-03-14", "2019-03-15", "2019-03-16", "2019-03-17", "2019-03-18", "2019-03-19", "2019-03-20", "2019-03-21", "2019-03-22", "2019-03-23", "2019-03-24", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28", "2019-03-29", "2019-03-30", "2019-03-31", "2019-04-01", "2019-04-02", "2019-04-03", "2019-04-04", "2019-04-05", "2019-04-06", "2019-04-07", "2019-04-08", "2019-04-09", "2019-04-10", "2019-04-11", "2019-04-12", "2019-04-13", "2019-04-14", "2019-04-15", "2019-04-16", "2019-04-18", "2019-04-20", "2019-04-21", "2019-04-22", "2019-04-23", "2019-04-24", "2019-04-25", "2019-04-26", "2019-04-27", "2019-04-29", "2019-04-30", "2019-05-01", "2019-05-02", "2019-05-03", "2019-05-04", "2019-05-05", "2019-05-06", "2019-05-07", "2019-05-09"], "y": [2, 3, 3, 2, 2, 3, 2, 2, 2, 2, 2, 3, 4, 2, 2, 2, 2, 4, 3, 3, 2, 3, 3, 3, 2, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 4, 3, 3, 4, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 2, 3, 3, 2, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3], "type": "scatter", "uid": "f6210907-4a01-4285-b01f-6e3a8b5ee194"}, {"name": "Due Dates", "x": ["2019-01-22", "2019-01-23", "2019-01-24", "2019-01-25", "2019-01-26", "2019-01-27", "2019-01-28", "2019-01-29", "2019-01-30", "2019-01-31", "2019-02-01", "2019-02-02", "2019-02-03", "2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08", "2019-02-09", "2019-02-10", "2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15", "2019-02-16", "2019-02-17", "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22", "2019-02-23", "2019-02-24", "2019-02-25", "2019-02-26", "2019-02-27", "2019-02-28", "2019-03-01", "2019-03-02", "2019-03-03", "2019-03-04", "2019-03-05", "2019-03-06", "2019-03-07", "2019-03-08", "2019-03-09", "2019-03-10", "2019-03-11", "2019-03-12", "2019-03-13", "2019-03-14", "2019-03-15", "2019-03-16", "2019-03-17", "2019-03-18", "2019-03-19", "2019-03-20", "2019-03-21", "2019-03-22", "2019-03-23", "2019-03-24", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28", "2019-03-29", "2019-03-30", "2019-03-31", "2019-04-01", "2019-04-02", "2019-04-03", "2019-04-04", "2019-04-05", "2019-04-06", "2019-04-07", "2019-04-08", "2019-04-09", "2019-04-10", "2019-04-11", "2019-04-12", "2019-04-13", "2019-04-14", "2019-04-15", "2019-04-16", "2019-04-17", "2019-04-18", "2019-04-19", "2019-04-20", "2019-04-21", "2019-04-22", "2019-04-23", "2019-04-24", "2019-04-25", "2019-04-26", "2019-04-27", "2019-04-28", "2019-04-29", "2019-04-30", "2019-05-01", "2019-05-02", "2019-05-03", "2019-05-04", "2019-05-05", "2019-05-06", "2019-05-07", "2019-05-08", "2019-05-09", "2019-05-10"], "y": [0, 2, 2, 1, 3, 2, 1, 1, 1, 1, 3, 3, 0, 0, 1, 1, 2, 1, 1, 0, 0, 1, 0, 2, 3, 0, 3, 1, 1, 2, 2, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 3, 1, 2, 0, 0, 4, 0, 1, 1, 3, 2, 4, 1, 0, 1, 1, 0, 0, 1, 5, 4, 1, 0, 1, 2, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 3, 0, 0, 0, 1, 3, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], "yaxis": "y2", "type": "bar", "uid": "184c65b0-3112-4a0d-8c9b-89537a5c89e3"}], {"title": "Stress Level vs Number of Due Dates", "yaxis": {"overlaying": "y2", "range": [0, 8], "title": "Stress Level"}, "yaxis2": {"range": [0, 8], "side": "right", "title": "Due Dates"}}, {"showLink": true, "linkText": "Export to plot.ly"});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("d6dc48d1-be77-4873-a8f7-253902846b1c"));});</script>


## Wordcloud of Due Dates on Stressful vs Non-stressful Days


```python
stressful_dates = daily_log_df[daily_log_df.Stress > 3].Date
nonstressful_dates = daily_log_df[daily_log_df.Stress <= 3].Date
```


```python
stress_tasks = asana_df[asana_df['Due Date'].isin(stressful_dates)]
nonstress_tasks = asana_df[asana_df['Due Date'].isin(nonstressful_dates)]
```


```python
# concatenate all name fields from tasks separated by duration of 3 days
stress_text = ' '.join(list(stress_tasks['Name'].dropna()))
nonstress_text = ' '.join(list(nonstress_tasks['Name'].dropna()))

# prep text
stress_wordcloud = custom_utils.generate_wordcloud(stress_text)
nonstress_wordcloud = custom_utils.generate_wordcloud(nonstress_text)

# display wordclouds using matplotlib
f, axes = plt.subplots(1, 2, sharex=True)
f.set_size_inches(18, 10)
axes[0].imshow(stress_wordcloud, interpolation="bilinear")
axes[0].set_title('Tasks on Stressful Days', fontsize=36)
axes[0].axis("off")
axes[1].imshow(nonstress_wordcloud, interpolation="bilinear")
axes[1].set_title('Tasks on Non-stressful Days', fontsize=36)
axes[1].axis("off")
```




    (-0.5, 399.5, 199.5, -0.5)




![png](/img/blog/2019-06-12-stress-tracking-pt2/output_17_1.png)


## Number of Incomplete Tasks vs Stress


```python
incomplete_task_counts = []
for i in range((END_DATE - START_DATE).days + 1):
    date = START_DATE + dt.timedelta(days=i)
    incomplete_task_counts.append({
        'date': date,
        'num': len(asana_df[(asana_df['Created At'] <= date) & ((asana_df['Completed At'] >= date) | (asana_df['Completed At'].isnull()))].index)
    })
incomplete_task_counts = pd.DataFrame(incomplete_task_counts)
```


```python
incomplete_tasks_trace = go.Bar(
    x=incomplete_task_counts.date,
    y=incomplete_task_counts.num,
    name='Incomplete Tasks',
    yaxis='y2'
)

data = [stress_trace, incomplete_tasks_trace]
layout = go.Layout(
    title='Stress Level vs Number of Incomplete Tasks',
    yaxis1=dict(
        title='Stress Level',
        overlaying='y2',
        range=[0, 8]
    ),
    yaxis2=dict(
        title='Incomplete Tasks',
        side='right',
        range=[0, 16]
    ),
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='rhr-vs-due-dates')
```


<div id="3a748285-bcb2-458d-9a19-47c1a59f42b0" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("3a748285-bcb2-458d-9a19-47c1a59f42b0", [{"fill": "tozeroy", "name": "Stress Level", "x": ["2019-01-22", "2019-01-23", "2019-01-24", "2019-01-25", "2019-01-26", "2019-01-27", "2019-01-28", "2019-01-29", "2019-01-30", "2019-02-01", "2019-02-03", "2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08", "2019-02-09", "2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15", "2019-02-16", "2019-02-17", "2019-02-17", "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22", "2019-02-23", "2019-02-24", "2019-02-25", "2019-02-26", "2019-02-27", "2019-02-28", "2019-03-01", "2019-03-02", "2019-03-03", "2019-03-04", "2019-03-05", "2019-03-06", "2019-03-09", "2019-03-10", "2019-03-11", "2019-03-12", "2019-03-13", "2019-03-14", "2019-03-15", "2019-03-16", "2019-03-17", "2019-03-18", "2019-03-19", "2019-03-20", "2019-03-21", "2019-03-22", "2019-03-23", "2019-03-24", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28", "2019-03-29", "2019-03-30", "2019-03-31", "2019-04-01", "2019-04-02", "2019-04-03", "2019-04-04", "2019-04-05", "2019-04-06", "2019-04-07", "2019-04-08", "2019-04-09", "2019-04-10", "2019-04-11", "2019-04-12", "2019-04-13", "2019-04-14", "2019-04-15", "2019-04-16", "2019-04-18", "2019-04-20", "2019-04-21", "2019-04-22", "2019-04-23", "2019-04-24", "2019-04-25", "2019-04-26", "2019-04-27", "2019-04-29", "2019-04-30", "2019-05-01", "2019-05-02", "2019-05-03", "2019-05-04", "2019-05-05", "2019-05-06", "2019-05-07", "2019-05-09"], "y": [2, 3, 3, 2, 2, 3, 2, 2, 2, 2, 2, 3, 4, 2, 2, 2, 2, 4, 3, 3, 2, 3, 3, 3, 2, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 4, 3, 3, 4, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 2, 3, 3, 2, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3], "type": "scatter", "uid": "a859c255-a92f-47cd-a54c-830ead13355d"}, {"name": "Incomplete Tasks", "x": ["2019-01-22", "2019-01-23", "2019-01-24", "2019-01-25", "2019-01-26", "2019-01-27", "2019-01-28", "2019-01-29", "2019-01-30", "2019-01-31", "2019-02-01", "2019-02-02", "2019-02-03", "2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08", "2019-02-09", "2019-02-10", "2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15", "2019-02-16", "2019-02-17", "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22", "2019-02-23", "2019-02-24", "2019-02-25", "2019-02-26", "2019-02-27", "2019-02-28", "2019-03-01", "2019-03-02", "2019-03-03", "2019-03-04", "2019-03-05", "2019-03-06", "2019-03-07", "2019-03-08", "2019-03-09", "2019-03-10", "2019-03-11", "2019-03-12", "2019-03-13", "2019-03-14", "2019-03-15", "2019-03-16", "2019-03-17", "2019-03-18", "2019-03-19", "2019-03-20", "2019-03-21", "2019-03-22", "2019-03-23", "2019-03-24", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28", "2019-03-29", "2019-03-30", "2019-03-31", "2019-04-01", "2019-04-02", "2019-04-03", "2019-04-04", "2019-04-05", "2019-04-06", "2019-04-07", "2019-04-08", "2019-04-09", "2019-04-10", "2019-04-11", "2019-04-12", "2019-04-13", "2019-04-14", "2019-04-15", "2019-04-16", "2019-04-17", "2019-04-18", "2019-04-19", "2019-04-20", "2019-04-21", "2019-04-22", "2019-04-23", "2019-04-24", "2019-04-25", "2019-04-26", "2019-04-27", "2019-04-28", "2019-04-29", "2019-04-30", "2019-05-01", "2019-05-02", "2019-05-03", "2019-05-04", "2019-05-05", "2019-05-06", "2019-05-07", "2019-05-08", "2019-05-09", "2019-05-10"], "y": [4, 10, 6, 6, 9, 11, 9, 5, 5, 5, 3, 2, 4, 4, 6, 6, 9, 7, 6, 7, 10, 8, 10, 13, 12, 13, 9, 10, 7, 6, 5, 5, 4, 4, 7, 9, 6, 6, 5, 5, 4, 4, 4, 4, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 9, 9, 9, 8, 8, 10, 9, 8, 6, 11, 15, 12, 11, 15, 10, 8, 7, 8, 6, 7, 7, 14, 11, 8, 7, 7, 5, 9, 8, 8, 8, 9, 9, 8, 9, 10, 10, 8, 6, 6, 7, 7, 5, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1], "yaxis": "y2", "type": "bar", "uid": "9c03f43d-537e-44d2-b4b4-a9ead2167b99"}], {"title": "Stress Level vs Number of Incomplete Tasks", "yaxis": {"overlaying": "y2", "range": [0, 8], "title": "Stress Level"}, "yaxis2": {"range": [0, 16], "side": "right", "title": "Incomplete Tasks"}}, {"showLink": true, "linkText": "Export to plot.ly"});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("3a748285-bcb2-458d-9a19-47c1a59f42b0"));});</script>


## Exams vs Stress


```python
with open('data/Exams_2alvmakoou6sa9ks0roaq79nic@group.calendar.google.com.ics', 'r') as f:
    exams_cal = Calendar(f.readlines())
```


```python
exam_counts = []
for i in range((END_DATE_UTC - START_DATE_UTC).days + 1):
    date = START_DATE_UTC + dt.timedelta(days=i)
    num = 0
    for event in exams_cal.events:
        if (event.begin - date).days == 0:
            num += 1
    
    exam_counts.append({
        'date': date.strftime('%Y-%m-%d'),
        'num': num
    })
    
exam_counts = pd.DataFrame(exam_counts)
```


```python
exams_trace = go.Bar(
    x=exam_counts.date,
    y=exam_counts.num,
    name='Exams',
    yaxis='y2'
)

data = [stress_trace, exams_trace]
layout = go.Layout(
    title='Stress Level vs Exams',
    yaxis1=dict(
        title='Stress Level',
        range=[0, 8]
    ),
    yaxis2=dict(
        title='Exams',
        overlaying='y',
        side='right',
        range=[0, 2]
    ),
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='stress-vs-exams')
```


<div id="ea91a9e6-97b1-4ff5-9a40-c1911f2a4d15" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("ea91a9e6-97b1-4ff5-9a40-c1911f2a4d15", [{"fill": "tozeroy", "name": "Stress Level", "x": ["2019-01-22", "2019-01-23", "2019-01-24", "2019-01-25", "2019-01-26", "2019-01-27", "2019-01-28", "2019-01-29", "2019-01-30", "2019-02-01", "2019-02-03", "2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08", "2019-02-09", "2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15", "2019-02-16", "2019-02-17", "2019-02-17", "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22", "2019-02-23", "2019-02-24", "2019-02-25", "2019-02-26", "2019-02-27", "2019-02-28", "2019-03-01", "2019-03-02", "2019-03-03", "2019-03-04", "2019-03-05", "2019-03-06", "2019-03-09", "2019-03-10", "2019-03-11", "2019-03-12", "2019-03-13", "2019-03-14", "2019-03-15", "2019-03-16", "2019-03-17", "2019-03-18", "2019-03-19", "2019-03-20", "2019-03-21", "2019-03-22", "2019-03-23", "2019-03-24", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28", "2019-03-29", "2019-03-30", "2019-03-31", "2019-04-01", "2019-04-02", "2019-04-03", "2019-04-04", "2019-04-05", "2019-04-06", "2019-04-07", "2019-04-08", "2019-04-09", "2019-04-10", "2019-04-11", "2019-04-12", "2019-04-13", "2019-04-14", "2019-04-15", "2019-04-16", "2019-04-18", "2019-04-20", "2019-04-21", "2019-04-22", "2019-04-23", "2019-04-24", "2019-04-25", "2019-04-26", "2019-04-27", "2019-04-29", "2019-04-30", "2019-05-01", "2019-05-02", "2019-05-03", "2019-05-04", "2019-05-05", "2019-05-06", "2019-05-07", "2019-05-09"], "y": [2, 3, 3, 2, 2, 3, 2, 2, 2, 2, 2, 3, 4, 2, 2, 2, 2, 4, 3, 3, 2, 3, 3, 3, 2, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 4, 3, 3, 4, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 2, 3, 3, 2, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3], "type": "scatter", "uid": "b54d0a25-2dfd-4fb5-a603-a6a3db73ddb1"}, {"name": "Exams", "x": ["2019-01-22", "2019-01-23", "2019-01-24", "2019-01-25", "2019-01-26", "2019-01-27", "2019-01-28", "2019-01-29", "2019-01-30", "2019-01-31", "2019-02-01", "2019-02-02", "2019-02-03", "2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08", "2019-02-09", "2019-02-10", "2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15", "2019-02-16", "2019-02-17", "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22", "2019-02-23", "2019-02-24", "2019-02-25", "2019-02-26", "2019-02-27", "2019-02-28", "2019-03-01", "2019-03-02", "2019-03-03", "2019-03-04", "2019-03-05", "2019-03-06", "2019-03-07", "2019-03-08", "2019-03-09", "2019-03-10", "2019-03-11", "2019-03-12", "2019-03-13", "2019-03-14", "2019-03-15", "2019-03-16", "2019-03-17", "2019-03-18", "2019-03-19", "2019-03-20", "2019-03-21", "2019-03-22", "2019-03-23", "2019-03-24", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28", "2019-03-29", "2019-03-30", "2019-03-31", "2019-04-01", "2019-04-02", "2019-04-03", "2019-04-04", "2019-04-05", "2019-04-06", "2019-04-07", "2019-04-08", "2019-04-09", "2019-04-10", "2019-04-11", "2019-04-12", "2019-04-13", "2019-04-14", "2019-04-15", "2019-04-16", "2019-04-17", "2019-04-18", "2019-04-19", "2019-04-20", "2019-04-21", "2019-04-22", "2019-04-23", "2019-04-24", "2019-04-25", "2019-04-26", "2019-04-27", "2019-04-28", "2019-04-29", "2019-04-30", "2019-05-01", "2019-05-02", "2019-05-03", "2019-05-04", "2019-05-05", "2019-05-06", "2019-05-07", "2019-05-08", "2019-05-09"], "y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1], "yaxis": "y2", "type": "bar", "uid": "664d00d2-6966-4463-bb75-ad8b46ddd3f8"}], {"title": "Stress Level vs Exams", "yaxis": {"range": [0, 8], "title": "Stress Level"}, "yaxis2": {"overlaying": "y", "range": [0, 2], "side": "right", "title": "Exams"}}, {"showLink": true, "linkText": "Export to plot.ly"});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("ea91a9e6-97b1-4ff5-9a40-c1911f2a4d15"));});</script>


# Body

How does my body respond to stress? 

### Setup Fitbit API Client


```python
with open('keys.json', 'r') as f:
    keys = json.loads(f.read())
    
server = Oauth2.OAuth2Server(keys['fitbit_client_id'], keys['fitbit_client_secret'])
server.browser_authorize()
ACCESS_TOKEN = str(server.fitbit.client.session.token['access_token'])
REFRESH_TOKEN = str(server.fitbit.client.session.token['refresh_token'])
fitbit_client = fitbit.Fitbit(keys['fitbit_client_id'], keys['fitbit_client_secret'], oauth2=True, access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN)
```

    [12/Jun/2019:19:22:39] ENGINE Listening for SIGTERM.
    [12/Jun/2019:19:22:39] ENGINE Listening for SIGHUP.
    [12/Jun/2019:19:22:39] ENGINE Listening for SIGUSR1.
    [12/Jun/2019:19:22:39] ENGINE Bus STARTING
    CherryPy Checker:
    The Application mounted at '' has an empty config.
    
    [12/Jun/2019:19:22:39] ENGINE Started monitor thread 'Autoreloader'.
    [12/Jun/2019:19:22:39] ENGINE Serving on http://127.0.0.1:8080
    [12/Jun/2019:19:22:39] ENGINE Bus STARTED


    127.0.0.1 - - [12/Jun/2019:19:22:41] "GET /?code=6ca26bec02cd3e194d7a3d292acd43579d406bdf&state=yWAKtBkwQV1kPMpOPi1dePViirgWh3 HTTP/1.1" 200 122 "" "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/74.0.3729.169 Chrome/74.0.3729.169 Safari/537.36"


    [12/Jun/2019:19:22:42] ENGINE Bus STOPPING
    [12/Jun/2019:19:22:47] ENGINE HTTP Server cherrypy._cpwsgi_server.CPWSGIServer(('127.0.0.1', 8080)) shut down
    [12/Jun/2019:19:22:47] ENGINE Stopped thread 'Autoreloader'.
    [12/Jun/2019:19:22:47] ENGINE Bus STOPPED
    [12/Jun/2019:19:22:47] ENGINE Bus EXITING
    [12/Jun/2019:19:22:47] ENGINE Bus EXITED
    [12/Jun/2019:19:22:47] ENGINE Waiting for child threads to terminate...


## Resting Heart Rate vs Stress


```python
heart_ts = fitbit_client.time_series('activities/heart', 
                                    base_date=START_DATE.strftime('%Y-%m-%d'), 
                                    end_date=END_DATE.strftime('%Y-%m-%d'))

rhr_data = []
for row in heart_ts['activities-heart']:
    try:
        restingHeartRate = row['value']['restingHeartRate']
    except:
        restingHeartRate = restingHeartRate
        
    rhr_data.append({
        'date': row['dateTime'],
        'rhr': restingHeartRate
    })
rhr_df = pd.DataFrame(rhr_data)
```


```python
rhr_trace = go.Scatter(
    x=rhr_df.date,
    y=rhr_df.rhr,
    name='Resting Heart Rate',
    yaxis='y2',
    fill='tozeroy',
)
data = [stress_trace, rhr_trace]
layout = go.Layout(
    title='Stress Level vs Resting Heart Rate',
    yaxis=dict(
        title='Stress Level',
        range=[0, 8]
    ),
    yaxis2=dict(
        title='Resting Heart Rate',
        overlaying='y',
        side='right',
        range=[45, 70]
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='stress-vs-rhr')
```


<div id="05e52acd-f2e0-4f38-9807-979015e15611" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("05e52acd-f2e0-4f38-9807-979015e15611", [{"fill": "tozeroy", "name": "Stress Level", "x": ["2019-01-22", "2019-01-23", "2019-01-24", "2019-01-25", "2019-01-26", "2019-01-27", "2019-01-28", "2019-01-29", "2019-01-30", "2019-02-01", "2019-02-03", "2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08", "2019-02-09", "2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15", "2019-02-16", "2019-02-17", "2019-02-17", "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22", "2019-02-23", "2019-02-24", "2019-02-25", "2019-02-26", "2019-02-27", "2019-02-28", "2019-03-01", "2019-03-02", "2019-03-03", "2019-03-04", "2019-03-05", "2019-03-06", "2019-03-09", "2019-03-10", "2019-03-11", "2019-03-12", "2019-03-13", "2019-03-14", "2019-03-15", "2019-03-16", "2019-03-17", "2019-03-18", "2019-03-19", "2019-03-20", "2019-03-21", "2019-03-22", "2019-03-23", "2019-03-24", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28", "2019-03-29", "2019-03-30", "2019-03-31", "2019-04-01", "2019-04-02", "2019-04-03", "2019-04-04", "2019-04-05", "2019-04-06", "2019-04-07", "2019-04-08", "2019-04-09", "2019-04-10", "2019-04-11", "2019-04-12", "2019-04-13", "2019-04-14", "2019-04-15", "2019-04-16", "2019-04-18", "2019-04-20", "2019-04-21", "2019-04-22", "2019-04-23", "2019-04-24", "2019-04-25", "2019-04-26", "2019-04-27", "2019-04-29", "2019-04-30", "2019-05-01", "2019-05-02", "2019-05-03", "2019-05-04", "2019-05-05", "2019-05-06", "2019-05-07", "2019-05-09"], "y": [2, 3, 3, 2, 2, 3, 2, 2, 2, 2, 2, 3, 4, 2, 2, 2, 2, 4, 3, 3, 2, 3, 3, 3, 2, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 4, 3, 3, 4, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 2, 3, 3, 2, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3], "type": "scatter", "uid": "dc4d94c6-8b4c-4e0e-984b-236ee8a9b7a3"}, {"fill": "tozeroy", "name": "Resting Heart Rate", "x": ["2019-01-22", "2019-01-23", "2019-01-24", "2019-01-25", "2019-01-26", "2019-01-27", "2019-01-28", "2019-01-29", "2019-01-30", "2019-01-31", "2019-02-01", "2019-02-02", "2019-02-03", "2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08", "2019-02-09", "2019-02-10", "2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15", "2019-02-16", "2019-02-17", "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22", "2019-02-23", "2019-02-24", "2019-02-25", "2019-02-26", "2019-02-27", "2019-02-28", "2019-03-01", "2019-03-02", "2019-03-03", "2019-03-04", "2019-03-05", "2019-03-06", "2019-03-07", "2019-03-08", "2019-03-09", "2019-03-10", "2019-03-11", "2019-03-12", "2019-03-13", "2019-03-14", "2019-03-15", "2019-03-16", "2019-03-17", "2019-03-18", "2019-03-19", "2019-03-20", "2019-03-21", "2019-03-22", "2019-03-23", "2019-03-24", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28", "2019-03-29", "2019-03-30", "2019-03-31", "2019-04-01", "2019-04-02", "2019-04-03", "2019-04-04", "2019-04-05", "2019-04-06", "2019-04-07", "2019-04-08", "2019-04-09", "2019-04-10", "2019-04-11", "2019-04-12", "2019-04-13", "2019-04-14", "2019-04-15", "2019-04-16", "2019-04-17", "2019-04-18", "2019-04-19", "2019-04-20", "2019-04-21", "2019-04-22", "2019-04-23", "2019-04-24", "2019-04-25", "2019-04-26", "2019-04-27", "2019-04-28", "2019-04-29", "2019-04-30", "2019-05-01", "2019-05-02", "2019-05-03", "2019-05-04", "2019-05-05", "2019-05-06", "2019-05-07", "2019-05-08", "2019-05-09", "2019-05-10"], "y": [62, 61, 61, 60, 62, 61, 59, 59, 59, 59, 59, 61, 63, 63, 62, 60, 60, 60, 60, 58, 56, 56, 55, 56, 56, 57, 57, 58, 55, 56, 57, 58, 60, 61, 59, 57, 57, 58, 59, 60, 59, 58, 58, 58, 59, 59, 60, 59, 59, 59, 58, 59, 59, 59, 59, 58, 57, 56, 57, 58, 60, 60, 58, 56, 56, 57, 57, 58, 58, 58, 58, 59, 58, 58, 59, 60, 58, 58, 58, 58, 58, 59, 57, 56, 54, 54, 55, 57, 58, 58, 57, 57, 57, 57, 56, 58, 59, 59, 58, 57, 57, 56, 57, 58, 58, 57, 57, 57, 57], "yaxis": "y2", "type": "scatter", "uid": "bad7462f-a7ff-466d-a421-2179570138be"}], {"title": "Stress Level vs Resting Heart Rate", "yaxis": {"range": [0, 8], "title": "Stress Level"}, "yaxis2": {"overlaying": "y", "range": [45, 70], "side": "right", "title": "Resting Heart Rate"}}, {"showLink": true, "linkText": "Export to plot.ly"});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("05e52acd-f2e0-4f38-9807-979015e15611"));});</script>


## Sleep vs Stress

The missing data in mid-May is because my Fitbit ran out of battery over spring break and I forgot to bring my charger.


```python
sleep_logs = []
for i in range((END_DATE - START_DATE).days + 1):
    try:
        date_str = (START_DATE + dt.timedelta(days=i))
        sleep_log = fitbit_client.get_sleep(date=date_str)
        sleep_logs.append({
            'date': date_str,
            'deep': sleep_log['summary']['stages']['deep'] / 60,
            'light': sleep_log['summary']['stages']['light']  / 60,
            'rem': sleep_log['summary']['stages']['rem']  / 60,
            'wake': sleep_log['summary']['stages']['wake']  / 60,
            'total': sleep_log['summary']['totalMinutesAsleep'] / 60
        })
    except Exception as e:
        print(e)
sleep_df = pd.DataFrame(sleep_logs)
```

    'stages'
    'stages'
    'stages'
    'stages'
    'stages'
    'stages'
    'stages'



```python
sleep_trace = go.Bar(
    x=sleep_df.date,
    y=sleep_df.total,
    name='Sleep',
    yaxis='y2'
)

data = [stress_trace, sleep_trace]
layout = go.Layout(
    title='Stress Level vs Sleep',
    barmode='stack',
    yaxis=dict(
        title='Stress Level',
        overlaying='y2',
        range=[0, 8]
    ),
    yaxis2=dict(
        title='Sleep (Hrs)',
        side='right',
        range=[0, 10]
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='stress-vs-sleep')
```


<div id="c24ccd8b-5fdb-4175-aaed-f90b4be892f4" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("c24ccd8b-5fdb-4175-aaed-f90b4be892f4", [{"fill": "tozeroy", "name": "Stress Level", "x": ["2019-01-22", "2019-01-23", "2019-01-24", "2019-01-25", "2019-01-26", "2019-01-27", "2019-01-28", "2019-01-29", "2019-01-30", "2019-02-01", "2019-02-03", "2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08", "2019-02-09", "2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15", "2019-02-16", "2019-02-17", "2019-02-17", "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22", "2019-02-23", "2019-02-24", "2019-02-25", "2019-02-26", "2019-02-27", "2019-02-28", "2019-03-01", "2019-03-02", "2019-03-03", "2019-03-04", "2019-03-05", "2019-03-06", "2019-03-09", "2019-03-10", "2019-03-11", "2019-03-12", "2019-03-13", "2019-03-14", "2019-03-15", "2019-03-16", "2019-03-17", "2019-03-18", "2019-03-19", "2019-03-20", "2019-03-21", "2019-03-22", "2019-03-23", "2019-03-24", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28", "2019-03-29", "2019-03-30", "2019-03-31", "2019-04-01", "2019-04-02", "2019-04-03", "2019-04-04", "2019-04-05", "2019-04-06", "2019-04-07", "2019-04-08", "2019-04-09", "2019-04-10", "2019-04-11", "2019-04-12", "2019-04-13", "2019-04-14", "2019-04-15", "2019-04-16", "2019-04-18", "2019-04-20", "2019-04-21", "2019-04-22", "2019-04-23", "2019-04-24", "2019-04-25", "2019-04-26", "2019-04-27", "2019-04-29", "2019-04-30", "2019-05-01", "2019-05-02", "2019-05-03", "2019-05-04", "2019-05-05", "2019-05-06", "2019-05-07", "2019-05-09"], "y": [2, 3, 3, 2, 2, 3, 2, 2, 2, 2, 2, 3, 4, 2, 2, 2, 2, 4, 3, 3, 2, 3, 3, 3, 2, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 4, 3, 3, 4, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 2, 3, 3, 2, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3], "type": "scatter", "uid": "aee557a7-865a-4bcd-ad83-d3ba0a83363a"}, {"name": "Sleep", "x": ["2019-01-22", "2019-01-23", "2019-01-24", "2019-01-25", "2019-01-26", "2019-01-27", "2019-01-28", "2019-01-29", "2019-01-30", "2019-01-31", "2019-02-01", "2019-02-02", "2019-02-03", "2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08", "2019-02-09", "2019-02-10", "2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15", "2019-02-16", "2019-02-17", "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22", "2019-02-23", "2019-02-24", "2019-02-25", "2019-02-26", "2019-02-27", "2019-02-28", "2019-03-01", "2019-03-02", "2019-03-03", "2019-03-04", "2019-03-05", "2019-03-06", "2019-03-07", "2019-03-08", "2019-03-09", "2019-03-10", "2019-03-12", "2019-03-13", "2019-03-14", "2019-03-19", "2019-03-20", "2019-03-21", "2019-03-22", "2019-03-23", "2019-03-24", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28", "2019-03-29", "2019-03-30", "2019-04-01", "2019-04-02", "2019-04-03", "2019-04-04", "2019-04-05", "2019-04-06", "2019-04-07", "2019-04-08", "2019-04-09", "2019-04-10", "2019-04-11", "2019-04-12", "2019-04-14", "2019-04-15", "2019-04-16", "2019-04-17", "2019-04-18", "2019-04-19", "2019-04-20", "2019-04-21", "2019-04-22", "2019-04-23", "2019-04-24", "2019-04-25", "2019-04-26", "2019-04-27", "2019-04-28", "2019-04-29", "2019-04-30", "2019-05-01", "2019-05-02", "2019-05-03", "2019-05-04", "2019-05-05", "2019-05-06", "2019-05-07", "2019-05-08", "2019-05-09", "2019-05-10"], "y": [6.416666666666667, 7.9, 7.216666666666667, 6.333333333333333, 7.733333333333333, 5.3, 5.483333333333333, 7.0, 5.35, 7.35, 5.466666666666667, 7.433333333333334, 7.366666666666666, 5.933333333333334, 8.0, 6.333333333333333, 7.5, 6.083333333333333, 7.85, 6.933333333333334, 7.65, 4.433333333333334, 5.7, 5.266666666666667, 6.883333333333334, 8.483333333333333, 6.816666666666666, 7.05, 7.116666666666666, 6.066666666666666, 6.116666666666666, 5.766666666666667, 8.5, 5.45, 7.2, 5.816666666666666, 6.316666666666666, 6.683333333333334, 5.733333333333333, 6.45, 6.966666666666667, 7.266666666666667, 4.866666666666666, 6.35, 6.933333333333334, 5.383333333333334, 8.85, 8.283333333333333, 8.2, 6.033333333333333, 4.0, 8.05, 5.116666666666666, 6.966666666666667, 5.833333333333333, 7.7, 7.866666666666666, 5.616666666666666, 5.933333333333334, 6.283333333333333, 6.116666666666666, 6.6, 9.3, 7.483333333333333, 5.783333333333333, 6.033333333333333, 6.516666666666667, 6.5, 7.766666666666667, 5.5, 6.45, 7.383333333333334, 5.916666666666667, 5.433333333333334, 6.766666666666667, 7.3, 5.65, 5.25, 6.666666666666667, 7.233333333333333, 6.75, 6.316666666666666, 8.166666666666666, 6.2, 7.383333333333334, 3.75, 7.083333333333333, 7.433333333333334, 8.066666666666666, 6.85, 6.7, 7.116666666666666, 5.683333333333334, 5.766666666666667, 5.966666666666667, 5.866666666666666, 5.85, 5.55, 7.233333333333333, 7.183333333333334, 5.05, 8.133333333333333], "yaxis": "y2", "type": "bar", "uid": "8e6e1f50-5ed5-4b61-a337-7a43dfd9ed08"}], {"barmode": "stack", "title": "Stress Level vs Sleep", "yaxis": {"overlaying": "y2", "range": [0, 8], "title": "Stress Level"}, "yaxis2": {"range": [0, 10], "side": "right", "title": "Sleep (Hrs)"}}, {"showLink": true, "linkText": "Export to plot.ly"});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("c24ccd8b-5fdb-4175-aaed-f90b4be892f4"));});</script>


## Pie Chart of Sleep Stages before Stressful vs Non-stressful Days


```python
stress_sleep_logs = sleep_df[sleep_df['date'].isin(stressful_dates)]
nonstress_sleep_logs = sleep_df[sleep_df['date'].isin(nonstressful_dates)]

stress_sleep_sums = stress_sleep_logs.sum()
nonstress_sleep_sums = nonstress_sleep_logs.sum()
```


```python
fig = {
    "data": [
        {
            "labels": stress_sleep_sums.keys(),
            "values": stress_sleep_sums.values,
            "domain": {"x": [0, .48]},
            "name": "Stressful Sleep Stages",
            "hoverinfo":"label+percent+name",
            "hole": .4,
            "type": "pie"
        },
        {
            "labels": nonstress_sleep_sums.keys(),
            "values": nonstress_sleep_sums.values,
            "domain": {"x": [.52, 1]},
            "name": "Non-stressful Sleep Stages",
            "hoverinfo":"label+percent+name",
            "hole": .4,
            "type": "pie"
        }
    ],
    "layout": {
        "title": "Sleep Stages before Stressful vs Non-stressful Days",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Stress",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Non-stress",
                "x": 0.83,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
```


<div id="4fe0a02e-bb6c-48a9-9691-36fe95b53c3d" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("4fe0a02e-bb6c-48a9-9691-36fe95b53c3d", [{"domain": {"x": [0, 0.48]}, "hole": 0.4, "hoverinfo": "label+percent+name", "labels": ["deep", "light", "rem", "total", "wake"], "name": "Stressful Sleep Stages", "values": [10.916666666666666, 32.93333333333333, 11.866666666666665, 59.96666666666667, 7.633333333333333], "type": "pie", "uid": "37739ff8-abd3-4853-8a7c-e5782dd1fb94"}, {"domain": {"x": [0.52, 1]}, "hole": 0.4, "hoverinfo": "label+percent+name", "labels": ["deep", "light", "rem", "total", "wake"], "name": "Non-stressful Sleep Stages", "values": [97.46666666666664, 312.76666666666665, 94.66666666666669, 543.15, 65.61666666666667], "type": "pie", "uid": "a443e59f-3fc1-4e67-a0c1-88cac974e750"}], {"annotations": [{"font": {"size": 20}, "showarrow": false, "text": "Stress", "x": 0.2, "y": 0.5}, {"font": {"size": 20}, "showarrow": false, "text": "Non-stress", "x": 0.83, "y": 0.5}], "title": "Sleep Stages before Stressful vs Non-stressful Days"}, {"showLink": true, "linkText": "Export to plot.ly"});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("4fe0a02e-bb6c-48a9-9691-36fe95b53c3d"));});</script>


# Habits

How do my actions change when I'm stressed?

## Timeseries Caloric Intake vs Stress


```python
meal_type = { 1: 'Breakfast', 2: 'Morning Snack', 3: 'Lunch', 4: 'Afternoon Snack', 5: 'Dinner', 7: 'Anytime' }
cals_per_nutrient = { 'carbs': 4, 'fat': 9, 'protein': 4 }
```


```python
macronutrient_logs = []
for i in range((END_DATE - START_DATE).days + 1):
    date = START_DATE + dt.timedelta(days=i)
    food_log = fitbit_client.foods_log(date=date)
    macronutrient_logs.append({
        'date': date,
        'cals_from_carbs': food_log['summary']['carbs']*cals_per_nutrient['carbs'],
        'cals_from_fat': food_log['summary']['fat']*cals_per_nutrient['fat'],
        'cals_from_protein': food_log['summary']['protein']*cals_per_nutrient['protein'],
        'total_cals': food_log['summary']['calories'],
        'foods_eaten': [item['loggedFood']['name'] for item in food_log['foods']]
    })
macronutrient_df = pd.DataFrame(macronutrient_logs)
```


```python
cals_trace = go.Bar(
    x=macronutrient_df.date,
    y=macronutrient_df.total_cals,
    name='Calories',
    yaxis='y2'
)

data = [stress_trace, cals_trace]
layout = go.Layout(
    title='Stress Level vs Caloric Intake',
    barmode='stack',
    yaxis=dict(
        title='Stress Level',
        overlaying='y2',
        range=[0, 8]
    ),
    yaxis2=dict(
        title='Calories',
        side='right',
        range=[0, 4000]
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='stacked-bar')
```


<div id="2933d05e-3c1c-41bc-9e1f-33f11b4d8da7" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("2933d05e-3c1c-41bc-9e1f-33f11b4d8da7", [{"fill": "tozeroy", "name": "Stress Level", "x": ["2019-01-22", "2019-01-23", "2019-01-24", "2019-01-25", "2019-01-26", "2019-01-27", "2019-01-28", "2019-01-29", "2019-01-30", "2019-02-01", "2019-02-03", "2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08", "2019-02-09", "2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15", "2019-02-16", "2019-02-17", "2019-02-17", "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22", "2019-02-23", "2019-02-24", "2019-02-25", "2019-02-26", "2019-02-27", "2019-02-28", "2019-03-01", "2019-03-02", "2019-03-03", "2019-03-04", "2019-03-05", "2019-03-06", "2019-03-09", "2019-03-10", "2019-03-11", "2019-03-12", "2019-03-13", "2019-03-14", "2019-03-15", "2019-03-16", "2019-03-17", "2019-03-18", "2019-03-19", "2019-03-20", "2019-03-21", "2019-03-22", "2019-03-23", "2019-03-24", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28", "2019-03-29", "2019-03-30", "2019-03-31", "2019-04-01", "2019-04-02", "2019-04-03", "2019-04-04", "2019-04-05", "2019-04-06", "2019-04-07", "2019-04-08", "2019-04-09", "2019-04-10", "2019-04-11", "2019-04-12", "2019-04-13", "2019-04-14", "2019-04-15", "2019-04-16", "2019-04-18", "2019-04-20", "2019-04-21", "2019-04-22", "2019-04-23", "2019-04-24", "2019-04-25", "2019-04-26", "2019-04-27", "2019-04-29", "2019-04-30", "2019-05-01", "2019-05-02", "2019-05-03", "2019-05-04", "2019-05-05", "2019-05-06", "2019-05-07", "2019-05-09"], "y": [2, 3, 3, 2, 2, 3, 2, 2, 2, 2, 2, 3, 4, 2, 2, 2, 2, 4, 3, 3, 2, 3, 3, 3, 2, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 4, 3, 3, 4, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 2, 3, 3, 2, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3], "type": "scatter", "uid": "8f732bf4-bb72-4e7c-a3dd-2bc8721dea8e"}, {"name": "Calories", "x": ["2019-01-22", "2019-01-23", "2019-01-24", "2019-01-25", "2019-01-26", "2019-01-27", "2019-01-28", "2019-01-29", "2019-01-30", "2019-01-31", "2019-02-01", "2019-02-02", "2019-02-03", "2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08", "2019-02-09", "2019-02-10", "2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15", "2019-02-16", "2019-02-17", "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22", "2019-02-23", "2019-02-24", "2019-02-25", "2019-02-26", "2019-02-27", "2019-02-28", "2019-03-01", "2019-03-02", "2019-03-03", "2019-03-04", "2019-03-05", "2019-03-06", "2019-03-07", "2019-03-08", "2019-03-09", "2019-03-10", "2019-03-11", "2019-03-12", "2019-03-13", "2019-03-14", "2019-03-15", "2019-03-16", "2019-03-17", "2019-03-18", "2019-03-19", "2019-03-20", "2019-03-21", "2019-03-22", "2019-03-23", "2019-03-24", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28", "2019-03-29", "2019-03-30", "2019-03-31", "2019-04-01", "2019-04-02", "2019-04-03", "2019-04-04", "2019-04-05", "2019-04-06", "2019-04-07", "2019-04-08", "2019-04-09", "2019-04-10", "2019-04-11", "2019-04-12", "2019-04-13", "2019-04-14", "2019-04-15", "2019-04-16", "2019-04-17", "2019-04-18", "2019-04-19", "2019-04-20", "2019-04-21", "2019-04-22", "2019-04-23", "2019-04-24", "2019-04-25", "2019-04-26", "2019-04-27", "2019-04-28", "2019-04-29", "2019-04-30", "2019-05-01", "2019-05-02", "2019-05-03", "2019-05-04", "2019-05-05", "2019-05-06", "2019-05-07", "2019-05-08", "2019-05-09", "2019-05-10"], "y": [2723, 2503, 2520, 2995, 3104, 2478, 3338, 2377, 2800, 2065, 3079, 2905, 3074, 2228, 2183, 2564, 1925, 2266, 2508, 2524, 2623, 2025, 2326, 2112, 2303, 3163, 2721, 1949, 2209, 2733, 2704, 2293, 3268, 2895, 2052, 1836, 3086, 2235, 2215, 2559, 2588, 1680, 2671, 2960, 1551, 2755, 2096, 2386, 1503, 3362, 1942, 1820, 2469, 2052, 3113, 3468, 3181, 1830, 2123, 3312, 1897, 2718, 2375, 3122, 2785, 2494, 1641, 2370, 2847, 2146, 2235, 2096, 1609, 3379, 2227, 1916, 2771, 1889, 2206, 1958, 2062, 3496, 1736, 2339, 2528, 2092, 2217, 2683, 3072, 1623, 2581, 1920, 2009, 2004, 2681, 1479, 1086, 2015, 1483, 2184, 240, 2123, 3887, 2499, 2378, 1706, 2458, 1927, 1802], "yaxis": "y2", "type": "bar", "uid": "fef85996-ebcd-491e-bd49-3dc4ed4d51b2"}], {"barmode": "stack", "title": "Stress Level vs Caloric Intake", "yaxis": {"overlaying": "y2", "range": [0, 8], "title": "Stress Level"}, "yaxis2": {"range": [0, 4000], "side": "right", "title": "Calories"}}, {"showLink": true, "linkText": "Export to plot.ly"});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("2933d05e-3c1c-41bc-9e1f-33f11b4d8da7"));});</script>


## Pie Chart of Carbs, Fat, Protein Avg on Stressful vs Non-stressful Days


```python
stress_food_logs = macronutrient_df[macronutrient_df['date'].isin(stressful_dates)]
nonstress_food_logs = macronutrient_df[macronutrient_df['date'].isin(nonstressful_dates)]

stress_food_sums = stress_food_logs.sum()
nonstress_food_sums = nonstress_food_logs.sum()
```


```python
fig = {
    "data": [
        {
            "labels": stress_food_sums.keys(),
            "values": stress_food_sums.values,
            "domain": {"x": [0, .48]},
            "name": "Stressful Caloric Intake",
            "hoverinfo":"label+percent",
            "hole": .4,
            "type": "pie"
        },
        {
            "labels": nonstress_food_sums.keys(),
            "values": nonstress_food_sums.values,
            "domain": {"x": [.52, 1]},
            "name": "Non-stressful Caloric Intake",
            "hoverinfo":"label+percent",
            "hole": .4,
            "type": "pie"
        }
    ],
    "layout": {
        "title": "Caloric Intake on Stressful vs Non-stressful Days",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Stress",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Non-stress",
                "x": 0.83,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
```


<div id="1d7329e6-94e2-4af9-96d6-6eccf6b55822" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("1d7329e6-94e2-4af9-96d6-6eccf6b55822", [{"domain": {"x": [0, 0.48]}, "hole": 0.4, "hoverinfo": "label+percent", "labels": ["cals_from_carbs", "cals_from_fat", "cals_from_protein", "foods_eaten", "total_cals"], "name": "Stressful Caloric Intake", "values": [12596.52, 7867.8, 4681.200000000001, ["Ham & Provolone on Kaiser Roll", "Muffin, Double Chocolate", "Yogurt, Lowfat, Plain", "California Roll", "Turkey & Provolone Cheese on Kaiser Roll", "Oatmeal Raisin Nut Cookie", "Yogurt, Lowfat, Plain", "Tostones, Patacones Original", "Fried Clams, Crunchy", "Potato Wedges", "Hotdog Buns", "Chicken Salad on Wheat", "Pistachios", "UDI's Gluten Free Blueberry Muffin", "Banana, Raw", "Turkey & Pepperjack on Multigrain Bread", "Multigrain Snacks, Harvest Cheddar", "Cookies, Sandwich, Chocolate", "Sandwich Crackers, Peanut Butter", "Java Chocolate Chunk Ice Cream", "Black Bean Patty - Regular", "Mexican Casserole", "Roasted Red Pepper Hummus", "Carrots", "Coffee Drink, Mocha", "Crunchy Granola Bars, Oats & Honey", "Hamburger Roll", "Chicken Parmesan", "Banana, Raw", "Buffalo Chicken Wrap", "Chocolate Fat Free Milk", "Snack Mix", "WG White Choc Macaroon Bar (Almonds)", "French Fries", "Chicken Tenders", "Banana, Raw", "Greek Yogurt", "Buffalo Chicken Salad", "Chewy Granola Bar, Chocolate Chunk", "Granola", "Black Bean Quesadilla", "Chocolate Cake", "Clam, Fried", "Rice", "Stuffed Mushrooms", "Black Beans", "Organic Yogurt, New England Maple", "Craisins Dried Strawberry", "Crunchy Granola Bars, Oats & Honey", "Ham & Swiss on Multigrain Bread", "Trail Mix Granola Bar, Fruit & Nut", "Caramel Sauce", "Egg", "Crepes", "Ham", "Banana, Raw", "Cappuccino", "Caesar Salad", "Chicken", "Hummus", "Lentil, Boiled W/salt", "Rice", "Flatbread", "Hummus", "Ham and Cheese Sandwich", "Avocado Chicken Burger", "Ice Cream Sandwich, Salted Caramel", "French Fries", "Cookies", "White Roll", "Quince, Raw", "Chicken", "Yogurt, Lowfat, Plain", "Turkey & Provolone Cheese on Kaiser Roll", "Granola", "Protein Box w/ Egg, Cheese & Muesli", "Banana, Raw", "Turkey & Provolone Cheese on Kaiser Roll", "California Roll", "Japanese Beef Bowl", "Salmon Burger", "Black Bean Quesadilla", "Enchilada W/cheese", "Black Beans", "Rice", "Nutri Grain Bar, Apple Cinnamon", "Chocolate Milk, Low Fat", "Crunchy Granola Bars, Oats & Honey", "Banana, Raw", "Chicken Parmesan", "Pop Tarts, Frosted Brown Sugar Cinnamon", "Rehab, Peach Tea + Energy", "Broccoli Florets", "Nonfat Yogurt, Plain", "Vegan Cheese Pizza", "Black Bean Patty", "Crunchy Granola Bars, Oats & Honey", "Chicken Caesar Salad", "Granola", "Yogurt, Lowfat, Plain", "Turkey & Provolone Cheese on Kaiser Roll", "California Roll", "Protein Box w/ Egg, Cheese & Muesli", "Vitamin Water", "Granola", "Yogurt, Lowfat, Plain", "Black Bean Quesadilla", "Enchilada W/cheese & Beef", "Banana, Raw", "Chicken Burger", "Banana, Raw", "Buffalo Chicken Salad", "Turkey & Pepperjack on Multigrain Bread", "Yogurt, Lowfat, Plain", "Muffin, Double Chocolate", "Granola", "Banana, Raw", "Chocolate Fat Free Milk", "Protein Box w/ Egg, Cheese & Muesli", "Turkey & Pepperjack on Multigrain Bread", "Greek Nonfat Yogurt, Vanilla", "Lasagna", "Sweet Potato", "Samosa", "Granola", "Greek Yogurt"], 25526], "type": "pie", "uid": "1de52b28-8feb-4790-b236-0f39388b37a4"}, {"domain": {"x": [0.52, 1]}, "hole": 0.4, "hoverinfo": "label+percent", "labels": ["cals_from_carbs", "cals_from_fat", "cals_from_protein", "foods_eaten", "total_cals"], "name": "Non-stressful Caloric Intake", "values": [109485.08, 65601.27, 36425.96, ["Lowfat Yogurt, Vanilla", "Turkey and Cheese Sandwich", "Muffin, Double Chocolate", "Granola Bars, Chewy Fudge Dipped Chocolate Chip", "Turkey & Provolone Cheese on Kaiser Roll", "Banana, Raw", "Granola, Oats Honey Raisins & Almonds", "Muffin, Double Chocolate", "Grapes", "Nonfat Yogurt", "Stuffed Pepper", "Chicken Enchilada Suiza", "Mac and Cheese", "Granola Bars, Chewy Fudge Dipped Chocolate Chip", "Scrambled Eggs", "Chorizo Hash", "Skim Milk", "Baked Beans", "Blueberry Scone", "Blends Nonfat Yogurt, French Vanilla", "Mocha Scuffin", "Turkey & Pepperjack on Multigrain Bread", "California Roll", "Vegetarian Bowl", "Granola Bars, Chewy Fudge Dipped Chocolate Chip", "Muffin, Double Chocolate", "Low Fat Iced Coffee", "Turkey & Provolone Cheese on Kaiser Roll", "Lowfat Yogurt, Vanilla", "California Roll", "Turkey & Pepperjack on Multigrain Bread", "Banana, Raw", "Rice Snacks, Caramel Corn", "Peach Smoothie", "Cake Cone", "Veggie Wrap", "Crispy Battered Fish Fillet", "Tuna Roll", "Vanilla Frozen Yogurt", "Muffin, Double Chocolate", "Granola Bars, Chewy Fudge Dipped Chocolate Chip", "Ham & Provolone on Kaiser Roll", "Lowfat Yogurt, Vanilla", "Dumpling, Shrimp", "Pork Bun", "Vege Potstickers", "Blueberry Parfait 12oz", "Buffalo Chicken Pizza", "Granola Bars, Chewy Fudge Dipped Chocolate Chip", "Parmesan Chicken Sandwich, Skinny", "Chicken Tikka Masala", "Trail Mix", "Vodka", "Banana, Raw", "White Choc Cran Almond Cookie", "Spicy Italian Grinder", "Chicken Bacon Ranch Original Crust Pizza, Small", "Quesadilla Pizza", "Chewy Granola Bar, Chocolate Chunk", "Trail Mix Granola Bar, Fruit & Nut", "Crunchy Granola Bars, Oats & Honey", "White Choc Cran Almond Cookie", "Falafel Hummus Wrap", "California Roll", "White Bean Mac N' Cheese Crunch", "Eggplant Parmesan", "1% Low Fat Milk", "Pop Tarts Cereal, Frosted Brown Sugar Cinnamon", "Ham and Cheese Omelette", "Fudge Dipped Chocolate Chip Granola Bar", "Turkey & Provolone Cheese on Kaiser Roll", "Triple Chocolate Cookie", "Rehab Energy Drink, Tea + Lemonade + Energy", "Peanut Butter and Jelly Sandwich", "Sushi Tofu Roll", "Garden Salad", "Ice Cream, Homemade Vanilla", "Muffin, Double Chocolate", "Turkey & Provolone Cheese on Kaiser Roll", "Blueberries", "Banana, Raw", "Lowfat Yogurt, Vanilla", "Blueberries", "California Roll", "Sprite, Fountain", "Multigrain Snacks, Harvest Cheddar", "Muffin, Double Chocolate", "Coffee Drink, Mocha", "Peanut Power Bar", "Burrito, Veggie", "Stir-Fry Pork", "Tart Cherry", "Cake, Chocolate", "Pop Tarts Cereal, Frosted Brown Sugar Cinnamon", "Cauliflower, Raw", "Blends, Low fat French Vanilla", "Turkey & Pepperjack on Multigrain Bread", "Blueberries", "Muffin, Double Chocolate", "Blends, Low fat French Vanilla", "Rice Snacks, Caramel Corn", "Blueberries", "Ham & Provolone on Kaiser Roll", "Banana, Raw", "Chocolate Chip Cookie, Sugar Free", "Egg, Chicken, Hard-boiled", "Spicy Tuna Roll", "Balsamic Vinaigrette Dressing", "Java Chocolate Chunk Ice Cream", "Sticky Bun", "Baked Beans", "Fried Potatoes", "Organic Yogurt, New England Maple", "Scrambled Eggs", "Black Bean Burger", "Buffalo Chicken Wrap", "Fries, Sweet Potato", "Vegan Cheese Pizza", "Meatloaf", "Tagalongs", "California Roll", "Broccoli Cheddar Cakes", "Pop Tarts Cereal, Frosted Brown Sugar Cinnamon", "Nonfat Yogurt, Plain", "Banana, Raw", "Mexican Casserole", "Turkey Sausage", "Scrambled Eggs", "Strawberry / Strawberries", "Ginger Ale", "Chicken Caesar Salad", "Seasoned Croutons", "Scoops", "Burrito Bowls", "Half & Half, Just a Tad Sweet", "M&M Fudge Bar", "Ham & Swiss on Kaiser Roll", "Ham & Provolone on Kaiser Roll", "Vegetable Mushroom Bun", "Dumpling, Shrimp", "Lotus Leaf Bun", "Scrambled Eggs", "Hash Browns", "Turkey Sausage", "Trail Mix", "Turkey & Provolone Cheese on Kaiser Roll", "Whole Wheat Chocolate Chip Cookie", "Chicken Parmesan", "Beer", "Lagunitas IPA, 12 fl oz", "Turkey & Pepperjack on Multigrain Bread", "Blueberries", "Muffin, Double Chocolate", "Lowfat Yogurt, Vanilla", "Banana, Raw", "California Roll", "Protein Box w/ Egg, Cheese & Muesli", "Chicken Parmesan", "Banana, Raw", "Oatmeal Raisin Nut Cookie", "Tortilla Chips, Cool Ranch", "Chicken Fingers", "Chewy Granola Bar, Chocolate Chunk", "Pop Tarts Cereal, Frosted Brown Sugar Cinnamon", "Muffin, Double Chocolate", "Banana, Raw", "Yogurt, Lowfat, Plain", "Blueberries", "Turkey & Pepperjack on Multigrain Bread", "Carrots", "Macaroni Salad", "Yogurt, Lowfat, Plain", "Buffalo Chicken Wrap", "Turkey & Provolone Cheese on Kaiser Roll", "Banana, Raw", "Trail Mix Granola Bar, Fruit & Nut", "White Choc Cran Almond Cookie", "Rice Snacks, Caramel Corn", "Banana, Raw", "Yogurt, Lowfat, Plain", "Ham & Provolone on Kaiser Roll", "Blueberries", "Turkey & Pepperjack on Multigrain Bread", "Sun Chips Harvest Cheddar", "Banana, Raw", "Curried Eggplant and Lentil Soup, Small", "Wild Salmon Sliders, Individual", "California Roll", "Tangerine", "Trail Mix Granola Bar, Fruit & Nut", "Banana, Raw", "Scrambled Eggs", "Corned Beef Hash", "Yogurt, Lowfat, Plain", "California Roll", "Blueberries", "Yogurt, Lowfat, Plain", "Ham & Provolone on Kaiser Roll", "Crunchy Granola Bars, Oats & Honey", "Chicken over Rice, Korean Style", "Turkey & Provolone Cheese on Kaiser Roll", "Beer, Regular", "Nutri Grain Bar, Apple Cinnamon", "Home Fries", "Baked Beans", "Bacon", "Scrambled Eggs", "Organic Yogurt, New England Maple", "Nonfat Yogurt, Blueberry On The Bottom", "Turkey Pesto Croissant", "Cauliflower, Raw", "Black Bean & Cheese Enchilada", "Vegetable Fried Rice", "Roasted Red Pepper Hommus with Hommus Chips", "Banana, Raw", "Turkey & Pepperjack on Multigrain Bread", "Muffin, Double Chocolate", "Rice Krispies Treats, the Original", "Banana, Raw", "Turkey & Provolone Cheese on Kaiser Roll", "Pistachios", "Banana, Raw", "Tortilla Chips, Cool Ranch", "Ham & Cheese Sandwich", "Trail Mix Granola Bar, Fruit & Nut", "Home Fries", "Greek Yogurt, Vanilla", "Scrambled Eggs", "Beans", "Corned Beef Hash, Canned", "Soy Milk, Chocolate (soymilk)", "Carrots", "Yogurt, Lowfat, Plain", "Macaroni Salad", "Buffalo Chicken Wrap", "Cherries & Almonds in Dark Chocolate, 55% cocoa", "Multigrain Snacks, Harvest Cheddar", "Cinnamon Thin Cookies", "Dr. Pepper", "Turkey & Pepperjack on Multigrain Bread", "Falafel Wrap", "Muffin, Double Chocolate", "Yogurt, Lowfat, Plain", "Banana, Raw", "Turkey & Pepperjack on Multigrain Bread", "Banana, Raw", "Falafel Wrap", "Crunchy Granola Bars, Oats & Honey", "Blueberries", "Apple Fritters", "Yogurt, Lowfat, Plain", "Corned Beef Hash, Canned", "Scrambled Eggs", "Blueberries", "California Roll", "Banana, Raw", "Yogurt, Lowfat, Plain", "BBQ Chicken Flatbread, 1 Flatbread", "Lemonade", "Potato Chips", "Pop Tarts Cereal, Frosted Brown Sugar Cinnamon", "Granola Bar, Chocolate Chunk", "Oreo Coconut Bar", "Banana, Raw", "Scrambled Eggs", "Greek Yogurt, Vanilla", "Raisins", "Blueberry Scone", "Corned Beef Hash, Canned", "Baked Beans", "Home Fries", "Stuffed Peppers", "Roasted Red Pepper Hummus", "Vegan Pizza", "Carrot", "Pork Enchiladas, Small", "Haddock, Cooked", "Eggplant Rollettes with Sauce & Cheese", "Chocolate Strawberry Square", "Chicken Parm Sandwich", "Candy Bars", "Banana, Raw", "Chewy Granola Bar, Chocolate Chunk", "Ham & Swiss on Kaiser Roll", "Tangerine, Mandarin, Raw", "Vegetable Mushroom Bun", "Pork Potstickers", "Red Bean Bun", "Peanuts", "Greek Yogurt, Vanilla", "Home Cooked Salted Virginia Peanuts", "Chewy Granola Bar, Chocolate Chunk", "Frozen Yogurt, Fat Free Chocolate", "Walnut Brownie", "Pizza", "Banana, Raw", "Greek Vanilla Nonfat Yogurt", "Ham & Swiss on Kaiser Roll", "Granola", "Energy & Juice Drink, Pacific Punch", "Breaded Pork Sirloin", "California Roll", "Carrots", "Sweet Potato, Baked W/salt", "Pickles", "Home Cooked Salted Virginia Peanuts", "Pop Tarts Cereal, Frosted Brown Sugar Cinnamon", "Candies, Chewy, Original Fruits", "Chicken Salad on Wheat", "Yogurt, Lowfat, Plain", "Banana, Raw", "Muffin, Double Chocolate", "California Roll", "Blueberries", "Banana, Raw", "Yogurt, Lowfat, Plain", "Mini Wheats", "Rice", "Guacamole", "Beans", "Clam, Fried", "Hot Dog Buns", "White Cheddar Poppables", "Yogurt, Lowfat, Plain", "Blueberries", "Turkey & Provolone Cheese on Kaiser Roll", "Muffin, Double Chocolate", "Trail Mix Granola Bar, Fruit & Nut", "Rice Snacks, Caramel Corn", "Protein Box w/ Egg, Cheese & Muesli", "Banana, Raw", "Banana, Raw", "Chocolate Fat Free Milk", "Crunchy Granola Bars, Oats & Honey", "Pizza", "WG Cran Sunflower Choc Chip Cookie", "Meatball Sub, 12\" Sub", "Muffin, Double Chocolate", "Turkey & Provolone Cheese on Kaiser Roll", "Yogurt, Lowfat, Plain", "Blueberries", "Banana, Raw", "Turkey & Pepperjack on Multigrain Bread", "Protein Box w/ Egg, Cheese & Muesli", "Chocolate Fat Free Milk", "Nutri Grain Bar, Apple Cinnamon", "Ham & Swiss on Multigrain Bread", "Chocolate Fat Free Milk", "Chocolate Frozen Yogurt", "Cake Cone", "Crispy Chicken Wrap", "Scrambled Eggs", "Yogurt, Lowfat, Plain", "Mexican Casserole", "Banana, Raw", "Turkey Sausage", "Turkey & Pepperjack on Multigrain Bread", "Muffin, Double Chocolate", "Pop Tarts Cereal, Frosted Brown Sugar Cinnamon", "Rice, Bowl", "Trail Mix", "Guacamole", "Scrambled Eggs", "Enchilada W/cheese & Beef", "Nan, Wheat", "Bacon", "Crab Cake, Blue", "Banana, Raw", "Rice, Bowl", "Crunchy Granola Bars, Oats & Honey", "Powerade Mountain Blast", "Chicken Tenders", "French Fries", "Banana, Raw", "Turkey & Provolone Cheese on Kaiser Roll", "Chicken Parmesan", "Ham & Provolone on Kaiser Roll", "Chicken Bacon Ranch", "Trail Mix", "Seven Layer Bar", "Nutri Grain Bar, Apple Cinnamon", "Monterey Jack Cheese Snack Sticks", "Peanut Butter and Fluff Sandwich", "Crunchy Granola Bars, Oats & Honey", "Nutri-Grain Cereal Bar, Apple Cinnamon", "Whole Wheat Chocolate Chip Cookie", "Strawberry / Strawberries", "Yogurt, Lowfat, Plain", "Muffin, Double Chocolate", "Turkey & Provolone Cheese on Kaiser Roll", "Banana, Raw", "Crunchy Granola Bars, Oats & Honey", "Protein Box w/ Egg, Cheese & Muesli", "California Roll", "Falafel Wrap", "Trail Mix Granola Bar, Fruit & Nut", "Blueberries", "Yogurt, Lowfat, Plain", "Protein Box w/ Egg, Cheese & Muesli", "Turkey & Pepperjack on Multigrain Bread", "Banana, Raw", "Turkey & Provolone Cheese on Kaiser Roll", "Protein Box w/ Egg, Cheese & Muesli", "Banana, Raw", "Cookie Dough Ice Cream", "Rainbow Salad", "Spicy Tuna Roll", "Turkey & Pepperjack on Multigrain Bread", "Banana, Raw", "Chicken Caesar Salad", "WG Cran Sunflower Choc Chip Cookie", "Yogurt, Lowfat, Plain", "Blueberries", "California Roll", "Sun Chips", "Crunchy Granola Bars, Oats & Honey", "Chocolate Fat Free Milk", "Meatball Sub", "Banana, Raw", "Trail Mix Granola Bar, Fruit & Nut", "Yogurt, Soft Serve, Vanilla", "WG Cran Sunflower Choc Chip Cookie", "Iced Tea, Sweetened", "Yogurt, Lowfat, Plain", "Home Cooked Salted Virginia Peanuts", "Muffin, Double Chocolate", "Ham & Provolone on Kaiser Roll", "Blueberries", "Pretzels, Tiny Twists Original", "Protein Box w/ Egg, Cheese & Muesli", "Salmon Burger", "Eat a Bowl, Chana Masala Chickpeas & Rice", "Biscuits with Peanut Butter", "Cheese Snack Sticks, Lite Mild Cheddar", "Trail Mix Granola Bar, Fruit & Nut", "Blueberries", "Turkey & Pepperjack on Multigrain Bread", "Yogurt, Lowfat, Plain", "Muffin, Double Chocolate", "Banana, Raw", "California Roll", "Protein Box w/ Egg, Cheese & Muesli", "Nachos", "Horchata", "Pupusas, Bean & Cheese Gorditas", "Baked Beans", "Milk", "Kielbasa, Polish, Turkey & Beef", "Nonfat Greek Yogurt, Plain", "Granola", "Scrambled Eggs", "Crunchy Granola Bars, Oats & Honey", "Chicken Teriyaki Roll, Avocado, Brown Rice", "Pork Potstickers", "Broccoli Florets", "Lotus Leaf Bun", "Chocolate Pudding", "California Roll", "Turkey & Pepperjack on Multigrain Bread", "Oatmeal Raisin Cookie", "Banana, Raw", "Turkey & Provolone Cheese on Kaiser Roll", "Crunchy Granola Bars, Oats & Honey", "Protein Box w/ Egg, Cheese & Muesli", "Yogurt, Lowfat, Plain", "Stuffed Peppers", "Broccoli Cheddar Cakes", "Greek Yogurt", "Broccoli Florets", "Banana, Raw", "Crunchy Granola Bars, Oats & Honey", "Caesar Dressing", "Yogurt, Lowfat, Plain", "Seasoned Premium Croutons", "Chicken Caesar Salad", "Turkey & Provolone Cheese on Kaiser Roll", "Muffin, Double Chocolate", "Protein Box w/ Egg, Cheese & Muesli", "Pretzels, Tiny Twists Original", "Blonde Brownie w/ White Choc Chip", "Chicken Caesar Wrap", "Craisins Dried Strawberry", "Blue Cheese (Bleu)", "Chicken Tenders", "Omelette", "Blueberries", "Cinnamon Rolls, with Icing, Reduced Fat", "Yogurt, Lowfat, Blueberry", "Granola, Fruit & Nut", "Turkey, Spinach and Swiss Cheese Wrap", "Macaroni & Cheese", "Granola Bar, Chocolate Chip", "Orange Juice", "Scrambled Eggs", "Blueberry Muffin", "Turkey Sausage", "Turkey & Pepperjack on Multigrain Bread", "Garden Veggie Straws Sea Salt", "Yogurt Mousse, Vanilla Creme", "Classic Hummus", "Yogurt Parfait", "Chicken Pesto Sandwich", "Red Wine", "Pasta, Boiled", "Potato Salad", "Whole Wheat Dinner Roll", "Chicken Fried Rice", "Beer, Regular", "White Bread", "Bacon", "Orange Juice", "Sausage", "Egg", "Baked Beans", "Coffee with Cream Liqueur (34 Proof)", "Spaghetti", "Chicken Milanese", "Masala Dosa", "Beer, Regular", "Idli", "Egg", "White Bread", "Avocado, Raw", "Baked Beans", "Chicken & Steak Fajitas Chiquitas", "French Fries", "M&Ms", "Sicilian Pizza, Individual", "Miso Soup", "Gyoza, Pork", "Tuna Roll", "Fruit and Yogurt Parfait", "Ham", "Cinnamon Donut", "Sourdough Toast", "Coffee Creamer", "Beer, Regular", "Vegetarian Burger, Veggie, Garden", "Hamburger Roll", "French Fries", "M&Ms", "Cuban, 6\"", "Chocolate Partly Skimmed Milk", "Yogurt Parfait", "Trail Mix Granola Bar, Fruit & Nut", "Cheeseburger", "Tiny Twists, Original", "Rice", "Raisins", "Beans", "Greek Yogurt, Vanilla", "Peanut, Raw", "Black Bean Quesadilla", "Guacamole", "M&Ms", "Gold Bears", "Stuffed Sandwiches, Chicken Melt", "Chewy Granola Bar, Chocolate Chunk", "Muffin, Double Chocolate", "Turkey & Pepperjack on Multigrain Bread", "Yogurt, Lowfat, Plain", "Banana, Raw", "California Roll", "Banana, Raw", "Protein Box w/ Egg, Cheese & Muesli", "Falafel Wrap, Lemony Roasted Garlic Hummus", "Home Fries", "Sausage", "Yogurt, Lowfat, Plain", "Scrambled Eggs", "Banana, Raw", "Banana, Raw", "Protein Box w/ Egg, Cheese & Muesli", "Turkey & Provolone Cheese on Kaiser Roll", "WG Cran Sunflower Choc Chip Cookie", "Banana, Raw", "Trail Mix Granola Bar, Fruit & Nut", "Crunchy Granola Bars, Oats & Honey", "Banana, Raw", "Yogurt, Lowfat, Plain", "Muffin, Double Chocolate", "Turkey & Pepperjack on Multigrain Bread", "Rice Snacks, Caramel Corn", "Turkey & Pepperjack on Multigrain Bread", "California Roll", "1% Lowfat Milk", "Turkey & Provolone Cheese on Kaiser Roll", "Nutri-Grain Cereal Bar, Apple Cinnamon", "Nekot Cookies, Real Peanut Butter", "Lamb, Stew Meat", "Rice", "M&Ms", "Potato Chips, Jalapeno", "Hamburger Roll", "Chicken Parmesan", "Chewy Granola Bar, Chocolate Chunk", "Quesadilla", "Coffee Drink, Mocha", "Refried Beans", "Rice", "Pupusas, Pork & Cheese Stuffed Corn Tortillas", "Rice", "Banana, Raw", "Lamb, Retail Cuts", "Trail Mix Granola Bar, Fruit & Nut", "Chicken Quesadilla", "Yogurt, Whole Milk, Simply Plain", "Vegetable Bun", "Breakfast Pizza", "Granola", "Pork Pot Stickers", "Blueberry Muffins", "Granola", "Greek Nonfat Yogurt, Vanilla", "Rainbow Salad", "Granola", "Muffin, Double Chocolate", "Yogurt, Lowfat, Plain", "Turkey & Provolone Cheese on Kaiser Roll", "Protein Box w/ Egg, Cheese & Muesli", "California Roll", "Broccoli Cheddar Cakes", "Granola", "Enchilada W/cheese & Beef", "Greek Nonfat Yogurt, Vanilla", "Rice", "Kidney Bean, Canned", "Broccoli Florets", "Craisins Dried Strawberry", "Chicken Caesar Salad", "Muffin, Double Chocolate", "Turkey & Provolone Cheese on Kaiser Roll", "High Protein Milk Shake, Chocolate", "WG Cran Sunflower Choc Chip Cookie", "Turkey & Provolone Cheese on Kaiser Roll", "Banana, Raw", "Turkey & Pepperjack on Multigrain Bread", "Greek Yogurt Cakes, Blueberry", "Chicken Enchilada Suiza Entree", "Stuffed Peppers", "Clam, Fried", "Greek Yogurt", "Granola", "Trail Mix Granola Bar, Fruit & Nut", "Chicken Caesar Salad", "Turkey & Provolone Cheese on Kaiser Roll", "Nutri Grain Bar, Apple Cinnamon", "Yogurt, Lowfat, Plain", "Granola", "Turkey & Pepperjack on Multigrain Bread", "Granola", "Yogurt, Lowfat, Plain", "California Roll", "Pretzels", "Meatball Grilled Flatbread", "WG Cran Sunflower Choc Chip Cookie", "Crunchy Granola Bars, Oats & Honey", "Cheese Stick, Cheddar", "Yogurt, Lowfat, Plain", "Granola", "Banana, Raw", "Casserole, Mexican Style", "Scrambled Eggs", "Bacon", "Falafel Wrap", "Hamburger Roll", "Vitamin Water", "Impossible Burger", "Cheese Stick, Cheddar", "Trail Mix Granola Bar, Fruit & Nut", "Gummy Bear", "Yogurt, Lowfat, Plain", "Banana, Raw", "Granola", "Turkey & Pepperjack on Multigrain Bread", "Muffin, Double Chocolate", "Protein Box w/ Egg, Cheese & Muesli", "Banana, Raw", "California Roll", "Trail Mix Granola Bar, Fruit & Nut", "Chicken Caesar Salad", "Beer, Regular", "Loca Moca Java Monster Coffee + Energy", "Buffalo Chicken Wrap", "Banana, Raw", "Chewy Granola Bar, Chocolate Chunk", "Rainbow Salad", "Banana, Raw", "Tres Leches", "Pork", "Rice, Brown, Raw", "Fried Plantains, Tostones", "Chicken Kabob", "Ham & Swiss on Multigrain Bread", "Trail Mix Granola Bar, Fruit & Nut", "Breakfast Burrito", "Black Beans", "Sour Cream Sauce", "Huevos ala Mexicana", "Blueberry Lemonade", "Guacamole", "Java Chocolate Chunk Ice Cream", "Granola", "Broccoli", "Vegan Cheese Pizza", "Yogurt, Lowfat, Plain", "California Roll", "Black Bean Patty", "Granola", "Home Fries", "Canadian Bacon", "Yogurt, Lowfat, Plain", "Scrambled Eggs", "Muffin, Double Chocolate", "Banana, Raw", "California Roll", "Rice Snacks, Caramel Corn", "Banana, Raw", "Roasted Cauliflower", "Quesadilla", "Yogurt, Lowfat, Plain", "Granola", "Coca-Cola Classic Coke", "Crunchy Granola Bars, Oats & Honey", "Yogurt, Lowfat, Plain", "Banana, Raw", "Turkey & Provolone Cheese on Kaiser Roll", "Granola", "Protein Box w/ Egg, Cheese & Muesli", "California Roll", "Trail Mix Granola Bar, Fruit & Nut", "Banana, Raw", "Grilled Ham and Cheese", "Greek Yogurt, Vanilla", "Banana, Raw", "Granola", "Chewy Granola Bar, Chocolate Chunk", "Yogurt, Lowfat, Plain", "Granola", "Turkey & Provolone Cheese on Kaiser Roll", "Chewy Granola Bar, Chocolate Chunk", "Chicken Caesar Salad", "Cookies, Sandwich, Chocolate", "Protein Box w/ Egg, Cheese & Muesli", "Black Bean Patty", "Granola", "Vegan Cheese Pizza", "Greek Yogurt, Vanilla", "Banana, Raw", "Yogurt, Lowfat, Plain", "Granola", "Turkey & Pepperjack on Multigrain Bread", "Chocolate Fat Free Milk", "Banana, Raw", "California Roll", "Chicken Caesar Salad", "WG Cran Sunflower Choc Chip Cookie", "Banana, Raw", "Crunchy Granola Bars, Oats & Honey", "Granola", "Muffin, Double Chocolate", "Yogurt, Lowfat, Plain", "Chicken Caesar Salad", "Turkey & Pepperjack on Multigrain Bread", "Protein Box w/ Egg, Cheese & Muesli", "Macaroni & Cheese", "Beans", "Italian Sausage", "Stuffed Cheesy Bread", "Chicken Tenders", "Medium Hand Tossed  2-3 Toppings Pizza", "Rainbow Salad", "Stonyfield Org Fat Free Frozen Yogurt Vanilla (3/4 cup)", "Granola", "Red Bull", "Yogurt, Soft Serve, Vanilla", "Broccoli Florets", "Chicken Parmesan", "Black Bean Patty", "Burrito Supreme - Beef", "Cinnamon Twists", "Taco, Volcano Taco", "Chalupa Supreme - Chicken", "Muffin, Double Chocolate", "Granola", "Yogurt, Lowfat, Plain", "Ham & Provolone on Kaiser Roll", "California Roll", "Chicken Caesar Salad", "Banana, Raw", "Banana, Raw", "Nutri Grain, Soft Baked Breakfast Bars, Apple Cinnamon", "Turkey & Provolone Cheese on Kaiser Roll", "Chewy Granola Bar, Chocolate Chunk", "Granola", "Yogurt, Lowfat, Plain", "Crunchy Granola Bars, Oats & Honey", "Chicken Caesar Salad", "Banana, Raw", "Turkey & Pepperjack on Multigrain Bread", "Protein Box w/ Egg, Cheese & Muesli", "Banana, Raw", "Nutri Grain Bar, Apple Cinnamon", "Dolmas", "Yogurt, Lowfat, Plain", "Tuna Roll", "Granola", "Dolmas", "Trail Mix Granola Bar, Fruit & Nut", "Nutri Grain Bar, Apple Cinnamon", "Chicken Caesar Salad", "Turkey & Pepperjack on Multigrain Bread", "Banana, Raw", "Greek Yogurt, Nonfat, Plain", "Protein Box w/ Egg, Cheese & Muesli", "Granola", "Chocolate Fat Free Milk", "Buffalo Chicken Salad", "Gelato", "Enchilada W/cheese & Beef", "Samosa", "Crunchy Granola Bars, Oats & Honey", "Scrambled Eggs", "Corned Beef Hash, Canned", "Oikos, Greek Yogurt, Quart Size, Fat Free Plain", "Granola", "Baked Beans", "California Roll", "Turkey & Pepperjack on Multigrain Bread", "Banana, Raw", "Chewy Granola Bar, Chocolate Chunk", "Chicken Caesar Wrap", "Banana, Raw", "Trail Mix Granola Bar, Fruit & Nut", "Trail Mix Granola Bar, Fruit & Nut", "Turkey & Provolone Cheese on Kaiser Roll", "Crunchy Granola Bars, Oats & Honey", "Chewy Granola Bar, Chocolate Chunk", "Banana, Raw", "Scrambled Eggs", "Zebra Cake", "Vegetable Bun", "Sausage", "Baked Beans", "Dumplings", "Guacamole", "Fried Plantains", "Chicken", "Rice", "Refried Beans", "Cauliflower, Raw", "White Cheddar Poppables", "Cupcakes, Chocolate", "Scrambled Eggs", "Baked Beans", "Kielbasa, Pork & Beef (kolbassy)", "Rice", "Broccoli, Raw", "Guacamole", "Beans, Black", "Beans", "Broccoli, Raw", "Guacamole", "Rice", "Trail Mix Granola Bar, Fruit & Nut", "Scrambled Eggs", "Granola", "Greek Yogurt, Nonfat, Plain", "Chorizo Hash", "Chewy Granola Bar, Chocolate Chunk", "Banana, Raw", "Trail Mix Granola Bar, Fruit & Nut", "Ravioli", "Chicken", "Burrito Ingredient, Brown Rice", "Burrito Bowl Ingredient, Cheese", "Burrito Bowl Ingredient, Guacamole", "Burrito Ingredient, Black Beans", "Yogurt, Lowfat, Plain", "Banana, Raw", "Turkey & Provolone Cheese on Kaiser Roll", "Granola", "Muffin, Double Chocolate", "Chicken Caesar Salad", "Banana, Raw", "California Roll", "Nan, Wheat", "Clam, Fried", "Greek Nonfat Yogurt, Vanilla", "Stuffed Pepper", "Falafel", "Banana, Raw", "Chewy Granola Bar, Chocolate Chunk", "Protein Instant Oatmeal, Banana Nut", "Scrambled Eggs", "Sausage", "Chocolate Chip Cookies", "Milk, 2%", "Battered Fish", "Rigatoni & Vegetables", "Home Fries", "Baked Beans", "Cheeseburger", "Sausage", "Everything Bagel", "Cinnamon Roll", "Scrambled Eggs", "Bacon", "Ham", "Broccoli", "Turkey", "Squash, Strained", "Mashed Potatoes", "Whey Protein Plus, Professional Strength, Triple Chocolate", "Muffin, Double Chocolate", "Turkey & Provolone Cheese on Kaiser Roll", "Yogurt, Lowfat, Plain", "Granola", "Banana, Raw", "Protein Box w/ Egg, Cheese & Muesli", "California Roll", "Falafel Wrap", "Trail Mix Granola Bar, Fruit & Nut", "Banana, Raw", "Chicken Caesar Salad", "Turkey & Pepperjack on Multigrain Bread", "Yogurt, Lowfat, Plain", "Granola", "Protein Box w/ Egg, Cheese & Muesli", "Cookies, Sandwich, Chocolate", "Chocolate Fat Free Milk", "Tiny Twists, Original", "Turkey on Rye", "Meatloaf", "Pork", "Banana, Raw", "Yogurt, Lowfat, Plain", "Granola", "BBQ Chicken Pizza", "Banana, Raw", "California Roll", "Starbucks Mocha Frappuccino", "Turkey & Pepperjack on Multigrain Bread", "Pork", "Noodles & Beef Entree", "Meatloaf", "Granola", "Turkey & Pepperjack on Multigrain Bread", "Muffin, Double Chocolate", "Yogurt, Lowfat, Plain", "Rice Cakes", "California Roll", "Burrito Bowl, Black Beans", "Greek Yogurt", "Banana, Raw", "Burrito Bowl, Black Beans", "Granola", "Vanilla Yogurt Raisins", "Yogurt, Lowfat, Plain", "Granola", "Turkey & Pepperjack on Multigrain Bread", "Chicken Caesar Salad", "Yogurt, Lowfat, Plain", "Yogurt, Lowfat, Plain", "Granola", "Turkey & Pepperjack on Multigrain Bread", "Chicken Caesar Salad", "Nutri Grain Bar, Apple Cinnamon", "Corona Extra", "Burrito Bowl, Black Beans", "Mango Drink", "Yogurt, Lowfat, Plain", "Vanilla Yogurt Raisins", "Meatball Sub", "Granola", "Popcorn", "Mango Drink", "Potsticker", "Red Bean Bun", "Greek Yogurt", "Vegetable Lo Mein", "Trail Mix", "California Roll", "Rainbow Salad", "Soft Serve Ice Cream", "Turkey & Provolone Cheese on Kaiser Roll", "Muffin, Double Chocolate", "Granola", "Yogurt, Lowfat, Plain", "Banana, Raw", "Banana, Raw", "California Roll", "Protein Box w/ Egg, Cheese & Muesli", "Greek Yogurt", "Meatloaf", "Banana, Raw", "Chewy Granola Bar, Chocolate Chunk", "Turkey & Provolone Cheese on Kaiser Roll", "Granola", "Yogurt, Lowfat, Plain", "Banana, Raw", "Protein Box w/ Egg, Cheese & Muesli", "California Roll", "Greek Yogurt", "Enchilada, Chicken", "Chewy Granola Bar, Chocolate Chunk", "Granola", "Yogurt, Lowfat, Plain", "Chicken Caesar Salad", "Turkey & Pepperjack on Multigrain Bread", "Protein Box w/ Egg, Cheese & Muesli", "Banana, Raw", "Broccoli Cuts", "Baked Bean, Homestyle", "Cheese Quesadilla", "Vanilla Ice Cream", "Potato, Baked", "Nutri-Grain Cereal Bar, Apple Cinnamon", "Granola", "Turkey & Pepperjack on Multigrain Bread", "Vanilla Yogurt Raisins", "Banana, Raw", "Turkey & Pepperjack on Multigrain Bread", "Protein Box w/ Egg, Cheese & Muesli", "Banana, Raw", "Trail Mix Granola Bar, Fruit & Nut", "Lamb Curry with Rice", "Yogurt Covered Raisins", "Trail Mix Granola Bar, Fruit & Nut", "Banana, Raw", "Chicken Caesar Salad", "Pizza", "Peanut, Dried", "Greek Yogurt, Vanilla", "Potsticker", "Toast Chee, Crackers, Peanut Butter", "Toasty Crackers, Peanut Butter", "Whole Wheat Chocolate Chip Cookie", "Chicken", "Lemonade with Strawberry", "Broccoli Cuts", "Szechuan Pork", "Rice", "Vegetarian Calzone", "Zebra, Cake", "Turkey & Pepperjack on Multigrain Bread", "Banana, Raw", "Dark Mint Chocolate Chip", "California Roll", "Black Bean Patty", "Coffee Ice Cream", "Carrots", "Whey Protein Plus, Professional Strength, Triple Chocolate", "2% Reduced Fat Milk, Dairy Pure", "Milk Chocolate, Creamy Chocolate", "Granola Bars, Oats 'n Honey", "Danish Pastry, Fruit", "Yogurt, Lowfat, Plain", "Granola", "Turkey & Provolone Cheese on Kaiser Roll", "Protein Bar, Cookies N Cream", "Burrito Bowl, Black Beans", "Chicken Tenders", "Almonds & Sea Salt in Dark Chocolate, 55% Cocoa", "Dairy Pure Milk, 2% Reduced Fat", "Almonds & Sea Salt in Dark Chocolate, 55% Cocoa", "Turkey & Provolone Cheese on Kaiser Roll", "Nutri Grain Bar, Apple Cinnamon", "Chewy Granola Bar, Chocolate Chunk", "Turkey & Pepperjack on Multigrain Bread", "Burrito Bowl, Chicken", "Home Fries", "Bacon", "Scrambled Eggs", "California Roll", "Protein Box w/ Egg, Cheese & Muesli", "Muffin, Double Chocolate", "Chocolate Ice Cream", "Burrito, Bean & Meat"], 211872], "type": "pie", "uid": "a5005975-56bc-4829-ab02-37d7e48d86ba"}], {"annotations": [{"font": {"size": 20}, "showarrow": false, "text": "Stress", "x": 0.2, "y": 0.5}, {"font": {"size": 20}, "showarrow": false, "text": "Non-stress", "x": 0.83, "y": 0.5}], "title": "Caloric Intake on Stressful vs Non-stressful Days"}, {"showLink": true, "linkText": "Export to plot.ly"});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("1d7329e6-94e2-4af9-96d6-6eccf6b55822"));});</script>


## Wordcloud of Foods Eaten on Stressful vs Non-stressful Days


```python
stress_food_str = ' '.join([' '.join(foods_for_day) for foods_for_day in stress_food_logs['foods_eaten'].values])
nonstress_food_str = ' '.join([' '.join(foods_for_day) for foods_for_day in nonstress_food_logs['foods_eaten'].values])
```


```python
# prep text
stress_wordcloud = custom_utils.generate_wordcloud(stress_food_str)
nonstress_wordcloud = custom_utils.generate_wordcloud(nonstress_food_str)

# display wordclouds using matplotlib
f, axes = plt.subplots(1, 2, sharex=True)
f.set_size_inches(18, 10)
axes[0].imshow(stress_wordcloud, interpolation="bilinear")
axes[0].set_title('Stressed', fontsize=36)
axes[0].axis("off")
axes[1].imshow(nonstress_wordcloud, interpolation="bilinear")
axes[1].set_title('Not Stressed', fontsize=36)
axes[1].axis("off")
```




    (-0.5, 399.5, 199.5, -0.5)




![png](/img/blog/2019-06-12-stress-tracking-pt2/output_47_1.png)


## Pie Chart Top 10 Website Usage on Stressful vs Non-stressful Days


```python
webtracker_df = pd.read_csv('data/webtime-tracker.csv', index_col=0).transpose()
webtracker_df.head()
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
      <th>Domain</th>
      <th>0.0.0.0</th>
      <th>1.1.1.1</th>
      <th>127.0.0.1</th>
      <th>192.168.1.1</th>
      <th>192.168.123.1</th>
      <th>2.bp.blogspot.com</th>
      <th>2ality.com</th>
      <th>326webprojectslack.slack.com</th>
      <th>4.bp.blogspot.com</th>
      <th>66.media.tumblr.com</th>
      <th>...</th>
      <th>www8.garmin.com</th>
      <th>xarray.pydata.org</th>
      <th>yalebooks.yale.edu</th>
      <th>yourbasic.org</th>
      <th>youthful-sage.glitch.me</th>
      <th>youtu.be</th>
      <th>yutsumura.com</th>
      <th>z-table.com</th>
      <th>zellwk.com</th>
      <th>zulko.github.io</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-11-22</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-11-22.1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-11-23</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-11-24</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2018-11-25</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1183 columns</p>
</div>




```python
stressful_date_strs = list(stressful_dates.apply(lambda x: x.strftime('%Y-%m-%d')))
nonstressful_date_strs = list(nonstressful_dates.apply(lambda x: x.strftime('%Y-%m-%d')))

stress_domains = webtracker_df.loc[stressful_date_strs].sum(axis=0).sort_values(ascending=False)
nonstress_domains = webtracker_df.loc[nonstressful_date_strs].sum(axis=0).sort_values(ascending=False)
```


```python
fig = {
    "data": [
        {
            "labels": list(stress_domains[:10].keys()) + ['Other'],
            "values": list(stress_domains[:10].values) + [stress_domains[10:].sum()],
            "domain": {"x": [0, .48]},
            "name": "Stressful Surfing",
            "hoverinfo":"label+percent",
            "hole": .4,
            "type": "pie"
        },
        {
            "labels": list(nonstress_domains[:10].keys()) + ['Other'],
            "values": list(nonstress_domains[:10].values) + [nonstress_domains[10:].sum()],
            "domain": {"x": [.52, 1]},
            "name": "Non-stressful Surfing",
            "hoverinfo":"label+percent",
            "hole": .4,
            "type": "pie"
        }
    ],
    "layout": {
        "title": "Website Usage on Stressful vs Non-stressful Days",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Stress",
                "x": 0.19,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Non-stress",
                "x": 0.85,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')
```


<div id="cb1d792d-d664-466b-9e62-af65977eba54" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("cb1d792d-d664-466b-9e62-af65977eba54", [{"domain": {"x": [0, 0.48]}, "hole": 0.4, "hoverinfo": "label+percent", "labels": ["www.overleaf.com", "www.youtube.com", "mail.google.com", "docs.google.com", "www.messenger.com", "drive.google.com", "github.com", "app.asana.com", "localhost", "www.google.com", "Other"], "name": "Stressful Surfing", "values": [14675, 11896, 5237, 4789, 4560, 4061, 3383, 2780, 2625, 2247, 28210], "type": "pie", "uid": "1f9b0ba1-9ff4-49d3-be21-f6b769f58586"}, {"domain": {"x": [0.52, 1]}, "hole": 0.4, "hoverinfo": "label+percent", "labels": ["www.overleaf.com", "www.youtube.com", "localhost", "mail.google.com", "docs.google.com", "www.messenger.com", "drive.google.com", "app.asana.com", "www.google.com", "www.facebook.com", "Other"], "name": "Non-stressful Surfing", "values": [181447, 119467, 53152, 48376, 44922, 34524, 21034, 18845, 18619, 18497, 252760], "type": "pie", "uid": "2c97a7b8-e38c-428f-bfeb-c0a613ac2ed3"}], {"annotations": [{"font": {"size": 20}, "showarrow": false, "text": "Stress", "x": 0.19, "y": 0.5}, {"font": {"size": 20}, "showarrow": false, "text": "Non-stress", "x": 0.85, "y": 0.5}], "title": "Website Usage on Stressful vs Non-stressful Days"}, {"showLink": true, "linkText": "Export to plot.ly"});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("cb1d792d-d664-466b-9e62-af65977eba54"));});</script>


## Timeseries Stress Level vs Minutes Sedentary


```python
sed_ts = fitbit_client.time_series('activities/minutesSedentary', 
                                    base_date=START_DATE.strftime('%Y-%m-%d'), 
                                    end_date=END_DATE.strftime('%Y-%m-%d'))

sed_data = []
for row in sed_ts['activities-minutesSedentary']:
    sed_data.append({
        'date': row['dateTime'],
        'rhr': row['value']
    })
sed_df = pd.DataFrame(sed_data)
```


```python
sedentary_trace = go.Scatter(
    x=sed_df.date,
    y=sed_df.rhr,
    name='Minutes Sedentary',
    yaxis='y2',
    fill='tozeroy',
)
data = [stress_trace, sedentary_trace]
layout = go.Layout(
    title='Stress Level vs Minutes Sedentary',
    yaxis=dict(
        title='Stress Level',
        range=[0, 8]
    ),
    yaxis2=dict(
        title='Minutes Sedentary',
        overlaying='y',
        side='right',s
        range=[0, 1600]
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='stress-vs-min-sedentary')
```


<div id="436dfc8c-d69c-41eb-af39-699fa1bc5cd7" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">Plotly.newPlot("436dfc8c-d69c-41eb-af39-699fa1bc5cd7", [{"fill": "tozeroy", "name": "Stress Level", "x": ["2019-01-22", "2019-01-23", "2019-01-24", "2019-01-25", "2019-01-26", "2019-01-27", "2019-01-28", "2019-01-29", "2019-01-30", "2019-02-01", "2019-02-03", "2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08", "2019-02-09", "2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15", "2019-02-16", "2019-02-17", "2019-02-17", "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22", "2019-02-23", "2019-02-24", "2019-02-25", "2019-02-26", "2019-02-27", "2019-02-28", "2019-03-01", "2019-03-02", "2019-03-03", "2019-03-04", "2019-03-05", "2019-03-06", "2019-03-09", "2019-03-10", "2019-03-11", "2019-03-12", "2019-03-13", "2019-03-14", "2019-03-15", "2019-03-16", "2019-03-17", "2019-03-18", "2019-03-19", "2019-03-20", "2019-03-21", "2019-03-22", "2019-03-23", "2019-03-24", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28", "2019-03-29", "2019-03-30", "2019-03-31", "2019-04-01", "2019-04-02", "2019-04-03", "2019-04-04", "2019-04-05", "2019-04-06", "2019-04-07", "2019-04-08", "2019-04-09", "2019-04-10", "2019-04-11", "2019-04-12", "2019-04-13", "2019-04-14", "2019-04-15", "2019-04-16", "2019-04-18", "2019-04-20", "2019-04-21", "2019-04-22", "2019-04-23", "2019-04-24", "2019-04-25", "2019-04-26", "2019-04-27", "2019-04-29", "2019-04-30", "2019-05-01", "2019-05-02", "2019-05-03", "2019-05-04", "2019-05-05", "2019-05-06", "2019-05-07", "2019-05-09"], "y": [2, 3, 3, 2, 2, 3, 2, 2, 2, 2, 2, 3, 4, 2, 2, 2, 2, 4, 3, 3, 2, 3, 3, 3, 2, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 4, 3, 3, 4, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 2, 3, 3, 2, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3], "type": "scatter", "uid": "8614efc1-5674-41c3-9c00-7a26a02e5d9c"}, {"fill": "tozeroy", "name": "Minutes Sedentary", "x": ["2019-01-22", "2019-01-23", "2019-01-24", "2019-01-25", "2019-01-26", "2019-01-27", "2019-01-28", "2019-01-29", "2019-01-30", "2019-01-31", "2019-02-01", "2019-02-02", "2019-02-03", "2019-02-04", "2019-02-05", "2019-02-06", "2019-02-07", "2019-02-08", "2019-02-09", "2019-02-10", "2019-02-11", "2019-02-12", "2019-02-13", "2019-02-14", "2019-02-15", "2019-02-16", "2019-02-17", "2019-02-18", "2019-02-19", "2019-02-20", "2019-02-21", "2019-02-22", "2019-02-23", "2019-02-24", "2019-02-25", "2019-02-26", "2019-02-27", "2019-02-28", "2019-03-01", "2019-03-02", "2019-03-03", "2019-03-04", "2019-03-05", "2019-03-06", "2019-03-07", "2019-03-08", "2019-03-09", "2019-03-10", "2019-03-11", "2019-03-12", "2019-03-13", "2019-03-14", "2019-03-15", "2019-03-16", "2019-03-17", "2019-03-18", "2019-03-19", "2019-03-20", "2019-03-21", "2019-03-22", "2019-03-23", "2019-03-24", "2019-03-25", "2019-03-26", "2019-03-27", "2019-03-28", "2019-03-29", "2019-03-30", "2019-03-31", "2019-04-01", "2019-04-02", "2019-04-03", "2019-04-04", "2019-04-05", "2019-04-06", "2019-04-07", "2019-04-08", "2019-04-09", "2019-04-10", "2019-04-11", "2019-04-12", "2019-04-13", "2019-04-14", "2019-04-15", "2019-04-16", "2019-04-17", "2019-04-18", "2019-04-19", "2019-04-20", "2019-04-21", "2019-04-22", "2019-04-23", "2019-04-24", "2019-04-25", "2019-04-26", "2019-04-27", "2019-04-28", "2019-04-29", "2019-04-30", "2019-05-01", "2019-05-02", "2019-05-03", "2019-05-04", "2019-05-05", "2019-05-06", "2019-05-07", "2019-05-08", "2019-05-09", "2019-05-10"], "y": ["692", "512", "632", "424", "533", "660", "697", "719", "659", "597", "636", "559", "569", "721", "619", "680", "550", "622", "560", "567", "604", "854", "769", "680", "569", "602", "698", "676", "637", "642", "647", "526", "456", "733", "588", "683", "617", "686", "533", "656", "590", "651", "705", "661", "666", "617", "677", "702", "1078", "343", "540", "427", "435", "539", "1113", "1069", "587", "725", "595", "581", "653", "541", "758", "601", "623", "679", "670", "424", "745", "681", "748", "627", "698", "552", "455", "669", "579", "592", "675", "691", "412", "762", "595", "680", "751", "660", "690", "516", "792", "527", "676", "607", "780", "638", "485", "612", "670", "641", "684", "697", "684", "656", "547", "495", "689", "703", "435", "734", "630"], "yaxis": "y2", "type": "scatter", "uid": "aaf93880-642d-413f-993d-b33ff4f47233"}], {"title": "Stress Level vs Minutes Sedentary", "yaxis": {"range": [0, 8], "title": "Stress Level"}, "yaxis2": {"overlaying": "y", "range": [0, 1600], "side": "right", "title": "Minutes Sedentary"}}, {"showLink": true, "linkText": "Export to plot.ly"});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("436dfc8c-d69c-41eb-af39-699fa1bc5cd7"));});</script>

