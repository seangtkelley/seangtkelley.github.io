---
layout: post
title: "Story Discovery Using K-Means Clustering on News Articles"
desc: "Now, perhaps more than ever, journalism has become instrumental in allowing us to understand current events. We rely on organizations to bring us the truth about the state of our country and world. However, with the rise of clickbait and fake news in this so-called 'post-truth' era, it's critical that our news diet is accurate and that we are not stuck in a content bubble, only consuming what the algorithms deem will generate the most revenue."
tag: "Data Analysis"
author: "Sean Kelley"
thumb: "/img/blog/2018-01-03-news-article-clustering/news-banner.jpg"
date: 2018-01-03
---

Now, perhaps more than ever, journalism has become instrumental in allowing us to understand current events. We rely on organizations to bring us the truth about the state of our country and world. However, with the rise of clickbait and fake news in this so-called "post-truth" era, it's critical that our news diet is accurate and that we are not stuck in a content bubble, only consuming what the algorithms deem will generate the most revenue.

In this post, I attempt to present a method for aggregating and clustering news articles from different news sources. Hopefully, by finding articles covering the same story from different sources, the true nature of the event can be discovered and bias mitigated.

I was inspired to explore this idea from the great work done by Brandon Rose in his article [Document Clustering with Python](http://brandonrose.org/clustering). Although I understand the clustering setup process, he does a much better job in explaining the parts and I highly recommend you read that article if you want a more in-depth look at document clustering.

## Gathering Articles from Sources

First, let's start by gathering information about our sources. Our goal will be to eventually extract current and past articles from the site in a reusable and efficient way. We could web scrape each site for the current stories, but this process would be extremely hard to effectively code given the HTML structure of each news source is vastly different. 

So let's think. Is there a way to find currently published articles by a news source that would lend itself to efficient, reusable code? As it turns out, there is! We can take advantage of the fact that almost all news sources have active RSS feeds broadcasting the most recent articles from a range of topics. 

To further aid us, Kurt McKee and Mark Pilgrim put together [feedparser](https://pypi.python.org/pypi/feedparser), a library specifically for extracting the information from an RSS feed. Let's start by testing this using Reuters.


```python
import feedparser

rss_url = "http://feeds.reuters.com/Reuters/PoliticsNews"

feed = feedparser.parse(rss_url)

feed['entries'][0]['link']
```




    'http://feeds.reuters.com/~r/Reuters/PoliticsNews/~3/ADO0oDYpJeo/senate-swears-in-democrats-from-alabama-minnesota-idUSKBN1ES1J0'



Fantastic! The RSS feeds provide a plethora of information but as we want to cluster articles by story, we will focus on gathering the links to the articles.

But how might we gather news from yesterday, last month, or even a year ago? For this we can turn to the incredible resource that is [archive.org](https://archive.org). Their site provides an endpoint that allows you to query their website archives by date, which is exactly what we need.


```python
from datetime import datetime, timedelta

# create datetime object from 30 days ago
delta = timedelta(days=30)
current_date = datetime.utcnow()
query_date = current_time - delta
query_date_str = query_date.strftime("%Y%m%d")

# base web archive url
archive_url = "https://web.archive.org/web/"

# build url to query archived rss feed
query_url = archive_url + query_date_str + "/" + rss_url

historic_feed = feedparser.parse(query_url)

historic_feed['entries'][0]['link']
```




    'http://feeds.reuters.com/~r/Reuters/PoliticsNews/~3/WqQkLE_rMp8/supreme-court-lets-trumps-latest-travel-ban-go-into-full-effect-idUSKBN1DY2NY'



Now that we have a way to get current and past articles from a news source using their RSS feed, let's expand this to more sources. In this post, I'm going to use [CNN](http://www.cnn.com), [Fox](http://www.foxnews.com), [The Washington Post](https://www.washingtonpost.com), [The New York Times](https://www.nytimes.com), and [Reuters](https://www.reuters.com).

We also need to get the raw text of the articles. For this we can use the amazing [newspaper](https://github.com/codelucas/newspaper) library created by [@codelucas](https://github.com/codelucas). The library has tons of options for streamlined web scraping of articles but we will just use it to grab the raw text.


```python
import pandas as pd
import newspaper

rss_urls = ["http://rss.cnn.com/rss/cnn_allpolitics.rss", "http://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
            "http://feeds.washingtonpost.com/rss/politics", "http://feeds.foxnews.com/foxnews/politics",
            "http://feeds.reuters.com/Reuters/PoliticsNews"]

articles = []
for rss_url in rss_urls:
    feed = feedparser.parse(rss_url)
    
    # find all article links
    article_links = []
    for entry in feed['entries']:
        article_links.append(entry['link'])
    
    # get text for each article
    for link in article_links:
        article = newspaper.Article(link)
        
        article.download()
        article.parse()

        articles.append({
            "url": article.url,
            "title": article.title,
            "text": article.text
        })

articles = pd.DataFrame(articles)
```


```python
articles.head()
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
      <th>text</th>
      <th>title</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Washington (CNN) President Donald Trump excori...</td>
      <td>Trump unloads on former top aide Bannon</td>
      <td>http://rss.cnn.com/~r/rss/cnn_allpolitics/~3/h...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The idea that a meeting between the three top ...</td>
      <td>Steve Bannon is 100% right about Russia and th...</td>
      <td>http://rss.cnn.com/~r/rss/cnn_allpolitics/~3/Z...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Story highlights Tina Smith replaces Sen. Al F...</td>
      <td>Two senators sworn into office amid #MeToo mov...</td>
      <td>http://rss.cnn.com/~r/rss/cnn_allpolitics/~3/l...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(CNN) The White House on Wednesday released a ...</td>
      <td>Read Trump's official statement about Steve Ba...</td>
      <td>http://rss.cnn.com/~r/rss/cnn_allpolitics/~3/Y...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Story highlights The book, "Fire and Fury: Ins...</td>
      <td>Bannon: 2016 Trump Tower meeting was 'treasonous'</td>
      <td>http://rss.cnn.com/~r/rss/cnn_allpolitics/~3/L...</td>
    </tr>
  </tbody>
</table>
</div>



For different dates, simply change the url passed to feed parser to include the archive.org base url and the date for which you want to query.

## Article Clustering

With articles now in hand, we can start clustering. Again, as mentioned above, check out the [wonderful post](http://brandonrose.org/clustering) by Brandon Rose on document cluster for a better insight into the steps I'm taking to cluster the articles.

First, let's start by appending contractions to the current list of stopwords.


```python
from sklearn.feature_extraction import text

eng_contractions = ["ain't", "amn't", "aren't", "can't", "could've", "couldn't",
                    "daresn't", "didn't", "doesn't", "don't", "gonna", "gotta", 
                    "hadn't", "hasn't", "haven't", "he'd", "he'll", "he's", "how'd",
                    "how'll", "how's", "I'd", "I'll", "I'm", "I've", "isn't", "it'd",
                    "it'll", "it's", "let's", "mayn't", "may've", "mightn't", 
                    "might've", "mustn't", "must've", "needn't", "o'clock", "ol'",
                    "oughtn't", "shan't", "she'd", "she'll", "she's", "should've",
                    "shouldn't", "somebody's", "someone's", "something's", "that'll",
                    "that're", "that's", "that'd", "there'd", "there're", "there's", 
                    "these're", "they'd", "they'll", "they're", "they've", "this's",
                    "those're", "tis", "twas", "twasn't", "wasn't", "we'd", "we'd've",
                    "we'll", "we're", "we've", "weren't", "what'd", "what'll", 
                    "what're", "what's", "what've", "when's", "where'd", "where're",
                    "where's", "where've", "which's", "who'd", "who'd've", "who'll",
                    "who're", "who's", "who've", "why'd", "why're", "why's", "won't",
                    "would've", "wouldn't", "y'all", "you'd", "you'll", "you're", 
                    "you've", "'s", "s"
                     ]

nltk.download('stopwords')
nltk.download('punkt')

custom_stopwords = text.ENGLISH_STOP_WORDS.union(eng_contractions)
```

    [nltk_data] Downloading package stopwords to /home/sean/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to /home/sean/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!


Next, we import the SnowballStemmer to stem words. By stemming words, we break them down into their roots, hopefully mitigating ambiguity introduced by alternate forms of words.


```python
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
```

Here, I've combined Rose's two tokenizing and stemming functions into one. Providing the option to not stem the words is important given that after clustering, we will want to convert the stems back to their full words.


```python
import nltk
import re

def tokenize_and_stem(text, do_stem=True):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
            
    # stem filtered tokens
    stems = [stemmer.stem(t) for t in filtered_tokens]
    
    if do_stem:
        return stems
    else:
        return filtered_tokens
```

Using that function, we can create two vocabulary lists: one stemmed and one only tokenized.


```python
# not super pythonic, no, not at all.
# use extend so it's a big flat list of vocab
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in articles['text']:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)
    
    allwords_tokenized = tokenize_and_stem(i, False)
    totalvocab_tokenized.extend(allwords_tokenized)
```

From the vocabulary lists, we can make what is a essentially a lookup table for stems to full word. However, since stems can refer to multiple words, this lookup table will only return the first viable word.


```python
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
vocab_frame.head()
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
      <th>words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>washington</th>
      <td>washington</td>
    </tr>
    <tr>
      <th>cnn</th>
      <td>cnn</td>
    </tr>
    <tr>
      <th>presid</th>
      <td>president</td>
    </tr>
    <tr>
      <th>donald</th>
      <td>donald</td>
    </tr>
    <tr>
      <th>trump</th>
      <td>trump</td>
    </tr>
  </tbody>
</table>
</div>



Next, we can implement a TF-IDF Vectorizer algorithm on the article texts.


```python
from sklearn.feature_extraction.text import TfidfVectorizer

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words=custom_stopwords,
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(articles['text']) #fit the vectorizer to synopses

print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()
```

    (85, 282)


Finally, we can using K-Means clustering to cluster the articles. Here is the description that Sci-Kit Learn gives for the K-Means algorithm:

> The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares.

More information can be found [here](https://en.wikipedia.org/wiki/K-means_clustering) and [here](http://scikit-learn.org/stable/modules/clustering.html)

In my implementation, instead of hard-coding a predetermined number of clusters, I chose to use a slightly modified version of the best method I could find of estimating the amount of clusters that will appear in your dataset.


```python
from sklearn.cluster import KMeans
import math

num_clusters = int(math.sqrt(articles.shape[0] / 2) * 1.5)

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=9, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=None, tol=0.0001, verbose=0)



Now we can associate each article with its respective cluster.


```python
clusters = km.labels_.tolist()

articles['cluster'] = clusters
articles.head()
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
      <th>text</th>
      <th>title</th>
      <th>url</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Washington (CNN) President Donald Trump excori...</td>
      <td>Trump unloads on former top aide Bannon</td>
      <td>http://rss.cnn.com/~r/rss/cnn_allpolitics/~3/h...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The idea that a meeting between the three top ...</td>
      <td>Steve Bannon is 100% right about Russia and th...</td>
      <td>http://rss.cnn.com/~r/rss/cnn_allpolitics/~3/Z...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Story highlights Tina Smith replaces Sen. Al F...</td>
      <td>Two senators sworn into office amid #MeToo mov...</td>
      <td>http://rss.cnn.com/~r/rss/cnn_allpolitics/~3/l...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(CNN) The White House on Wednesday released a ...</td>
      <td>Read Trump's official statement about Steve Ba...</td>
      <td>http://rss.cnn.com/~r/rss/cnn_allpolitics/~3/Y...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Story highlights The book, "Fire and Fury: Ins...</td>
      <td>Bannon: 2016 Trump Tower meeting was 'treasonous'</td>
      <td>http://rss.cnn.com/~r/rss/cnn_allpolitics/~3/L...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Finally, we can print out the clusters and the titles of their associate articles.


```python
print("Top terms per cluster:")
print()

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0], end=',')
    print() #add whitespace
    print() #add whitespace
    
    print("Cluster %d titles:" % i, end='')
    print()
    for title in articles[articles['cluster'] == i]['title'].values.tolist():
        print(' - %s' % title)
    print() #add whitespace
    print() #add whitespace
    
print()
print()
```

    Top terms per cluster:
    
    Cluster 0 words: white, white, house, meeting, campaign, only,
    
    Cluster 0 titles:
     - Trump unloads on former top aide Bannon
     - Steve Bannon is 100% right about Russia and the Trump campaign
     - Read Trump's official statement about Steve Bannon
     - Bannon: 2016 Trump Tower meeting was 'treasonous'
     - Trump UN pick who praised Milo Yiannopoulos among nominations sent back to White House
     - Donald Trump's huge golf hypocrisy
     - DACA talks hinge on Trump
     - Trump slams Bannon: ‘When he was fired, he not only lost his job, he lost his mind’
     - ‘He lost his mind’: Trump’s excommunication of Steve Bannon, annotated
     - New Trump book: Bannon’s ‘treasonous’ claim, Ivanka’s presidential ambitions and Melania’s first-lady concerns
     - Trump slams Bannon after criticism, says ex-chief strategist 'lost his mind'
     - Trump blasts former aide Bannon after 'treasonous' comments
    
    
    Cluster 1 words: republican, senate, democrats, moore, election, seat,
    
    Cluster 1 titles:
     - Two senators sworn into office amid #MeToo movement
     - Three vice presidents at the Senate on Wednesday
     - Roy Moore's campaign manager to run for Congress
     - There is a wave of Republicans leaving Congress
     - Doug Jones joins the Senate: How will he vote?
     - 10 Senate seats most likely to flip in 2018
     - Democrats Outline Demands as Threat of Shutdown Looms
     - Doug Jones is sworn in, shrinking GOP Senate majority
     - Meet Roy Moore’s Jewish attorney. He campaigned for his friend, Doug Jones.
     - Judges deny Democrats' request to undo recount decision
     - Doug Jones, new Alabama senator: What to know
     - Tina Smith selected to replace Al Franken in the Senate: Who is she?
     - Democrats Doug Jones and Tina Smith sworn in as senators
     - With Senate up for grabs, control may come down to handful of races
     - Senate swears in Democrats from Alabama, Minnesota
     - Virginia court rejects Democrat's bid in tied state House race: media
     - Republican Senator Hatch to retire, opening door for bid by Romney
     - U.S. House transportation panel chair says he will not seek re-election
     - As U.S. budget fight looms, Republicans flip their fiscal script
    
    
    Cluster 2 words: nuclear, north, kim, powerful, war, launch,
    
    Cluster 2 titles:
     - How the nuclear football actually works
     - There's no such thing as a "nuclear button"
     - The nuclear war tweet heard 'round the world
     - Donald Trump's nuclear button is way bigger than yours
     - The ‘Nuclear Button’ Explained: For Starters, There’s No Button
     - Trump’s big-stick approach to North Korea suddenly becomes extremely literal
     - Why nuclear war with North Korea is less likely than you think
     - Trump to North Korean leader Kim: My ‘Nuclear Button’ is ‘much bigger & more powerful’
    
    
    Cluster 3 words: u.s., jan., administration, federal, ways, decision,
    
    Cluster 3 titles:
     - Trump administration eases penalties against negligent nursing homes
     - The Energy 202: January may be a make-or-break month for the U.S. solar business
     - American Action Network: 'Thank You: Rep. Paulsen' | Campaign 2018
     - Lawyer claims donations fall far short of high costs of defamation suit against Trump
     - Trump threatens to cut off U.S. aid to Palestinians over Jerusalem row
     - Senior U.S. refugee official to retire this month
     - FEMA allows churches to apply retroactively for disaster aid
    
    
    Cluster 4 words: senate, republican, committee, retired, allies, tax,
    
    Cluster 4 titles:
     - Fusion GPS co-founders slam GOP's 'fake investigations'
     - Romney could become Trump's new Washington foe
     - Trump begged Orrin Hatch to run again. The senator retired anyway.
     - Republican Sen. Orrin Hatch of Utah will retire, opening door for a Romney candidacy
     - Trump's fixation with size
     - This is what Orrin G. Hatch’s retirement means for the Senate
     - The Health 202: Hatch's retirement means the Senate could get even less bipartisan on health care
     - The Finance 202: Hatch exit sets up race for Senate Finance Committee gavel
     - Republicans passed their tax bill. Now they’re spending $10 million to promote it.
     - Republican Sen. Orrin Hatch of Utah will retire, opening door for a Romney candidacy
     - Trump takes hard line on ‘dreamers,’ but remains interested in a deal
     - With Hatch’s retirement, Trump is losing an ally — and might be gaining a foe
     - Chuck Grassley pushes Fusion GPS founders to testify in public
    
    
    Cluster 5 words: email, department, clinton, reporters, states, case,
    
    Cluster 5 titles:
     - Trump again at war with 'deep state' Justice Department
     - Trump Accuses Former Clinton Aide of Failing to Follow Security Protocols
     - Fact-checking President Trump’s post-New Year’s tweets
     - FBI exonerated Clinton before getting key evidence, report says
     - 93-year-old World War II vet sworn in as mayor in New Jersey town
    
    
    Cluster 6 words: mr., united, united, states, north, please,
    
    Cluster 6 titles:
     - Trump Says His ‘Nuclear Button’ Is ‘Much Bigger’ Than North Korea’s
     - Trump Says Bannon Has ‘Lost His Mind’ After Bannon Insults Donald Trump Jr.
     - Trump’s First Big Twitter Day of 2018: Analyzing Nuclear Buttons and the ‘Corrupt Media’
     - Trump Reiterates Support for Iranian Protesters, but Also Criticizes Obama
     - Showtime’s ‘The Circus’ Will Go On Without Mark Halperin
     - Trump’s Aviation Boast Fails to Get in the Air
    
    
    Cluster 7 words: tweeted, trump, n't, years, twitter, aides,
    
    Cluster 7 titles:
     - Palestinian officials slam Trump's threat to cut US aid
     - Trump's 16-tweet Tuesday is the story of his presidency
     - Trump lawyers talked with special counsel team
     - Why Anthony Scaramucci returning to the White House makes perfect sense
     - 10 of the most questionable things Trump's claimed credit for
     - Trump threatens aid to Palestinians, appears to contradict himself on Jerusalem
     - Social Media Shudders After Trump Mocks North Korea’s ‘Button’
     - U.S. Service Member Is Killed in Afghan Province
     - The Daily 202: North Korea my-button-is-bigger brinkmanship again spotlights Trump’s fixation on size
     - Trump says Iranian protests will see support from US 'at the appropriate time'
     - Trump draws rebukes after touting aviation safety record
     - Trump says U.S. has gotten 'nothing' from Pakistan aid
    
    
    Cluster 8 words: program, action, obama, continued, security, took,
    
    Cluster 8 titles:
     - US says 2 terrorists killed in Somalia airstrike
     - Ex-Homeland Security Officials Urge Faster Action on DACA
     - Former DHS secretaries Chertoff, Napolitano and Johnson warn Congress over ‘dreamers’ deadline
    
    
    
    


## Conclusion

Looks like we generated some pretty interesting clusters! As you can see, some are better than others but, solely based on the titles, K-Means seems to have separated articles into relatively coherent clusters.

I would definitely like to explore more methods of creating better clusters:

- remove outliers from clusters
- more data specific method of determining amount of clusters
- test other clustering algorithms

Finally, if anyone is interested, I have created a simple web app that updates hourly with news clusters from the current day, last week, and last month.

## [News Clustering Web App](http://radical-raccoon.herokuapp.com/)

Like always, corrections, suggestions or comments are always welcome!
