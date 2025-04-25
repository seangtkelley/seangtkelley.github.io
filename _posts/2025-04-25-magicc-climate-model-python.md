---
layout: post
title: "Climate Model from your Laptop with MAGICC7 and Pymagicc"
desc: "Until recently, I thought that all climate models were incredible complex and only run on massive supercomputers. However, through my Master's, I discovered many modern models can be run on everyday computer hardware."
tag: "Climate"
author: "Sean Kelley"
thumb: "/img/blog/2025-04-25-magicc-climate-model-python/magicc-python-post_14_1.png"
date: 2025-04-25
---

Until recently, I thought that all climate models were incredible complex and only run on massive supercomputers. However, through my Master's, I became accustomed to the fact that many models, [especially early models](https://github.com/seangtkelley/hansen-et-al-1981-climate-model), are quite simple. This might be initially surprising, but when combined with the fact that the science behind climate change dates back over 100 years, it seems clear that simpler models could still be powerful.

# MAGICC

One of the most common simplified models is called MAGICC. The acronym stands for "**M**odel for the **A**ssessment of **G**reenhouse Gas **I**nduced **C**limate **C**hange" and focuses on modeling the carbon cycle. It is commonly used by IPCC, other decision makers, and scientists when they need to quickly but accurately create climate projections.

![](/img/blog/2025-04-25-magicc-climate-model-python/_images_MAGICCLOGOv7_onlyearth2_400png.png)

*Logo image used by MAGICC*

More specific information about the model can be found on the [website](https://magicc.org/) and [wiki](http://wiki.magicc.org/index.php?title=Model_Description). As of this blog post, binaries of the current version, MAGICC7, are available for download [via a form](https://magicc.org/download/magicc7) on the site. The previous version, MAGICC6, is also available but [via the wiki](http://wiki.magicc.org/index.php?title=Download_MAGICC6). Once downloaded, you will need to extract the contents and **ensure the binaries have executable permissions**.

# Pymagicc

However, being a avid python user, I immediately went to see if there was a port. Thankfully, I was greeted by [pymagicc](https://github.com/openscm/pymagicc). The team at [OpenSCM](https://github.com/openscm) graciously wrote a python wrapper for the binary. However, there are a few caveats. The python package by default uses a Windows version of MAGICC6 as that was previously the only binary released by the MAGICC team. This means that on any other operating system, you are forced to use an emulator like [wine](https://www.winehq.org/). While wine is very powerful, I wanted to use the macOS-native MAGICC7 binary.

Once again, the OpenSCM team has gone above and beyond by already creating a [tutorial notebook](https://github.com/openscm/pymagicc/blob/master/notebooks/MAGICC7.ipynb) for MAGICC7. In short, we can provide the binary path directly to pymagicc and it will use that instead of the built-in version.

# Basic Usage

I'm only a beginner with using this model, but here I will demonstrate how easy it is to use the model to make projections based on different emission scenarios [RCPs](https://en.wikipedia.org/wiki/Representative_Concentration_Pathway). Let's begin by installing the required pip packages.


```python
# !pip install matplotlib scmdata pymagicc
```

Now we can import pymagicc, along with some data and plotting libraries.


```python
import matplotlib.pyplot as plt

import pymagicc
import scmdata
from pymagicc import rcps
```

    /Users/seangtkelley/miniconda3/envs/climate-model-pymagicc/lib/python3.10/site-packages/scmdata/database/_database.py:9: TqdmExperimentalWarning: Using `tqdm.autonotebook.tqdm` in notebook mode. Use `tqdm.tqdm` instead to force console mode (e.g. in jupyter console)
      import tqdm.autonotebook as tqdman


Next, we will point pymagicc to the MAGICC7 binary. We can do this using the `MAGICC_EXECUTABLE_7` environment variable. Jupyter notebooks provide an easy way to set environment variables using the `%env` command. 


```python
%env MAGICC_EXECUTABLE_7=/Users/seangtkelley/Source/blog-posts/climate-model-magicc-python/magicc-v7.5.3/bin/magicc-darwin-arm64
```

    env: MAGICC_EXECUTABLE_7=/Users/seangtkelley/Source/blog-posts/climate-model-magicc-python/magicc-v7.5.3/bin/magicc-darwin-arm64


If you encounter an error complaining about `libgfortran`, you likely also need to specify a path for the this library.

On MacOS, this will likely be be in your `gcc` install, but you can find its full path with the following command:
`find /path/to/search -name "libgfortran*.dylib"`


```python
%env DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/gcc/14.2.0_1/lib/gcc/current:$DYLD_LIBRARY_PATH
```

    env: DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/gcc/14.2.0_1/lib/gcc/current:$DYLD_LIBRARY_PATH


Finally, we can run the model! We will group the RCPs by their specific scenario and run the model on each.

**NOTE**: I have set `out_temperature=1` which is required to instruct MAGICC to output temperature data. This value (along with many other configuration values) can also be set in `MAGCFG_USER.CFG` in the `run` folder. More information can be found in the tutorial notebook and [this issue](https://github.com/openscm/pymagicc/issues/341).


```python
rcps["scenario"].unique()
```




    array(['RCP26', 'RCP45', 'RCP60', 'RCP85'], dtype=object)




```python
results = []
for scen in rcps.groupby("scenario"):
    results_scen = pymagicc.run(scen, magicc_version=7, out_temperature=1)
    results.append(results_scen)

results = scmdata.run_append(results)
```

    /Users/seangtkelley/miniconda3/envs/climate-model-pymagicc/lib/python3.10/site-packages/pymagicc/io/scen7.py:43: UserWarning: MAGICC6 RCP region naming (R5*) is not compatible with MAGICC7, automatically renaming to MAGICC7 compatible regions (R5.2*)
      warnings.warn(warn_msg)
    /Users/seangtkelley/miniconda3/envs/climate-model-pymagicc/lib/python3.10/site-packages/pymagicc/io/scen7.py:43: UserWarning: MAGICC6 RCP region naming (R5*) is not compatible with MAGICC7, automatically renaming to MAGICC7 compatible regions (R5.2*)
      warnings.warn(warn_msg)
    /Users/seangtkelley/miniconda3/envs/climate-model-pymagicc/lib/python3.10/site-packages/pymagicc/core.py:436: UserWarning: magicc logged a ERROR message. Check the 'stderr' key of the result's `metadata` attribute.
      warnings.warn(
    /Users/seangtkelley/miniconda3/envs/climate-model-pymagicc/lib/python3.10/site-packages/pymagicc/io/scen7.py:43: UserWarning: MAGICC6 RCP region naming (R5*) is not compatible with MAGICC7, automatically renaming to MAGICC7 compatible regions (R5.2*)
      warnings.warn(warn_msg)
    /Users/seangtkelley/miniconda3/envs/climate-model-pymagicc/lib/python3.10/site-packages/pymagicc/io/scen7.py:43: UserWarning: MAGICC6 RCP region naming (R5*) is not compatible with MAGICC7, automatically renaming to MAGICC7 compatible regions (R5.2*)
      warnings.warn(warn_msg)
    /Users/seangtkelley/miniconda3/envs/climate-model-pymagicc/lib/python3.10/site-packages/pymagicc/core.py:436: UserWarning: magicc logged a ERROR message. Check the 'stderr' key of the result's `metadata` attribute.
      warnings.warn(
    /Users/seangtkelley/miniconda3/envs/climate-model-pymagicc/lib/python3.10/site-packages/pymagicc/io/scen7.py:43: UserWarning: MAGICC6 RCP region naming (R5*) is not compatible with MAGICC7, automatically renaming to MAGICC7 compatible regions (R5.2*)
      warnings.warn(warn_msg)
    /Users/seangtkelley/miniconda3/envs/climate-model-pymagicc/lib/python3.10/site-packages/pymagicc/io/scen7.py:43: UserWarning: MAGICC6 RCP region naming (R5*) is not compatible with MAGICC7, automatically renaming to MAGICC7 compatible regions (R5.2*)
      warnings.warn(warn_msg)
    /Users/seangtkelley/miniconda3/envs/climate-model-pymagicc/lib/python3.10/site-packages/pymagicc/core.py:436: UserWarning: magicc logged a ERROR message. Check the 'stderr' key of the result's `metadata` attribute.
      warnings.warn(
    /Users/seangtkelley/miniconda3/envs/climate-model-pymagicc/lib/python3.10/site-packages/pymagicc/io/scen7.py:43: UserWarning: MAGICC6 RCP region naming (R5*) is not compatible with MAGICC7, automatically renaming to MAGICC7 compatible regions (R5.2*)
      warnings.warn(warn_msg)
    /Users/seangtkelley/miniconda3/envs/climate-model-pymagicc/lib/python3.10/site-packages/pymagicc/io/scen7.py:43: UserWarning: MAGICC6 RCP region naming (R5*) is not compatible with MAGICC7, automatically renaming to MAGICC7 compatible regions (R5.2*)
      warnings.warn(warn_msg)
    /Users/seangtkelley/miniconda3/envs/climate-model-pymagicc/lib/python3.10/site-packages/pymagicc/core.py:436: UserWarning: magicc logged a ERROR message. Check the 'stderr' key of the result's `metadata` attribute.
      warnings.warn(


From what I understand, the warnings here are expected since the data available in the `pymagicc` library is designed for MAGICC6 so it needs to be converted.

Either way, with the results, we can now plot the global average surface temperature relative to the pre-industrial mean.


```python
temperature_rel_to_1850_1900 = (
    results
    .filter(variable="Surface Temperature", region="World")
    .relative_to_ref_period_mean(year=range(1850, 1900 + 1))
)

temperature_rel_to_1850_1900.lineplot()
plt.title("Global Mean Temperature Projection")
plt.ylabel("°C over pre-industrial (1850-1900 mean)")
plt.show()
```

    /Users/seangtkelley/miniconda3/envs/climate-model-pymagicc/lib/python3.10/site-packages/scmdata/run.py:905: FutureWarning: Dtype inference on a pandas object (Series, Index, ExtensionArray) is deprecated. The Index constructor will keep the original dtype in the future. Call `infer_objects` on the result to get the old behavior.
      df.columns = pd.Index(columns, name="time")
    /Users/seangtkelley/miniconda3/envs/climate-model-pymagicc/lib/python3.10/site-packages/scmdata/run.py:905: FutureWarning: Dtype inference on a pandas object (Series, Index, ExtensionArray) is deprecated. The Index constructor will keep the original dtype in the future. Call `infer_objects` on the result to get the old behavior.
      df.columns = pd.Index(columns, name="time")
    /Users/seangtkelley/miniconda3/envs/climate-model-pymagicc/lib/python3.10/site-packages/scmdata/run.py:905: FutureWarning: Dtype inference on a pandas object (Series, Index, ExtensionArray) is deprecated. The Index constructor will keep the original dtype in the future. Call `infer_objects` on the result to get the old behavior.
      df.columns = pd.Index(columns, name="time")
    /Users/seangtkelley/miniconda3/envs/climate-model-pymagicc/lib/python3.10/site-packages/scmdata/plotting.py:81: FutureWarning: 
    
    The `ci` parameter is deprecated. Use `errorbar='sd'` for the same effect.
    
      ax = sns.lineplot(data=plt_df, **kwargs)



    
![png](/img/blog/2025-04-25-magicc-climate-model-python/magicc-python-post_14_1.png)
    


# Conclusion
That's it! In just a few lines of code, we have run state of the art climate model to project global temperatures through the next 75 years. MAGICC is obviously much more powerful than this basic example, but I hope it's now clear that climate models can be a lot more accessible than supercomputers and petabytes of data.

If you have climate model or python knowledge, please considering supporting pymagicc as it's completely open source and a wonderful tool for those who need efficient but accurate climate modeling in the world's most popular programming language.
