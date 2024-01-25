#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries :
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


# In[2]:


data = pd.read_csv("C:/two/computer_lab_24.csv")
data.head()


# In[3]:


data.describe()


# In[4]:


kmf = KaplanMeierFitter()
# survival times and events
#Fitting our data
# the goal is to find the number of months a computer survived before the failure. SO our event of interest will be 
#"failure" which is stored in "event" column. The first argument it takes is the timeline for our experiment

kmf.fit(durations = data["survival_time"], event_observed = data["event"])


# In[5]:


#kmf.fit(durations = data["Surv"], event_observed = data["relapse"], label="kmf.plot_survival_function(show_censors=True, \ncensor_styles={'ms': 6, 'marker': 's'})")
# kmf.fit(durations = data["survival_time"], event_observed = data["event"], label="kmf.plot_survival_function(show CI=False)")

kmf.fit(durations = data["survival_time"], event_observed = data["event"], label="kmf.plot_survival_function")
#kmf.plot_survival_function(show_censors=True, ci_show=False, censor_styles={'ms': 6, 'marker': 's'})
kmf.plot_survival_function(ci_show=False, censor_styles={'ms': 6, 'marker': 's'})
plt.ylabel("Propability of survival  S(t)")
plt.xlabel("Time t [months]")
plt.title("Kaplan Meier Plot for Survival Times of Lab Computers")


# In[6]:


kmf.survival_function_


# In[7]:


#Import Cox regression Library
from lifelines import CoxPHFitter


# In[8]:


#Parameters we want to consider while fitting our model
data = data[['survival_time','event','repairs','HDD','motherboard','power_supply','miscellaneous','OS']]


# In[9]:


#Fit Data and print summary
cph = CoxPHFitter()
cph.fit(data, "survival_time",event_col="event")
cph.print_summary()


# In[10]:


kmf.median_survival_time_


# In[11]:


duration =data["survival_time"]  #t
event_observed = data["event"]  #E
kmf2 =kmf.fit(duration, event_observed, label ="Lab Computers")
kmf2.plot_survival_function(at_risk_counts=True)
plt.tight_layout


# In[13]:


ax = plt.subplot(111)
rep = (data["repairs"] == 1)

kmf.fit(duration[rep], event_observed=event_observed[rep], label="Major Repairs")#
kmf.survival_function_.plot(ax=ax)
kmf.fit(duration[~rep], event_observed=event_observed[~rep], label="Routine Maintenance")#
kmf.survival_function_.plot(ax=ax)
plt.title("Lifespan of Computers Grouped by Maintainance Status")


# In[17]:


ax = plt.subplot(111)
rep = (data["OS"] == 1)

kmf.fit(duration[rep], event_observed=event_observed[rep], label="Issue with OS")#
kmf.survival_function_.plot(ax=ax)
kmf.fit(duration[~rep], event_observed=event_observed[~rep], label="No issue with OS")#
kmf.survival_function_.plot(ax=ax)
plt.title("Lifespan of computers by OS stability")


# In[18]:


#It is observed above that the p value of different paramenters as we know that a p-valu

#checking which factors affects the most from the group
cph.plot()
plt.ylabel("Risk Factors(covariates)")
#plt.xlabel("log(HR)")
plt.title("Factors affected the most")


# In[ ]:




