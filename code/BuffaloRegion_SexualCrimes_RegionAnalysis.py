#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ## Data Preprocessing

# In[2]:


df = pd.read_csv("/Users/kushal/Downloads/Crime_Incidents.csv", low_memory=False)


# In[3]:


df2 = df[df['incident_type_primary'].isin({'SEXUAL ABUSE', 'RAPE', 'Other Sexual Offense',
                                 'Sexual Assault'})]


# In[4]:


df3 = df2.reset_index(drop = True)
df3.head()


# In[5]:


date_times = []
for num in range(df2.shape[0]):
    date_times.append(df2['incident_datetime'].iloc[num].split(" ", 2)[0])
date_times[:5]


# In[6]:


date_times_2 = pd.to_datetime(date_times,infer_datetime_format=True)
df3['year'] = date_times_2.year
df3['month'] = date_times_2.month
df3.head(5)


# In[7]:


df4 = df[df['incident_type_primary'].isin({'ASSAULT', 'LARCENY/THEFT',  'UUV', 'BURGLARY',
        'ROBBERY', 'CRIM NEGLIGENT HOMICIDE', 'THEFT OF SERVICES',
       'MURDER', 'Breaking & Entering', 'Assault', 'Theft', 'Robbery',
       'Theft of Vehicle', 'AGG ASSAULT ON P/OFFICER',
         'Homicide',
       'AGGR ASSAULT', 'MANSLAUGHTER'})]
df4.head(5)


# ## Latitude and longitude computation

# In[8]:


df_lat_long = df3.loc[np.where(df3['location'].isna() == False)[0],].reset_index()


# In[9]:


df_lat_long.shape
df_lat_long.tail()


# In[10]:


temp = df_lat_long['location'].str.split(" ")
df_lat_long['Longitude'] = [float(temp[x][1][1:]) for x in range(temp.shape[0])]
df_lat_long['Latitude'] = [float(temp[x][2][:-1]) for x in range(temp.shape[0])]


# In[11]:


temp = df_lat_long['location'].str.split(" ")
temp.head(3)


# In[12]:


df_with_latlong = df_lat_long
df_with_latlong.head()


# ## University at buffalo south campus region

# In[13]:


df_with_latlong_filtered1_sexual= df_with_latlong.loc[np.where((df_with_latlong['Latitude'] >= 42.948) & (df_with_latlong['Latitude'] <= 42.958) & 
         (df_with_latlong['Longitude'] <= -78.7815) &
        (df_with_latlong['Longitude'] >= -78.825))]


# In[39]:


df_with_latlong_filtered1_sexual.shape


# In[38]:


df_with_latlong_filtered1_sexual.head()


# ## Airport Region

# In[15]:


df_with_latlong_filtered2_sexual= df_with_latlong.loc[np.where((df_with_latlong['Latitude'] >= 42.875) & (df_with_latlong['Latitude'] <= 42.885) & 
         (df_with_latlong['Longitude'] <= -78.873) &
        (df_with_latlong['Longitude'] >= -78.883))]


# In[41]:


df_with_latlong_filtered2_sexual.shape


# In[40]:


df_with_latlong_filtered2_sexual.head()


# ##  University at Buffalo North Campus Region

# In[17]:


df_with_latlong_filtered3_sexual= df_with_latlong.loc[np.where((df_with_latlong['Latitude'] >= 42.975) & (df_with_latlong['Latitude'] <= 43.025) & 
         (df_with_latlong['Longitude'] <= -78.765) &
        (df_with_latlong['Longitude'] >= -78.815))]


# In[43]:


df_with_latlong_filtered3_sexual.shape


# In[42]:


df_with_latlong_filtered3_sexual.head()


# ## Downtown Region Buffalo
# 

# In[19]:


df_with_latlong_filtered4_sexual= df_with_latlong.loc[np.where((df_with_latlong['Latitude'] >= 42.881) & (df_with_latlong['Latitude'] <= 42.891) & 
         (df_with_latlong['Longitude'] <= -78.870) &
        (df_with_latlong['Longitude'] >= -78.880))]


# In[20]:


df_with_latlong_filtered4_sexual.shape


# In[21]:


df_with_latlong_filtered4_sexual.head()


# Here we observed downtown region in buffalo has the maximum number of sexual crimes. Second highest region is the airport region. Next is the south campus region where sexual crimes rate is average. North campus is the safest place observed.

# In[22]:


df_with_latlong.head()


# In[23]:


df_with_latlong_new= df_with_latlong.loc[np.where(df_with_latlong['year'] > 2009)].reset_index(drop = True)


# In[24]:


df_with_latlong_new.head()


# In[25]:


df_with_latlong_new.shape


# In[26]:


df_with_latlong_new.head()


# In[27]:


df_with_latlong_new['AreaCode']= np.nan


# In[28]:


a= np.where((df_with_latlong_new['Latitude'] >= 42.881) & (df_with_latlong_new['Latitude'] <= 42.891) & 
         (df_with_latlong_new['Longitude'] <= -78.870) &
        (df_with_latlong_new['Longitude'] >= -78.880))[0]

b= np.where((df_with_latlong_new['Latitude'] >= 42.875) & (df_with_latlong_new['Latitude'] <= 42.885) & 
         (df_with_latlong_new['Longitude'] <= -78.873) &
        (df_with_latlong_new['Longitude'] >= -78.883))[0]

c= np.where((df_with_latlong_new['Latitude'] >= 42.997) & (df_with_latlong_new['Latitude'] <=43.007 ) & 
         (df_with_latlong_new['Longitude'] <= -78.779) &
        (df_with_latlong_new['Longitude'] >= -78.789))[0]

d= np.where((df_with_latlong_new['Latitude'] >= 42.948) & (df_with_latlong_new['Latitude'] <=42.958 ) & 
         (df_with_latlong_new['Longitude'] <= -78.815) &
        (df_with_latlong_new['Longitude'] >= -78.825))[0]


# In[29]:


df_with_latlong_new.loc[(a),'AreaCode']= 'Downtown'
df_with_latlong_new.loc[(b),'AreaCode']= 'Airport'
df_with_latlong_new.loc[(c),'AreaCode']= 'UB NorthCampus'
df_with_latlong_new.loc[(d),'AreaCode']= 'UB SouthCampus'


# In[30]:


df_with_latlong_new.shape


# In[31]:


df_with_latlong_new['AreaCode'].unique()


# In[32]:


df_with_latlong_new.head()


# In[33]:


df_with_latlong_new['AreaCode'].fillna(0, inplace= True)
df_with_latlong_new.head()


# In[34]:


df_with_latlong_new['AreaCode'].replace(0,'other',inplace=True)
df_with_latlong_new.head()


# In[35]:


file = df_with_latlong_new[['incident_type_primary','year','AreaCode']] 
file['occurrence'] = 1


# In[36]:


file.head()


# In[37]:


file2 = file.groupby(['year','AreaCode', 'incident_type_primary']).agg({'occurrence':'sum'}).reset_index()
file2.head


# Thus after region analysis we can conclude that university at buffalo north campus is the most safest place for women
# as there were no sexual crimes noticed in the past 10 years. Also, downtown is the most unsafest place for women as there were total 227 sexual crimes since the past 10 years and airport region observes all total 130 sexual crimes in the past 10 years whereas university at buffalo south campus observes all total 33 sexual crimes.
