#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ## Data Preprocessing

# In[4]:


df = pd.read_csv("/Users/kushal/Downloads/Crime_Incidents.csv", low_memory=False)


# In[5]:


df.head(5)


# In[6]:


df.columns, df.shape


# In[7]:


df['incident_type_primary'].unique()


# In[8]:


df2 = df[df['incident_type_primary'].isin({'SEXUAL ABUSE', 'RAPE', 'Other Sexual Offense',
                                 'Sexual Assault'})]


# In[9]:


df2.shape, df2['incident_type_primary'].unique()


# In[10]:


df2.head(20)


# In[11]:


df2.dropna()


# In[12]:


df3 = df2.reset_index(drop = True)
df3.head()
#df3.shape


# In[13]:


## Cleaning and splitting incident_datetime column into year and month


# In[14]:


date_times = []
for num in range(df2.shape[0]):
    date_times.append(df2['incident_datetime'].iloc[num].split(" ", 2)[0])
date_times[:5]


# In[15]:


date_times_2 = pd.to_datetime(date_times,infer_datetime_format=True)


# In[16]:


df3['year'] = date_times_2.year
df3['month'] = date_times_2.month


# In[17]:


df3.head(5)


# In[18]:


df4 = df[df['incident_type_primary'].isin({'ASSAULT', 'LARCENY/THEFT',  'UUV', 'BURGLARY',
        'ROBBERY', 'CRIM NEGLIGENT HOMICIDE', 'THEFT OF SERVICES',
       'MURDER', 'Breaking & Entering', 'Assault', 'Theft', 'Robbery',
       'Theft of Vehicle', 'AGG ASSAULT ON P/OFFICER',
         'Homicide',
       'AGGR ASSAULT', 'MANSLAUGHTER'})]


# In[19]:


df4.shape


# In[20]:


df4.head(5)


# In[21]:


df4['incident_datetime'].isnull().sum()


# In[22]:


df5 = df4.reset_index(drop = True)
np.where(df5['incident_datetime'].isnull() == True)[0]
df6 = df5.drop(np.where(df5['incident_datetime'].isnull() == True)[0], axis = 0)


# In[23]:


df6.shape, df5.shape


# In[24]:


df6['incident_datetime'].isnull().sum()


# In[25]:


date_times = []
for num in range(df6.shape[0]):
    date_times.append(df6['incident_datetime'].iloc[num].split(" ", 2)[0])
date_times[:5]


# In[26]:


date_times_2 = pd.to_datetime(date_times,infer_datetime_format=True)


# In[27]:


df6['year'] = date_times_2.year
df6['month'] = date_times_2.month


# In[28]:


df6.head(5)


# In[118]:


## Subsetting only sexual crimes data 


# In[29]:


sub_df1= df3[['year','incident_type_primary']]


# In[30]:


sub_df1


# In[31]:


grouped= sub_df1.groupby(['year','incident_type_primary']).groups
grouped


# In[32]:


sub_df1['incident_type_primary'].unique()


# In[33]:


incident_type_primary_counts= sub_df1['incident_type_primary'].value_counts()
incident_type_primary_counts


# In[34]:


sub_df1['year'].unique()


# In[35]:


year_counts= sub_df1['year'].value_counts()
year_counts


# In[119]:


## Boxplot showing the sexual crime incidents and the corresponding years


# In[120]:


sns.boxplot(x='year', y = 'incident_type_primary', data = sub_df1)


# In[121]:


## Subsetting nonsexual crime data


# In[37]:


sub_df2= df6[['year','incident_type_primary']]
sub_df2


# In[38]:


sub_df2['year'].unique()


# In[39]:


year_counts= sub_df2['year'].value_counts()
year_counts


# In[40]:


df_crimes= {'Year': [2007,2009,2006,2010,2015,2018,2016,2011,2014,2017,2012,2013,2008,2019,2005,2004,2003,2002,
                          1998,1996,2000,2001,1999,1992,1978,1993,1963,1967,1983,1976],
                 'Sexualcrime_counts': [382,380,379,359,337,329,326,320,312,308,308,297,290,275,41,25,14,9,6,6,5,5,3,1,1,1,1,1,
                                1,1],
                 'Nonsexualcrime_counts': [21575,21453,19067,21348,16930,15126,16123,20152,17206,15089,20302,
                                              18353,13244,12862,274,45,33,24,6,6,51,21,2,2,2,1,1,1,1,1]}


# In[122]:


## Dataframe with year, sexual and nonsexual crimes


# In[41]:


df_new1 = pd.DataFrame(df_crimes)
df_new1


# In[42]:


df_new1_sorted= df_new1.sort_values(['Year'])


# In[43]:


df_new1_sorted1= df_new1_sorted.reset_index(drop = True)
df_new1_sorted1


# In[44]:


df_new1_sorted_proportionality= df_new1_sorted1['Sexualcrime_counts']/ df_new1_sorted1['Nonsexualcrime_counts']


# In[45]:


df_new1_sorted1['Proportion'] = df_new1_sorted_proportionality


# Since the data doesn't look stable from 1963 to 2009, therefore I am considering the last decade's data that is from 2010 to 2019. We do observe that with increase in time from 2006 to 2019 mostly nonsexual crimes decreased and there's no exact pattern for sexual crimes. As a result, proportionality increases with increase in time. Below are some plots to have better understanding of this.

# In[46]:


df_new1_sorted1.plot.scatter(x = 'Year', y = 'Sexualcrime_counts')


# The above scatterplot signifies that sexual crimes kind of increased approximately from 2007 onwards.

# In[47]:


df_new1_sorted1.plot.scatter(x = 'Year', y = 'Nonsexualcrime_counts')


# The scatterplot shows thats even nonsexual crimes increased from 2007 onwards. However, there's a sharp decrease of nonsexual crime incident around 2019.

# In[48]:


df_new1_sorted1.plot.scatter(x = 'Year', y = 'Proportion')


# In[49]:


plt.scatter(x = 'Year', y = 'Sexualcrime_counts')
plt.title('Year vs Sexualcrime_counts')
plt.xlabel('Year')
plt.ylabel('Sexualcrime_counts')
plt.show()


# In[50]:


df_new11= df_new1_sorted1.loc[np.where(df_new1_sorted1['Year'] > 2009)]
df_new11


# In[51]:


df_new11.plot.scatter(x = 'Year', y = 'Sexualcrime_counts')


# In[52]:


df_new11.plot.scatter(x = 'Year', y = 'Nonsexualcrime_counts')


# In[53]:


df_new11.plot.scatter(x = 'Year', y = 'Proportion')


# Here we see nonsexual crimes decreased over the period of time however sexual crimes almost varies and there is no specific pattern of sexual crimes.

# In[54]:


df3.head()


# In[55]:


df3['incident_type_primary'].unique()


# In[56]:


incident_type_primary_counts= df3['incident_type_primary'].value_counts()
incident_type_primary_counts


# Certainly, from the above analysis we can say the value counts of rape is the maximum in comparison to other sexual crimes.

# In[57]:


year_counts= df3['year'].value_counts()
year_counts


# In[58]:


df_filtered_rapes = df3['incident_type_primary']=="RAPE"
df_filtered_rapes


# In[59]:


df_new_filtered_RAPES= df3.loc[np.where(df3['incident_type_primary'] == "RAPE")]
df_new_filtered_RAPES


# In[60]:


df_new_filtered_RAPES['address_1'].unique()


# In[61]:


year_new_counts1=df_new_filtered_RAPES ['year'].value_counts().sort_index()
year_new_counts1


# In[62]:


df_new_filtered_Sexualabuse= df3.loc[np.where(df3['incident_type_primary'] == "SEXUAL ABUSE")]
year_new_counts3=df_new_filtered_Sexualabuse ['year'].value_counts().sort_index()
year_new_counts3


# In[63]:


df_sexualcrimes_final1= {'Year': [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019],
                 'Sexualabuse_counts': [154,168,129,121,141,160,157,160,186,168],
                 'Rapes_counts': [205,152,179,176,171,177,169,148,143,107]}


# In[64]:


df_sexualcrimes_final_1 = pd.DataFrame(df_sexualcrimes_final1)
df_sexualcrimes_final_1


# In[65]:


fig,ax=plt.subplots()
ax.plot(df_sexualcrimes_final_1.Year,df_sexualcrimes_final_1.Sexualabuse_counts, marker="o")
ax.set_xlabel("Year")
ax.set_ylabel("Sexualabuse_counts")
ax.plot(df_sexualcrimes_final_1.Year, df_sexualcrimes_final_1["Rapes_counts"], marker="o")
plt.show()



# In[66]:


df_sexualcrimes_final_1.plot(x='Year', y='Sexualabuse_counts' )
df_sexualcrimes_final_1.plot(x='Year', y='Rapes_counts' )


# Here we fitted a linear regression model to find the intercept values in order to see whether the variables are positively correlated or negatively correlated. Along with this, we computed the stats model.

# In[67]:


from sklearn.linear_model import LinearRegression
X_train= df_sexualcrimes_final_1['Year']
y_train= df_sexualcrimes_final_1['Sexualabuse_counts']
X_train = np.array(X_train).reshape(-1, 1)
regressor = LinearRegression(fit_intercept=True)
regressor.fit(X_train,y_train)
print(regressor.coef_)


# In[68]:


import statsmodels.api as sm
mod = sm.OLS(y_train,sm.add_constant(X_train))
fii = mod.fit()
p_values = fii.summary2().tables[1]['P>|t|']
p_values
fii.summary2()


# In[69]:


from sklearn.linear_model import LinearRegression
X_train= df_sexualcrimes_final_1['Year']
y_train= df_sexualcrimes_final_1['Rapes_counts']
X_train = np.array(X_train).reshape(-1, 1)
regressor = LinearRegression(fit_intercept=True)
regressor.fit(X_train,y_train)
print(regressor.coef_)


# In[70]:


import statsmodels.api as sm
mod = sm.OLS(y_train,sm.add_constant(X_train))
fii = mod.fit()
p_values = fii.summary2().tables[1]['P>|t|']
p_values
fii.summary2()

