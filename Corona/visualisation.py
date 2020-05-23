#!/usr/bin/env python
# coding: utf-8

# We are just going to visualize the dataset given in this python notebook using Choropleth Maps.
# 
# 
# A choropleth map is a type of thematic map where areas or regions are shaded in proportion to a given data variable.

# Static choropleth maps are most useful when you want to compare a desired variable by region. For example, if you wanted to compare the crime rate of each state in the US at a given moment, you could visualize it with a static choropleth.
# 
# 
# An animated or dynamic choropleth map is similar to a static choropleth map, except that you can compare a variable by region, over time. This adds a third dimension of information and is what makes these visualizations so interesting and powerful.

# In[2]:


# Import libraries
import numpy as np 
import pandas as pd 
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[4]:


# Read Data
df = pd.read_csv("covid_19_data.csv")
print(df.head())


# In[5]:


df = df.rename(columns={'Country/Region':'Country'})
df = df.rename(columns={'ObservationDate':'Date'})


# In[6]:


df.head()


# In[7]:


# Manipulate Dataframe
df_countries = df.groupby(['Country', 'Date']).sum().reset_index().sort_values('Date', ascending=False)
df_countries = df_countries.drop_duplicates(subset = ['Country'])
df_countries = df_countries[df_countries['Confirmed']>0]


# In[8]:


df.head()


# In[9]:


df_countries.head()


# In[10]:


# Create the Choropleth for confirmed cases
fig = go.Figure(data=go.Choropleth(
    locations = df_countries['Country'],
    locationmode = 'country names',
    z = df_countries['Confirmed'],
    colorscale = 'Reds',
    marker_line_color = 'black',
    marker_line_width = 0.5,
))
fig.update_layout(
    title_text = 'Confirmed Cases as of March 28, 2020',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
        projection_type = 'equirectangular'
    )
)


# In[11]:


# Manipulate Dataframe
df_countries1 = df.groupby(['Country', 'Date']).sum().reset_index().sort_values('Date', ascending=False)
df_countries1 = df_countries1.drop_duplicates(subset = ['Country'])
df_countries1 = df_countries1[df_countries1['Deaths']>0]


# In[12]:


df_countries1


# In[13]:


# Create the Choropleth for death cases
fig = go.Figure(data=go.Choropleth(
    locations = df_countries1['Country'],
    locationmode = 'country names',
    z = df_countries1['Deaths'],
    colorscale = 'Reds',
    marker_line_color = 'black',
    marker_line_width = 0.5,
))
fig.update_layout(
    title_text = 'Death Cases as of March 28, 2020',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
        projection_type = 'equirectangular'
    )
)


# In[14]:


# Manipulate Dataframe
df_countries2 = df.groupby(['Country', 'Date']).sum().reset_index().sort_values('Date', ascending=False)
df_countries2 = df_countries2.drop_duplicates(subset = ['Country'])
df_countries2 = df_countries2[df_countries2['Recovered']>0]


# In[15]:


df_countries2


# In[16]:


# Create the Choropleth for recovered cases
fig = go.Figure(data=go.Choropleth(
    locations = df_countries2['Country'],
    locationmode = 'country names',
    z = df_countries2['Recovered'],
    colorscale = 'Reds',
    marker_line_color = 'black',
    marker_line_width = 0.5,
))
fig.update_layout(
    title_text = 'Recovered Cases as of March 28, 2020',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
        projection_type = 'equirectangular'
    )
)


# In the above we are just inputting the location, location mode and z as the parameters, rest all of the code is the standard plotly code for choropleth graph. Reference: https://plotly.com/python/choropleth-maps/

# Let's look at how much more effective and engaging an animated choropleth map is compared to a static one.

# In[17]:


# Manipulating the original dataframe
df_countrydate = df[df['Confirmed']>0]
df_countrydate = df_countrydate.groupby(['Date','Country']).sum().reset_index()
df_countrydate


# In[18]:


# Creating the visualization
fig = px.choropleth(df_countrydate, 
                    locations="Country", 
                    locationmode = "country names",
                    color="Confirmed", 
                    hover_name="Country", 
                    animation_frame="Date"
                   )
fig.update_layout(
    title_text = 'Global Spread of Coronavirus (wrt confirmed cases)',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()


# In[19]:


# Manipulating the original dataframe
df_countrydate1 = df[df['Deaths']>0]
df_countrydate1 = df_countrydate1.groupby(['Date','Country']).sum().reset_index()
df_countrydate1


# In[21]:


# Creating the visualization
fig = px.choropleth(df_countrydate1, 
                    locations="Country", 
                    locationmode = "country names",
                    color="Deaths", 
                    hover_name="Country", 
                    animation_frame="Date"
                   )
fig.update_layout(
    title_text = 'Globally deceased due to Coronavirus (wrt deaths cases)',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()


# In[22]:


# Manipulating the original dataframe
df_countrydate2 = df[df['Deaths']>0]
df_countrydate2 = df_countrydate2.groupby(['Date','Country']).sum().reset_index()
df_countrydate2


# In[23]:


# Creating the visualization
fig = px.choropleth(df_countrydate2, 
                    locations="Country", 
                    locationmode = "country names",
                    color="Recovered", 
                    hover_name="Country", 
                    animation_frame="Date"
                   )
fig.update_layout(
    title_text = 'Global recovery due to Coronavirus (wrt recovered cases)',
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()

