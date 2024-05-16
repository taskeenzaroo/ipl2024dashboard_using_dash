#!/usr/bin/env python
# coding: utf-8

# # Importing all the necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import dash
from dash import Dash, html, dcc, callback, Input, Output
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
#from IPython.display import FileLink


# # Loading all the datasets

# In[2]:


match_sum = pd.read_csv(r"C:\Users\91962\Downloads\datasets\dim_match_summary.csv")
dim_players = pd.read_csv(r"C:\Users\91962\Downloads\datasets\dim_players.csv")
batting_sum = pd.read_csv(r"C:\Users\91962\Downloads\datasets\fact_batting_summary.csv")
bowling_sum = pd.read_csv(r"C:\Users\91962\Downloads\datasets\fact_bowling_summary.csv")


# In[3]:


match_sum.tail()


# In[4]:


dim_players.tail()


# In[5]:


batting_sum.tail()


# In[6]:


bowling_sum.tail()


# In[7]:


bowling_sum.rename(columns = {'0s': 'zeroes'}, inplace = True)
bowling_sum.rename(columns = {'4s': 'fours'}, inplace = True)
bowling_sum.rename(columns = {'6s': 'sixes'}, inplace = True)
batting_sum.rename(columns = {'4s': 'fours'}, inplace = True)
batting_sum.rename(columns = {'6s': 'sixes'}, inplace = True)


# # Checking for null values

# In[8]:


match_sum.isnull().sum()


# In[9]:


dim_players.isnull().sum()


# In[10]:


batting_sum.isnull().sum()


# In[11]:


bowling_sum.isnull().sum()


# # Dropping duplicate values

# In[12]:


match_sum['match_id'].drop_duplicates()


# In[13]:


dim_players.drop_duplicates()


# In[14]:


batting_sum.drop_duplicates()


# In[15]:


bowling_sum.drop_duplicates()


# In[16]:


dim_players.sort_values(by='name', inplace = True)


# In[17]:


dim_players.head()


# In[18]:


# Changing date format
match_sum['matchDate']=pd.to_datetime(match_sum['matchDate'])
match_sum.head()


# In[19]:


match_sum['year'] = match_sum['matchDate'].dt.year
match_sum = match_sum[match_sum['year'] != 2029]
match_sum.head()


# In[20]:


# Changing name format
batting_sum['batsmanName'] = batting_sum['batsmanName'].str.replace(r'(?<=\w)([A-Z])', r' \1')
bowling_sum['bowlerName'] = bowling_sum['bowlerName'].str.replace(r'(?<=\w)([A-Z])', r' \1')
dim_players['name'] = dim_players['name'].str.replace(r'(?<=\w)([A-Z])', r' \1')


# In[21]:


# Adding year to batting and bowling summaries
batting_sum = batting_sum.merge(match_sum[['match_id', 'year']], on='match_id', how='left')
bowling_sum = bowling_sum.merge(match_sum[['match_id', 'year']], on='match_id', how='left')


# In[22]:


batting_sum.head()


# In[23]:


batting_sum['year'] = batting_sum['year'].fillna(0).astype(int) 
batting_sum.head()


# # EDA

# In[24]:


batting_sum.describe()


# In[25]:


bowling_sum.describe()


# In[26]:


match_sum.describe()


# In[27]:


dim_players.describe()


# # 1. Top 10 batsmen based on total runs scored in the past 3 years

# In[28]:


# Join datasets to get the information about all the batsmen and total runs they scored
batting_sum.head()


# In[29]:


# Calculate total runs scored by each batsman
total_runs_by_batsman = batting_sum.groupby(['batsmanName', 'year'])['runs'].sum().reset_index()
total_runs_by_batsman


# In[30]:


# Sort total runs scored
top_10_batsmen = total_runs_by_batsman.sort_values(by='runs', ascending=False).head(10)


# In[31]:


#Graph
top_10_batsmen_graph = px.bar(top_10_batsmen, x='batsmanName', y='runs')
#top_10_batsmen_graph


# # Top 10 batsmen based on past 3 years batting average. (min 60 balls faced in each season)

# In[32]:


# Filter the batsmen who have faced more than 60 balls
batting_sum = batting_sum[batting_sum['balls'] >= 60]
batting_sum.head()


# In[33]:


# Aggregate total runs scored and total times out by each player
agg_batting_stats = batting_sum.groupby(['batsmanName','year']).agg({'runs':'sum','out/not_out':'count'}).reset_index()
agg_batting_stats.columns = ['batsmanName', 'year', 'total_runs', 'total_out']


# In[34]:


# Calculate the batting average
agg_batting_stats['batting_avg'] = agg_batting_stats['total_runs']/agg_batting_stats['total_out']
batting_sum['batting_avg'] = agg_batting_stats['batting_avg']
agg_batting_stats


# In[35]:


# Sort the top 10 values
top_10_batting_avg = agg_batting_stats.sort_values(by='total_runs', ascending=False).head(10)
top_10_batting_avg[['batsmanName','batting_avg']]


# In[36]:


#Graph
top_10_batting_avg_graph = px.bar(top_10_batting_avg, x='batsmanName', y='batting_avg')
#top_10_batting_avg_graph


# # Top 10 batsmen based on past 3 years strike rate (min 60 balls faced in each season)

# In[37]:


batting_sum.head()


# In[38]:


# Aggregate total runs scored and total balls faced by each player
batting_strike_rate = batting_sum.groupby(['batsmanName', 'year']).agg({'runs':'sum','balls':'sum'}).reset_index()
batting_strike_rate.columns= ['batsmanName','year','total_runs','total_balls']
batting_strike_rate


# In[39]:


# Calculate strike rate 
batting_strike_rate['strike_rate'] = (batting_strike_rate['total_runs'] / batting_strike_rate['total_balls']) * 100


# In[40]:


# Sort by strike rate
top_10_strike_rate = batting_strike_rate.sort_values(by='strike_rate', ascending=False).head(10)
top_10_strike_rate


# In[41]:


#Graph
top_10_strike_rate_graph = px.bar(top_10_strike_rate, x='batsmanName', y='strike_rate')
#top_10_strike_rate_graph


# # Top 10 bowlers based on past 3 years total wickets taken

# In[42]:


bowling_sum.head()


# In[43]:


top_bowlers = bowling_sum.groupby(['bowlerName','year'])['wickets'].sum().reset_index()
top_bowlers


# In[44]:


top_10_bowlers = top_bowlers.sort_values(by='wickets', ascending=False).head(10)
top_10_bowlers


# In[45]:


#Graph
top_10_bowlers_graph = px.bar(top_10_bowlers, x='bowlerName', y='wickets')
#top_10_bowlers_graph


# # Top 10 bowlers based on past 3 years bowling average (min 60 balls bowled in each season)

# In[46]:


# Since there is no "balls" column in bowling_summary dataset, we'll add the column manually using "overs" column
bowling_sum['total_balls'] = bowling_sum['overs'] * 6
bowling_sum.head()


# In[47]:


# Filter out the bowlers who have bowled more than 60 balls in each season
filtered_bowling_sum = bowling_sum.groupby(['bowlerName','year']).agg({'total_balls': 'sum', 'runs':'sum','wickets':'sum'}).reset_index()
#filtered_bowling_sum.head()
agg_bowling_stats = filtered_bowling_sum[filtered_bowling_sum['total_balls'] >= 60]
agg_bowling_stats


# In[48]:


# Calculate the bowling average
agg_bowling_stats.columns = ['bowlerName', 'year', 'total_balls','runs_conceded','wickets_taken']
agg_bowling_stats['bowling_avg'] = agg_bowling_stats['runs_conceded']/agg_bowling_stats['wickets_taken']
agg_bowling_stats


# In[49]:


# Sort the results
top_10_bowling_avg = agg_bowling_stats.sort_values(by='bowling_avg', ascending=False).head(10)
top_10_bowling_avg


# In[50]:


#Graph
top_10_bowling_avg_graph = px.bar(top_10_bowling_avg, x='bowlerName', y='bowling_avg')
#top_10_bowling_avg_graph


# # Top 10 bowlers based on past 3 years economy rate. (min 60 balls bowled in each season)

# In[51]:


# Filter out the bowlers who have bowled more than 60 balls in each season
filtered_bowling_sum = bowling_sum.groupby(['bowlerName','year']).agg({'total_balls': 'sum', 'runs':'sum','overs':'sum'}).reset_index()
#filtered_bowling_sum.head()
agg_bowling_stats2 = filtered_bowling_sum[filtered_bowling_sum['total_balls'] >= 60]
agg_bowling_stats2


# In[52]:


# Calculate the economy rate for all the bowlers
agg_bowling_stats2.columns = ['bowlerName', 'year','total_balls','runs_conceded','overs_bowled']
agg_bowling_stats2['economy_rate'] = agg_bowling_stats2['runs_conceded']/agg_bowling_stats2['overs_bowled']
agg_bowling_stats2


# In[53]:


# Sort the top 10 values
top_10_economy_rate = agg_bowling_stats2.sort_values(by='economy_rate',ascending=False).head(10)
top_10_economy_rate


# In[54]:


#Graph
top_10_economy_rate_graph = px.bar(top_10_economy_rate, x='bowlerName', y='economy_rate')
#top_10_economy_rate_graph


# # Top 5 batsmen based on past 3 years boundary % (fours and sixes).

# In[55]:


batting_sum.head()


# In[56]:


# Calculate total balls, fours, and sixes for each player
batting_sum_new = batting_sum.groupby(['batsmanName','year']).agg({'runs':'sum','fours':'sum','sixes':'sum'}).reset_index()
batting_sum_new.head()


# In[57]:


# Calculate boundary % = boundaries_scored [fours + sixes] / total_balls_faced) * 100%
batting_sum_new['boundary_runs'] = batting_sum_new['fours'] * 4 + batting_sum_new['sixes'] * 6
batting_sum_new['boundary%age'] = ((batting_sum_new['boundary_runs'] / batting_sum_new['runs']) * 100).round(2)
batting_sum_new


# In[58]:


# Sort the top 5 values
top_5_boundary_percentage = batting_sum_new.sort_values(by='boundary%age', ascending=False).head(5)
top_5_boundary_percentage


# In[59]:


#Graph
top_5_boundary_per_graph = px.bar(top_5_boundary_percentage, x='batsmanName', y='boundary%age')
#top_5_boundary_per_graph


# #  Top 5 bowlers based on past 3 years dot ball %.

# In[60]:


bowling_sum.head()


# In[61]:


bowling_sum_new = bowling_sum.groupby(['bowlerName', 'year']).agg({'zeroes':'sum', 'total_balls':'sum'}).reset_index()
bowling_sum_new.head()


# In[62]:


# Calculate dot ball percentage for all the bowlers
bowling_sum_new['dotball_%age'] = ((bowling_sum_new['zeroes'] / bowling_sum_new['total_balls']) * 100).round(2)
bowling_sum_new.columns = ['bowlerName', 'year', 'dot_balls','total_balls', 'dotball_%age']
bowling_sum_new.head()


# In[63]:


# Sort the top 5 values
top_5_dot_ball_percentage = bowling_sum_new.sort_values(by='dotball_%age', ascending=False).head(5)
top_5_dot_ball_percentage


# In[64]:


#Graph
top_5_dot_ball_per_graph = px.bar(top_5_dot_ball_percentage, x='bowlerName', y='dotball_%age')
#top_5_dot_ball_per_graph


# # Top 4 teams based on past 3 years winning %

# In[65]:


match_sum.head()


# In[66]:


# Group the data to get total matches and matches won
match_sum_new2 = pd.DataFrame({
    'total_matches': match_sum.groupby(['team1', 'year']).size() + match_sum.groupby(['team2','year']).size(),
     'matches_won': match_sum.groupby(['winner','year']).size(),
    }).reset_index()
match_sum_new2.rename(columns={'level_0':'team'}, inplace=True)
match_sum_new2.head()


# In[67]:


# Calculate winniing percentage
match_sum_new2['winning_percentage'] = (match_sum_new2['matches_won'] / match_sum_new2['total_matches']) * 100
match_sum_new2['winning_percentage'] = match_sum_new2['winning_percentage'].round(2)
match_sum_new2


# In[68]:


# Sort the top 4 values
top_4_winning_percentage = match_sum_new2.sort_values(by='winning_percentage', ascending=False).head(4)
top_4_winning_percentage


# In[69]:


#Graph
top_4_winning_per_graph = px.bar(top_4_winning_percentage, x='team', y='winning_percentage')
#top_4_winning_per_graph


# # Top 2 teams with the highest number of wins achieved by chasing targets over the past 3 years

# In[70]:


match_sum


# In[71]:


# Filter the matches where team that batted second won
chasing_teams = match_sum[match_sum['team2'] == match_sum['winner']]
chasing_teams.head()


# In[72]:


# Aggregate total wins achieved by each chasing team
chasing_team_wins = chasing_teams.groupby(['winner', 'year']).size().reset_index(name='win_count')
#chasing_team_wins


# In[73]:


# Sort the top 2 values
top_2_chasing_team_wins = chasing_team_wins.sort_values(by='win_count', ascending=False).head(2)
top_2_chasing_team_wins


# In[74]:


#Graph
top_2_chasing_team_wins_graph = px.bar(top_2_chasing_team_wins, x='winner', y='win_count')
#top_2_chasing_team_wins_graph


# In[75]:


teams = match_sum['team1'].unique()
years = match_sum['year'].unique()
batting_sum.head()


# # Creating interactive dashboard using Dash

# In[76]:


#Initialize the dash app
app = Dash(__name__)


# In[77]:


# color palette
colors = ['#af92b5', '#8b7991', '#6f597a', '#624b6e', '#503d5c', '#483952']

# additional layout preferences
chart_font = {'family': "Georgia, serif", 'size': 14, 'color': "#503d5c"}
layout_template = "plotly_white"


# In[78]:


# App Layout
app.layout = html.Div([
    html.H1("Cricket Performance Dashboard", style={'textAlign': 'center', 'color': '#483552', 'size': 18}),   
    # creating a dropdown to select the year
    html.Div([
        html.Label('Select year: ', style={'color': '#503d5c'}),
        dcc.Dropdown(
            id='year-selector',
            options=[{'label': year, 'value': year} for year in years],
            value=years[1],
            multi=False,
            style={'width': "100%"})
    ], style={'padding': 10, 'margin-bottom': '20px'}),
    
    # First set of charts
    html.Div([
        dcc.Graph(id='top-10-batsmen'),
        dcc.Graph(id='top-5-batting-avg'),
        dcc.Graph(id='top-5-strike-rate'),
        dcc.Graph(id='top-5-boundary-per')
    ], style={'width': '49%', 'display': 'inline-block'}),
    
    # Second set of charts
    html.Div([
        dcc.Graph(id='top-10-bowlers'),
        dcc.Graph(id='top-10-bowling-avg'),
        dcc.Graph(id='top-10-economy-rate'),
        dcc.Graph(id='top-5-dotball-per')
    ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
    
    # Last two charts side by side
    html.Div([
        html.Div([
            dcc.Graph(id='top-4-winning-per')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='top-2-chasing-team-wins')
        ], style={'width': '50%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'width': '100%'})
], style={'backgroundColor': '#00000'})


# In[79]:


# Callback for Top 10 Batsmen Runs
@app.callback(
    Output('top-10-batsmen', 'figure'),
    Input('year-selector', 'value')
)
def update_top_10_batsmen(selected_year):
    filtered_top_10_batsmen = total_runs_by_batsman[total_runs_by_batsman['year'] == int(selected_year)]
    filtered_top_10_batsmen = filtered_top_10_batsmen.sort_values(by='runs', ascending=False).head(10)
    fig = px.bar(filtered_top_10_batsmen, x='batsmanName', y='runs', color_discrete_sequence=[colors[0]], title="Top 10 Batsmen by runs")
    fig.update_layout(template=layout_template, font=chart_font)
    return fig

# Callback for Top 10 Batting Average
@app.callback(
    Output('top-5-batting-avg', 'figure'),
    Input('year-selector', 'value')
)
def update_top_10_batting_avg(selected_year):
    filtered_top_10_batting_avg = agg_batting_stats[(agg_batting_stats['year'] == int(selected_year))]
    filtered_top_10_batting_avg = filtered_top_10_batting_avg.sort_values(by='batting_avg', ascending=False).head(10)
    fig = px.bar(filtered_top_10_batting_avg, x='batting_avg', y='batsmanName', orientation='h', color_discrete_sequence=[colors[1]], title="Top 5 Batsmen by Batting Average")
    fig.update_layout(template=layout_template, font=chart_font)
    return fig

# Callback for Top 10 Strike Rate
@app.callback(
    Output('top-5-strike-rate', 'figure'),
    Input('year-selector', 'value')
)
def update_top_10_strike_rate(selected_year):
    filtered_top_10_strike_rate = batting_strike_rate[(batting_strike_rate['year'] == int(selected_year))]
    filtered_top_10_strike_rate = filtered_top_10_strike_rate.sort_values(by='strike_rate', ascending=False).head(10)
    fig = px.scatter(filtered_top_10_strike_rate, x='batsmanName', y='strike_rate', size='strike_rate', color_continuous_scale=[colors[2], colors[3]], title="Top 5 Batsmen by Strike Rate")
    fig.update_layout(template=layout_template, font=chart_font)
    fig.update_traces(marker=dict(color= colors[5], size=20))
    return fig

# Callback for Top 5 Boundary Percentage
@app.callback(
    Output('top-5-boundary-per', 'figure'),
    Input('year-selector', 'value')
)
def update_top_5_boundary_per(selected_year):
    filtered_top_5_boundary_per = batting_sum_new[(batting_sum_new['year'] == int(selected_year))]
    filtered_top_5_boundary_per = filtered_top_5_boundary_per.sort_values(by='boundary%age', ascending=False).head(5)
    fig = px.pie(filtered_top_5_boundary_per, names='batsmanName', values='boundary%age', hole=0.4, color_discrete_sequence=colors, title="Top 5 Batsmen by Boundary Percentage")
    fig.update_layout(template=layout_template, font=chart_font)
    return fig

# Callback for Top 10 Bowlers Wickets
@app.callback(
    Output('top-10-bowlers', 'figure'),
    Input('year-selector', 'value')
)
def update_top_10_bowlers(selected_year):
    filtered_top_10_bowlers = top_bowlers[(top_bowlers['year'] == int(selected_year))]
    filtered_top_10_bowlers = filtered_top_10_bowlers.sort_values(by='wickets', ascending=False).head(10)
    fig = px.bar(filtered_top_10_bowlers, x='bowlerName', y='wickets', color_discrete_sequence=[colors[4]], title="Top 10 Bowlers by wickets")
    fig.update_layout(template=layout_template, font=chart_font)
    return fig

# Callback for Top 10 Bowling Average
@app.callback(
    Output('top-10-bowling-avg', 'figure'),
    Input('year-selector', 'value')
)
def update_top_10_bowling_avg(selected_year):
    filtered_top_10_bowling_avg = agg_bowling_stats[(agg_bowling_stats['year'] == int(selected_year))]
    filtered_top_10_bowling_avg = filtered_top_10_bowling_avg.sort_values(by='bowling_avg', ascending=False).head(10)
    fig = px.bar(filtered_top_10_bowling_avg, x='bowling_avg', y='bowlerName', orientation='h', color_discrete_sequence=[colors[3]], title="Top 10 Bowlers by Bowling Average")
    fig.update_layout(template=layout_template, font=chart_font)
    return fig

# Callback for Top 10 Economy Rate
@app.callback(
    Output('top-10-economy-rate', 'figure'),
    Input('year-selector', 'value')
)
def update_top_10_economy_rate(selected_year):
    filtered_top_10_economy_rate = agg_bowling_stats2[(agg_bowling_stats2['year'] == int(selected_year))]
    filtered_top_10_economy_rate = filtered_top_10_economy_rate.sort_values(by='economy_rate', ascending=False).head(10)
    fig = px.scatter(filtered_top_10_economy_rate, x='bowlerName', y='economy_rate', size='economy_rate', color_continuous_scale=[colors[1], colors[2]], title="Top 10 Bowlers by Economy Rate")
    fig.update_layout(template=layout_template, font=chart_font)
    fig.update_traces(marker=dict(color= colors[5], size=20))
    return fig

# Callback for Top 5 Dot Ball Percentage
@app.callback(
    Output('top-5-dotball-per', 'figure'),
    Input('year-selector', 'value')
)
def update_top_5_dotball_per(selected_year):
    filtered_top_5_dotball_per = bowling_sum_new[(bowling_sum_new['year'] == int(selected_year))]
    filtered_top_5_dotball_per = filtered_top_5_dotball_per.sort_values(by='dotball_%age', ascending=False).head(5)
    fig = px.pie(filtered_top_5_dotball_per, names='bowlerName', values='dotball_%age', hole=0.4, color_discrete_sequence=[colors[3], colors[4]], title="Top 5 Bowlers by Dot Ball Percentage")
    fig.update_layout(template=layout_template, font=chart_font)
    return fig

# Callback for Top 4 Winning Percentage
@app.callback(
    Output('top-4-winning-per', 'figure'),
    Input('year-selector', 'value')
)
def update_top_4_winning_per(selected_year):
    filtered_top_4_winning_per = match_sum_new2[(match_sum_new2['year'] == int(selected_year))]
    filtered_top_4_winning_per = filtered_top_4_winning_per.sort_values(by='winning_percentage', ascending=False).head(4)
    fig = px.treemap(filtered_top_4_winning_per, path=['team'], values='winning_percentage', color='winning_percentage', color_continuous_scale=[colors[1], colors[4]], title="Top 4 Teams by Winning Percentage")
    fig.update_layout(template=layout_template, font=chart_font)
    return fig

# Callback for Top 2 Chasing Team Wins
@app.callback(
    Output('top-2-chasing-team-wins', 'figure'),
    Input('year-selector', 'value')
)
def update_top_2_chasing_team_wins(selected_year):
    filtered_top_2_chasing_team_wins = chasing_team_wins[(chasing_team_wins['year'] == int(selected_year))]
    filtered_top_2_chasing_team_wins = filtered_top_2_chasing_team_wins.sort_values(by='win_count', ascending=False).head(2)
    fig = px.sunburst(filtered_top_2_chasing_team_wins, path=['winner'], values='win_count', color='win_count', color_continuous_scale=[colors[1], colors[3]],title="Top 2 Chasing Team Wins")
    fig.update_layout(template=layout_template, font=chart_font)
    return fig


# In[80]:


#Run the app
if __name__ == '__main__':
    app.run(debug = True)


# In[ ]:




