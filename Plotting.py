# Plotting for Gaming Industry

###############################################################################
"""
In this .py file, we will perform plotting in matplotlib, seaborn, and plotly
on games, game sales, and twitch datasets, including for Shiny datasets

We have in total of 4 sections of plots for game industry analysis:
    1. Plots for 'games' dataset, relationship betweem rating, number of
    players, genre analysis
    2. Plots for 'twitch' dataset, Activision Blizzard games viewing statistics
    on Twitch over time, viewing statistics by publishers
    3. Plots for 'sales' dataset, on sales genre, used for Shiny
    4. Plots for text processing, word cloud and sentimental distribution

"""
###############################################################################

# import packages
import os
import sys
import random
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime
from matplotlib.ticker import MaxNLocator
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
import pandas as pd
from wordcloud import WordCloud
import ast
from IPython.core.display import HTML

PATH = r'C:\Users\zhang\OneDrive\Documents\GitHub\final-project-our_code_works'
DATAPATH = r'C:\Users\zhang\OneDrive\Documents\GitHub\final-project-our_code_works\data'
IMAGEPATH = r'C:\Users\zhang\OneDrive\Documents\GitHub\final-project-our_code_works\image'

sys.path.append(PATH)
import DataCleaning as data
import TextProcessing as textprocess

# Set the custom color palette
custom_palette = ["#ff6700", "#9400d3", "#ff8f00", "#800080"]
sns.set_palette(custom_palette)


def custom_colors(word, font_size, position, orientation, random_state=None,
                  **kwargs):
    """Customize color palatte to be randomly chosen from for word cloud."""
    """Provide a custom palatte to be randomly choosing from."""
    colors = ["#ff6700", "#9400d3", "#ffa500", "#ff7f50", "#ff4500",
              "#ff4f00", "#ffbf00", "#ff8f00", "#800080", "#dda0dd", "#9370db",
              "#ba55d3", "#6a5acd", "#8a2be2", "#9932cc", "#9f00c5",
              "#69359c", "#7851a9", "#dcd0ff",]
    return random.choice(colors)


# Run either to select display plotly in spyder or browser
# pio.renderers.default = "svg"  # using spyder to display plotly
# pio.renderers.default='browser'  # using browser to display plotly


# %% Read Data from DataCleaning, TextProcessing Files, Modify for
#    Visualization usage

games = textprocess.games
sales = data.import_and_clean_sales()
gamesgenre_exploded, genre_annual_data, genre_data = data.\
    get_games_explode_and_groupby_genre()
twitch = data.import_and_clean_twitch()
twitch_merge, actblz_stats = textprocess.merge_data_for_regression(
    twitch_df=twitch)


def get_twitch_merge_exploded_by_teams():
    """Modify merged twitch data for visualizations."""
    twitch_merge_exploded = twitch_merge.explode('Team').reset_index()
    twitch_merge_exploded.loc[twitch_merge_exploded['actblz_indicator'] == 1,
                              'Team'] = 'Activision Blizzard'
    exclude_values = ['GOA Games Services Ltd.', 'Rockstar North',
                      'Ubisoft Montreal', 'Respawn Entertainment',
                      'Starbreeze publishing AB']
    twitch_merge_exploded = twitch_merge_exploded[~twitch_merge_exploded[
        'Team'].isin(exclude_values)]
    return twitch_merge_exploded


# Execute function and merge twitch with sales for visualizations
twitch_merge_exploded = get_twitch_merge_exploded_by_teams()
twitch_merge_all = twitch_merge.merge(sales, left_on='Game', right_on='Name',
                                      how='inner').reset_index()


# %% Plots for games Dataset

def visualize_gameyear_players_jointplot(name):
    """Plot the relationship between year published and number of players
    for a game.
    """
    plt.figure(figsize=(20, 12))
    jp = sns.jointplot(data=games[games['Year'] > 2005], x="Year",
                       y="N_of_players", hue="Publisher",
                       xlim=[2005, 2024], ylim=[-10, 50000],
                       palette=custom_palette, kind='hist')
    plt.suptitle("Relationship between Year Published and Number of Players "
                 "for a Game, 2005-2023", fontsize=8, fontweight='bold')
    plt.tight_layout()
    jp.ax_joint.set_ylabel('Number of Players per Game')
    jp.ax_joint.set_xlabel('Year Published')
    plt.subplots_adjust(top=0.95)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2))
    jointplot = os.path.join(IMAGEPATH, f"{name}_jointplot.png")
    jp.fig.savefig(jointplot, dpi=300)
    plt.show()


def visualize_rating_players_jointplot(name):
    """Plot the relationship between year published and rating for a game."""
    sns.set_palette(custom_palette)
    jp = sns.jointplot(x="Rating", y="N_of_players",
                       data=games[games['Year'] > 2005],
                       kind="scatter", hue="Publisher", xlim=[1.8, 4.65],
                       ylim=[-10, 35000], palette=custom_palette)
    plt.suptitle("Relationship between Game Rating and Number of Players, " +
                 "2005-2023", fontsize=8, fontweight='bold')
    plt.tight_layout()
    jp.ax_joint.set_ylabel('Number of Players per Game')
    jp.ax_joint.set_xlabel('Game Rating, out of 5')
    plt.subplots_adjust(top=0.95)
    jointplot = os.path.join(IMAGEPATH, f"{name}_jointplot.png")
    jp.fig.savefig(jointplot, dpi=300)
    plt.show()


def visualize_comparative_rating_players_regplot(name):
    """Plot the relationship between year published and rating for a game."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    # Relationship between rating and number of players
    sns.regplot(x="Rating", y="N_of_players", data=games[games['Year'] > 2005],
                color=custom_palette[0], fit_reg=True, ax=ax1)
    ax1.set_title("Relationship between Game Rating and Number of Players " +
                  "for All Games")
    ax1.set_ylabel('Number of Players')
    ax1.set_xlabel('Game Rating')

    # Relationship between year published and number of players
    sns.regplot(x="Rating", y="N_of_players", data=games[(
        games['Year'] > 2005) & (games['Publisher'] == 'Activision Blizzard')],
        color=custom_palette[1], fit_reg=True, ax=ax2)
    ax2.set_title("Relationship between Game Rating and Number of Players " +
                  "for Activision Blizzard Games")
    ax2.set_ylabel('Number of players')
    ax2.set_xlabel('Game Rating')
    fig.suptitle("Comparative Analysis of Game Ratings and Number of " +
                 "Players for Games from Different Publishers, 2005-2023",
                 fontsize=15, fontweight='bold')
    regplot = os.path.join(IMAGEPATH, f"{name}_regplot.png")
    fig.savefig(regplot, dpi=300)
    plt.show()


def visualize_genre_year_heatmap(name):
    """Plot relevant game statistics per year published per genre."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
    # Average per game number of players per year by genre
    genre_annual_nplayers = genre_annual_data[
        genre_annual_data['Year'] > 2010].pivot(index="Genres",
                                                columns="Year",
                                                values="N_of_players")
    sns.heatmap(genre_annual_nplayers, annot=True, ax=ax1)
    ax1.set_title("Avg Number of Players per Game by Genre, Sized by " +
                  "Counts per Year Published, 2010-2023")
    ax1.set_xlabel('Year Published')
    ax1.set_ylabel('Average Game Rating')
    colorbar = ax1.collections[0].colorbar
    colorbar.set_label('Average Number of Players')

    # Average game rating per year by genre
    genre_annual_rating = genre_annual_data[
        genre_annual_data['Year'] > 2010].pivot(index="Genres",
                                                columns="Year",
                                                values="Rating")
    sns.heatmap(genre_annual_rating, annot=True, ax=ax2)
    ax2.set_title("Avg Game Rating by Genre, Sized by Counts per Year" +
                  " Published Published, 2010-2023")
    ax2.set_xlabel('Year Published')
    ax2.set_ylabel('Average Game Rating')
    colorbar = ax2.collections[0].colorbar
    colorbar.set_label('Average Game Rating')
    fig.suptitle("By Genre Analysis per Year Published, 2010-2023",
                 fontsize=15, fontweight='bold')
    heatmap = os.path.join(IMAGEPATH, f"{name}_heatmap.png")
    fig.savefig(heatmap, dpi=500)
    plt.show()


# Function execution
# visualize_gameyear_players_jointplot('Figure1')
# visualize_rating_players_jointplot('Figure2')
# visualize_comparative_rating_players_regplot('Figure3')
# visualize_genre_year_heatmap('Figure4')

# %% Plots for Genre Analysis using sales Dataset

###############################################################################
# Shiny 1: Given selected genre, plot a bubble plot of a genre's average
# rating, average n_players per year published

def visualize_genre_stats_by_genre(selected_genre):
    """Plot a bubble plot of a genre's average rating given genre."""
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.scatterplot(data=genre_annual_data[genre_annual_data[
        'Genres'] == selected_genre], x='Year', y='Rating',
        size='N_of_players', hue='Genres', alpha=0.5, sizes=(50, 1000))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
               title='Genres and Number of Players')
    plt.ylabel('Average Game Rating')
    plt.xlabel('Year Published')
    plt.title("Avg Annual Game Rating by Genre, Sized by Number of Players " +
              "per Year Published, 2005-2023", size=15)
    plt.show()


# Sample usage:
# visualize_genre_stats_by_genre('Strategy')

###############################################################################

###############################################################################
# Shiny 2: Plotly treemap that shows top five publishers with their sales
# per genre sales per publisher per genre
genre_pub_sales = sales.groupby(["Genre", "Publisher"])[['Global_Sales']]\
    .sum().reset_index()


def get_top_publishers(group):
    """Get top 5 publishers per genre based on global sales."""
    return group.nlargest(5, 'Global_Sales')


# All genre sales in plotly
# Apply the function to each genre group and plot the treemap
def visualize_top_publishers_per_salesgenre_master():
    """Plot the treemap of top 5 publishers with sales in a all genres."""
    top_pub_per_genre = genre_pub_sales.groupby('Genre').apply(
        get_top_publishers)
    top_pub_per_genre = top_pub_per_genre.reset_index(drop=True)

    # Plot in plotly, only as the master file
    # Use the below by genre plot in Shiny
    fig, ax = plt.subplots(figsize=(15, 10))
    fig = px.treemap(top_pub_per_genre,
                     path=[px.Constant('All Genres'), 'Genre', 'Publisher'],
                     values='Global_Sales',
                     color='Publisher',
                     title='Treemap of Top Publishers per Genre')
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    fig.show()


def visualize_top_publishers_per_salesgenre(selected_genre):
    """Plot the treemap of top 5 publishers with sales in a given genre."""
    # Get top 5 publishers for each genre based on global sales
    top_pub_per_genre = genre_pub_sales.groupby('Genre')\
        .apply(get_top_publishers)
    top_pub_per_genre = top_pub_per_genre.reset_index(drop=True)

    # Get and print top 5 publishers for the selected genre
    genre_pub = sales.groupby(['Genre', 'Publisher'])[['Global_Sales']]\
        .sum().sort_values(['Genre', 'Global_Sales'], ascending=False)\
        .reset_index()
    genre_top5 = genre_pub.loc[genre_pub['Genre'].isin([selected_genre])]\
        .reset_index(drop=True).head(5)
    print(genre_top5[['Publisher', 'Global_Sales']])

    # Plot the treemap in Plotly
    fig, ax = plt.subplots(figsize=(15, 10))
    fig = px.treemap(top_pub_per_genre[top_pub_per_genre['Genre'] == 'Action'],
                     path=[px.Constant('Genres'), 'Genre', 'Publisher'],
                     values='Global_Sales',
                     color='Publisher',
                     title='Treemap of Top 5 Publishers per Genre or for ' +
                     f'{selected_genre}')
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    fig.show()


# Sample
# visualize_top_publishers_per_salesgenre('Strategy')
# dropdown of genre list: salesgenre_list = list(set(genre_pub_sales['Genre']))

##############################################################################

# Function execution
# visualize_genre_year_heatmap('Figure5')


# %% Plots for twitch Dataset

# %%% Twitch Viewing Statistics by Publishers

# Aggregate total views and calculate average Hours_watched and
# avg_viewer_ratio for each publisher, select top 20 publishers and produce
# game_aggregates which contains only games from top 20 publishers
publisher_aggregates = twitch_merge_exploded.groupby('Team').agg({
    'Hours_watched': 'sum', 'Avg_viewer_ratio': 'mean'}).sort_values(
        by='Hours_watched', ascending=False).reset_index()
top_publishers = list(publisher_aggregates.head(20)['Team'])
game_aggregates = twitch_merge_exploded.groupby(['Team', 'Game'])\
    .agg({'Hours_watched': 'sum', 'Avg_viewer_ratio': 'mean',
          'N_of_players': 'mean'}).sort_values(
        by='Hours_watched', ascending=False).reset_index()
top_games_twitch = game_aggregates[game_aggregates['Team'].isin(
    top_publishers)]


def visualize_twitchstats_by_publisher(name):
    """Plot the top20 publishers' games' twitch stats from 2016-2020."""
    fig, ax = plt.subplots(figsize=(12, 10))
    bubbleplot = sns.scatterplot(data=top_games_twitch, x='Hours_watched',
                                 y='N_of_players', size='Hours_watched',
                                 hue='Team', alpha=0.5, sizes=(50, 1000))
    # Separate scatter plot for Activision Blizzard
    sns.scatterplot(data=top_games_twitch[top_games_twitch['Team'] ==
                                          'Activision Blizzard'],
                    x='Hours_watched', y='N_of_players',
                    size='Hours_watched', color='red', alpha=0.5,
                    sizes=(20, 30))
    plt.title('Game Viewing Statistics on Twitch for Top 20 Publishers',
              fontsize=15, fontweight='bold')
    plt.xlabel('Total Number of Hours Watched')
    plt.ylabel('Total Number of Players for a Game')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
               title='*Activision Blizzard games highlighted in red')
    plot = os.path.join(IMAGEPATH, f"{name}_bubbleplot.png")
    fig.savefig(plot, dpi=500, bbox_inches='tight')
    plt.show()


##############################################################################
# Shiny & Plotly Plot 3: Select year to display the top publishers ranked by
# Twitch hours watched
def get_twitchstats_by_year(df, year):
    """Select top 20 publishers based on total viewing hours."""
    publisher_year_aggregates = df.groupby(['Team', 'Year_x'])\
        .agg({'Hours_watched': 'sum', 'Avg_viewer_ratio': 'mean'})\
        .sort_values(by='Hours_watched', ascending=False).reset_index()
    top_publishers_year = publisher_year_aggregates[
        publisher_year_aggregates['Year_x'] == year].head(20)
    return top_publishers_year


def visualize_twitchstats_by_publisher_year_plotly(year, df):
    """Visualize Twitch stats for top 20 publishers' games in a given year.
    Args:
    ----
        df should be plotting.twitch_merge_exploded"""
    top_publishers_year = get_twitchstats_by_year(df, year)
    # Main scatter plot for all teams
    fig = px.scatter(top_publishers_year, x='Avg_viewer_ratio',
                     y='Hours_watched', size='Hours_watched', color='Team',
                     size_max=60, opacity=0.5,
                     title='Top 20 Publishers by Total Twitch Views in ' +
                     f'{year}, with Activision Blizzard highlighted')
    actblz_stats = top_publishers_year[top_publishers_year['Team'] ==
                                       'Activision Blizzard']
    # Highlight Activision Blizzard scatter in the plot
    fig.add_trace(go.Scatter(x=actblz_stats['Avg_viewer_ratio'],
                             y=actblz_stats['Hours_watched'],
                             mode='markers',
                             marker=dict(size=actblz_stats['Hours_watched'],
                                         sizemode='area',
                                         sizeref=2.*max(top_publishers_year[
                                             'Hours_watched'])/(60.**2),
                                         sizemin=4,
                                         color='red', line=dict(width=2)),
                             name='Activision Blizzard'))

    fig.update_layout(xaxis_title='Average Viewer Ratio',
                      yaxis_title='Average Hours Watched per Game this Year',
                      legend=dict(
                          title='Publisher', x=1, y=0.5,
                          xanchor='left', yanchor='middle'))
    fig.show(renderer='browser')

# Sample usage:
# visualize_twitchstats_by_publisher_year_plotly(2016, twitch_merge_exploded)

###############################################################################

# Function execution
# visualize_twitchstats_by_publisher('Figure5')


# %%%  Line plots of Activision Blizzard Top Games Twitch Viewing Stats

def visualize_actblz_twitchstats(name):
    """Plot Twitch viewing and streaming stats for ActBlz games."""
    fig, axs = plt.subplots(nrows=4, figsize=(14, 25))
    sns.lineplot(data=actblz_stats, x='datetime', y='Hours_watched',
                 hue='Game_same',
                 ax=axs[0]).set(
                     title='Activision Blizzard Games Hours Watched',
                     xlabel='Year', ylabel='Hours Watched (Hundred Millions)')
    axs[0].legend(title='Game')
    sns.lineplot(data=actblz_stats, x='datetime', y='Streamers',
                 hue='Game_same', ax=axs[1]).set(
                     title='Number of Streamers in Activision Blizzard Games',
                     xlabel='Year', ylabel='Streamers (Millions)')
    axs[1].legend(title='Game')
    sns.lineplot(data=actblz_stats, x='datetime', y='Avg_viewers',
                 hue='Game_same', ax=axs[2]).set(
                     title='Activision Blizzard Games Average Viewers',
                     xlabel='Year', ylabel='Average Viewers')
    axs[2].legend(title='Game')
    sns.lineplot(data=actblz_stats, x='datetime', y='Avg_viewer_ratio',
                 hue='Game_same', ax=axs[3]).set(
                     title='Activision Blizzard Games Average Viewer Ratio',
                     xlabel='Year', ylabel='Average Viewer Ratio')
    axs[3].legend(title='Game')
    fig.suptitle("Twitch Viewing and Streaming Statistics for " +
                 "Activision Blizzard Games, 2016-2023",
                 fontsize=15, fontweight='bold', y=0.9)
    lineplot = os.path.join(IMAGEPATH, f"{name}_lineplot.png")
    plt.savefig(lineplot, dpi=500, bbox_inches='tight')
    plt.show()


# visualize_actblz_twitchstats('Figure6')


# %% Section Text Processing Visualization on Game Reviews
# %%% Sub-Section Game Reviews WordCloud

positive_cleaned_text_keywords = textprocess.positive_cleaned_text_keywords
negative_cleaned_text_keywords = textprocess.negative_cleaned_text_keywords


def generate_wordcloud(text, title, name):
    """Generate and display WordCloud for a given text."""
    fig, axes = plt.subplots(figsize=(30, 15))
    wordcloud = textprocess.WordCloud(width=1000, height=600,
                                      color_func=custom_colors,
                                      background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Word Cloud for {title} Key Reviews Words", fontsize=15,
              fontweight='bold')
    plt.axis("off")
    wordcloud = os.path.join(IMAGEPATH, f"{name}_wordcloud.png")
    plt.savefig(wordcloud)
    plt.show()


# generate_wordcloud(text=positive_cleaned_text_keywords,
#                    title="Top 10% Rated Games'",
#                    name='Figure7_positivereview')
# generate_wordcloud(text=negative_cleaned_text_keywords,
#                    title="Bottom 10% Rated Games'",
#                    name='Figure8_negativereview')


# %%% Sub-Section Game Reviews Sentiment and Subjectivity Score Distribution

def visualize_sentiment_distribution(dataframe, name):
    """Visualize game reviews sentiment distribution with diff in ATVI."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=dataframe, x='Sentiment', bins=30, kde=True,
                 hue='actblz_indicator', multiple='stack', alpha=0.6)

    ax.set_title('Game Reviews Sentiment Score Distribution')
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Frequency')

    # Optional: Adding a legend to clarify the groups
    ax.legend(title='Activision Blizzard Indicator',
              labels=['Activision Blizzard', 'Other Publishers'])

    sentiment_plot = os.path.join(IMAGEPATH, f"{name}_sentiment_histplot.png")
    fig.savefig(sentiment_plot)
    plt.show()

# visualize_sentiment_distribution(dataframe=games, name='Figure9')
