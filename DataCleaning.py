# Data Wrangling--Data Cleaning

###############################################################################
"""
We have in total of 11 datasets, in 6 categories:

    Direct downloads:
    1. Popular video games 1980-2023
    https://www.kaggle.com/datasets/arnabchaki/popular-video-games-1980-2023
    2. Video Games Global Sales 1980-2020
    https://www.kaggle.com/code/brandonlogue/video-games-global-sales/input
    3. Top games on Twitch 2016 - 2023
    https://www.kaggle.com/datasets/rankirsh/evolution-of-top-games-on-twitch

    Automatic retrieval (which are retrieved in this .py file):
    4. Private video game companies revenue, scraped from
    https://www-statista-com.proxy.uchicago.edu/statistics/298766/supercell
    -annual-revenue/
    https://www-statista-com.proxy.uchicago.edu/statistics/510591/glu-mobile
    -annual-revenue/
    https://www-statista-com.proxy.uchicago.edu/statistics/288974/king-annual
    -revenue/
    https://www-statista-com.proxy.uchicago.edu/statistics/1232495/sumo-group
    -annual-revenue/
    https://www-statista-com.proxy.uchicago.edu/statistics/273567/zyngas-annual
    -revenue/
    5. List of most expensive video game acquisitions, scraped from
    https://en.wikipedia.org/wiki/List_of_most_expensive_video_game
    _acquisitions
    6. Public comparable video game companies' financials--market cap, revenue,
    gross margin, ebitda, net income, R&D expense, using Yahoo!Finance
    packages, combining into one dataset

In this py file, we will clean all directly downloaded datasets listed as
 top three above for later analysis.

"""
###############################################################################

# import packages
import os
import pandas as pd
import datetime
import numpy as np
import datetime
import ast

PATH = r'C:\Users\zhang\OneDrive\Documents\GitHub\final-project-our_code_works'
DATAPATH = r'C:\Users\zhang\OneDrive\Documents\GitHub\final-project-our_code_works\data'


# %% Import Datasets and Conduct Initial Cleaning

def import_and_clean_games():
    """Produce cleaned games dataset."""
    games = pd.read_csv(os.path.join(DATAPATH, 'popular_vg_1980-2023.csv'),
                        encoding="cp1252")
    games = games[['Title', 'Release Date', 'Team', 'Rating',
                   'Number of Reviews', 'Genres', 'Summary', 'Reviews',
                   'Plays', 'Playing', 'Backlogs', 'Wishlist']]
    games = games.drop_duplicates(keep='first')
    games = games.dropna().reset_index(drop=True)
    games = games[games['Release Date'] != 'releases on TBD']
    games = games[games['Rating'] >= 2]

    # Convert K to 1000 for several columns for analysis usage
    def convert_k_to_int(x):
        """Convert it to *1000 if K in string."""
        if 'K' in x:
            return int(float(x.replace("K", "")) * 1000)
        else:
            return int(x)

    columns_to_convert = ['Number of Reviews', 'Plays', 'Playing', 'Backlogs',
                          'Wishlist']
    games[columns_to_convert] = games[columns_to_convert].applymap(
        convert_k_to_int)
    games['N_of_players'] = games['Plays'] + games['Backlogs'] + \
        games['Wishlist']

    # convert Release Date column to datetime
    games['Release Date'] = pd.to_datetime(games['Release Date'],
                                           format="%d-%b-%y")
    # Extract numerical components
    games['Year'] = games['Release Date'].dt.year
    games['Month'] = games['Release Date'].dt.month
    games['Day'] = games['Release Date'].dt.day
    games = games[games['Year'] >= 2005]

    def filter_dev_teams(row):
        return 'Activision' in row['Team'] or 'Blizzard' in row['Team']

    # Apply the filter
    actblz_games = games[games.apply(filter_dev_teams, axis=1)]
    games['actblz_indicator'] = games['Title'].\
        apply(lambda x: 1 if x in list(actblz_games['Title']) else 0)
    games['Publisher'] = np.where(games['actblz_indicator'] == 1,
                                  "Activision Blizzard", "Other Publishers")

    # Clean text and convert certain string columns to lists
    games = games[~(games['Title'].str.contains('ï¿½') | games['Team']
                    .str.contains('ï¿½'))]
    games['Genres'] = games['Genres'].apply(lambda x: ast.literal_eval(x))
    games['Team'] = games['Team'].apply(lambda x: ast.literal_eval(x))

    return games


def import_and_clean_sales():
    """Produce cleaned sales dataset."""
    sales = pd.read_csv(os.path.join(DATAPATH, 'vgsales.csv'),
                        encoding="cp1252")
    sales = sales[(sales['Year'] >= 2005) & (sales['Year'] < 2020)]

    # Add actblz_indicator column to sales dataframe
    condition = sales['Publisher'].str.contains('Activision|Blizzard',
                                                case=False,
                                                na=False)
    sales['actblz_indicator'] = np.where(condition, 1, 0)
    sales = sales.groupby('Name').agg({'Global_Sales': 'sum', 'Genre': 'first',
                                       'Publisher': 'first', 'Year': 'first',
                                       'actblz_indicator': 'first'})\
        .reset_index()

    return sales


def import_and_clean_twitch():
    """Produce cleaned twitch dataset."""
    twitch = pd.read_csv(os.path.join(DATAPATH, 'Twitch_game_data.csv'))
    twitch['datetime'] = twitch.apply(lambda row: datetime.datetime(
        row['Year'], row['Month'], 1), axis=1)
    return twitch


# Execute functions for initially cleaned datasets
games = import_and_clean_games()
twitch = import_and_clean_twitch()
sales = import_and_clean_sales()


# %%% Create Genre Datasets for Plotting and Analysis

def get_games_explode_and_groupby_genre():
    """Explode the games dataset genres and groupby for analysis."""
    # explode the games dataset by genre
    games = import_and_clean_games()
    gamesgenre_exploded = games.explode('Genres')

    # Group by genre by year
    genre_annual_data = gamesgenre_exploded.groupby(['Genres', 'Year'])\
        .agg({'Rating': 'mean', 'Title': 'count', 'N_of_players': 'mean'})\
        .sort_values(by='Year', ascending=False).reset_index()
    genre_annual_data = genre_annual_data[~genre_annual_data['Genres']
                                          .isin(['Pinball',
                                                 'Quiz/Trivia',
                                                 'Real Time Strategy'])]

    #  Group by genre
    average_ratings = pd.DataFrame(gamesgenre_exploded.groupby('Genres')
                                   ['Rating'].mean())
    total_games_per_genre = pd.DataFrame(gamesgenre_exploded['Genres'].
                                         value_counts())
    genre_data = average_ratings.merge(total_games_per_genre, on='Genres',
                                       how='inner')
    genre_data = genre_data.reset_index()

    return gamesgenre_exploded, genre_annual_data, genre_data


# Execute functions
gamesgenre_exploded, genre_annual_data, genre_data =\
    get_games_explode_and_groupby_genre()
