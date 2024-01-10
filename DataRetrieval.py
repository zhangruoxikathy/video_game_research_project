# Data Wrangling--Automatic Data Retrieval

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

In this .py file, we will retrive data for later analysis.
"""
###############################################################################

# import packages
import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from yahoofinancials import YahooFinancials
import pandas as pd
import datetime
import numpy as np
import ast
from bs4 import BeautifulSoup
from selenium import webdriver
from IPython.core.display import HTML

PATH = r'C:\Users\zhang\OneDrive\Documents\GitHub\final-project-our_code_works'
DATAPATH = r'C:\Users\zhang\OneDrive\Documents\GitHub\final-project-our_code_works\data'


# %% Section Data Retrieval
# %%% Sub-Section Private Video Game Companies Revenue Web Scraping from Statista

def scrape_statista_revenue(company):
    """Scrape company revenue data from Statista."""
    filename = os.path.join(DATAPATH, f'{company}_rev.csv')
    # Check if the file already exists, if so, read the file
    if os.path.exists(filename):
        print(f"Data for {company} already downloaded.")
        revenue_df = pd.read_csv(filename)
    else:
        html = driver.page_source
        soup = BeautifulSoup(html, "lxml")
        table = soup.find_all('tbody')[0]
        rows = []
        for row in table.find_all('td'):
            rows.append([row.text])

        years = []
        revenue = []
        for i in range(len(rows)):
            # Extract the first element, strip whitespace, and remove commas
            cleaned_item = rows[i][0].strip().replace(',', '').replace('*', '')
            integer_item = float(cleaned_item)
            # Alternate between years and values
            if i % 2 == 0:  # year
                years.append(integer_item)
            else:
                revenue.append(integer_item)

        # Reverse the lists to get them in ascending order
        years.reverse()
        revenue.reverse()
        revenue_df = pd.DataFrame({
            "Year": years,
            "Revenue": revenue})
        revenue_df.to_csv(filename, index=False, header=True)

    return revenue_df


if os.path.exists(os.path.join(DATAPATH, 'supercell_rev.csv')) is not True:
    driver = webdriver.Chrome()
    driver.get("https://www-statista-com.proxy.uchicago.edu/statistics/298766/supercell-annual-revenue/")
supercell_revenue = scrape_statista_revenue('supercell')


if os.path.exists(os.path.join(DATAPATH, 'glumobile_rev.csv')) is not True:
    driver.get("https://www-statista-com.proxy.uchicago.edu/statistics/510591/glu-mobile-annual-revenue/")
glumobile_revenue = scrape_statista_revenue('glumobile')

if os.path.exists(os.path.join(DATAPATH, 'king_rev.csv')) is not True:
    driver.get("https://www-statista-com.proxy.uchicago.edu/statistics/288974/king-annual-revenue/")
king_revenue = scrape_statista_revenue('king')

if os.path.exists(os.path.join(DATAPATH, 'sumo_rev.csv')) is not True:
    driver.get("https://www-statista-com.proxy.uchicago.edu/statistics/1232495/sumo-group-annual-revenue/")
sumo_revenue = scrape_statista_revenue('sumo')

if os.path.exists(os.path.join(DATAPATH, 'synga_rev.csv')) is not True:
    driver.get("https://www-statista-com.proxy.uchicago.edu/statistics/273567/zyngas-annual-revenue/")
    synga_revenue = scrape_statista_revenue('synga')


# %%% Sub-Section Most Expensive Video Game Acquisitions Web Scraping from Wikipedia

def scrape_wikipedia_table(url):
    """Scrape the video game m&a deals wikipeida table."""
    filename = os.path.join(DATAPATH, 'video_game_m&a_deals.csv')
    # Check if the file already exists, if so, read the file
    if os.path.exists(filename):
        print("Data for wiki table already downloaded.")
        madeal_df = pd.read_csv(filename)
    else:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': "wikitable"})

        # Get a list of column names
        columns = []
        for column in table.find_all('tr')[0].find_all('th'):
            columns.append([column.text])
        # Get values for each column and create a dataframe
        acquirer = []
        target = []
        year = []
        deal_value = []
        inflation_adjusted = []
        for row in table.find_all('tr')[1:]:  # Skip the header row
            cells = row.find_all('td')
            acquirer.append(cells[0].get_text(strip=True))
            target.append(cells[1].get_text(strip=True))
            year.append(int(cells[2].get_text(strip=True).replace(',', '')))
            deal_value.append(float(cells[3].get_text(strip=True).
                                    replace(',', '')))
            inflation_adjusted.append(float(cells[4].get_text(strip=True).
                                            replace(',', '')))

        madeal_df = pd.DataFrame({
            columns[0][0]: acquirer,
            columns[1][0]: target,
            columns[2][0]: year,
            columns[3][0]: deal_value,
            columns[4][0]: inflation_adjusted})

        madeal_df.to_csv(filename, index=False, header=True)

    return madeal_df


url = "https://en.wikipedia.org/wiki/List_of_most_expensive_video_game_acquisitions"
video_game_ma_deals = scrape_wikipedia_table(url)


# %%% Sub-Section Financial Data Retrieval from Yahoo!Finance
# https://www.geeksforgeeks.org/get-financial-data-from-yahoo-finance-with-python/
# https://pypi.org/project/yahoofinancials/

# Revenue, Net income, Gross Margin, R&D Expense ratio, Market cap


def get_yahoofinance_financials(tickers, date_of_interest):
    """Extract the financials for a given list of tickers for given dates."""
    filename = os.path.join(DATAPATH, 'company_financials.csv')
    # Check if the file already exists, if so, read the file
    if os.path.exists(filename):
        print("Data for financials already downloaded.")
        financials_df = pd.read_csv(filename)
    else:
        financials_df = [['Company', 'Year', 'TotalRevenue', 'GrossProfit',
                          'EBITDA', 'NetIncome', 'R&DExpense', 'MarketCap']]
        financials = YahooFinancials(tickers)
        market_cap = financials.get_market_cap()

        for ticker in tickers:
            income_statement = financials.get_financial_stmts('annual',
                                                              'income')\
                ['incomeStatementHistory'][ticker]
            for year in income_statement:
                for date, details in year.items():
                    if date in date_of_interest:
                        if all(key in details for
                               key in ['totalRevenue', 'grossProfit', 'eBITDA',
                                       'netIncome', 'researchAndDevelopment']):
                            row = [ticker, date, details['grossProfit'],
                                   details['netIncome'], details['eBITDA'],
                                   details['totalRevenue'],
                                   details['researchAndDevelopment'],
                                   market_cap[ticker]]
                            financials_df.append(row)
        financials_df = pd.DataFrame(financials_df[1:],
                                     columns=financials_df[0])
        financials_df.to_csv(filename, index=False, header=True)

        return financials_df


date_of_interest_list = ['2020-03-31', '2021-03-31', '2022-03-31',
                         '2023-03-31', '2019-12-31', '2020-12-31',
                         '2021-12-31', '2022-12-31']
tickers_list = ['EA', 'UBSFY', 'TTWO', 'NTDOY', 'RBLX']  # ATVI from macrotrend
company_financials = get_yahoofinance_financials(tickers_list,
                                                 date_of_interest_list)


# %% Section Import Data and Conduct Initial Cleansing

# Tools for data exploration
# games.shape
# games.columns
# games.info()
# games.isna().any()


def import_and_clean_games():
    games = pd.read_csv(os.path.join(DATAPATH, 'popular_vg_1980-2023.csv'),
                        encoding="cp1252")
    games = games[['Title', 'Release Date', 'Team', 'Rating',
                   'Number of Reviews', 'Genres', 'Summary', 'Reviews',
                   'Plays', 'Playing', 'Backlogs', 'Wishlist']]
    # duplicate = games[games.duplicated()]  # explore duplicates
    games = games.drop_duplicates(keep='first')
    games = games.dropna().reset_index(drop=True)
    games = games[games['Release Date'] != 'releases on TBD']
    games = games[games['Rating'] >= 2]

    # convert K to 1000 for several columns
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

    games['Genres'] = games['Genres'].apply(lambda x: ast.literal_eval(x))

    return games


games = import_and_clean_games()


def import_and_clean_sales():
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


sales = import_and_clean_sales()


def import_and_clean_twitch():
    twitch = pd.read_csv(os.path.join(DATAPATH, 'Twitch_game_data.csv'))
    twitch['datetime'] = twitch.apply(lambda row: datetime.datetime(
        row['Year'], row['Month'], 1), axis=1)
    return twitch


twitch = import_and_clean_twitch()


# %%% Section Create Genre Datasets for Plotting and Analysis

def get_games_explode_and_groupby_genre():

    # explode the games dataset by genre
    games = import_and_clean_games()
    gamesgenre_exploded = games.explode('Genres')

    # Group by genre by year
    average_annual_ratings = pd.DataFrame(gamesgenre_exploded
                                          .groupby(['Genres', 'Year'])
                                          ['Rating'].mean())
    average_annual_count_per_genre = pd.DataFrame(gamesgenre_exploded[
        ['Genres', 'Year']].value_counts())
    genre_annual_data = average_annual_ratings.merge(
        average_annual_count_per_genre, on=['Genres', 'Year'], how='inner')
    genre_annual_data = genre_annual_data.reset_index()

    genre_annual_data['Year'] = genre_annual_data['Year']\
        .apply(lambda x: int(x))
    genre_annual_data = genre_annual_data[~genre_annual_data['Genres']
                                          .isin(['MOBA', 'Pinball',
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


gamesgenre_exploded, genre_annual_data, genre_data =\
    get_games_explode_and_groupby_genre()


# %% Section Per person sales and genre analysis

# Merge sales and gaming datasets
# game_sales_merge = games.merge(sales, left_on='Title',
#                                right_on='Name', how='inner')
# game_sales_merge['PP_Sales'] = game_sales_merge[
#     'Global_Sales']/game_sales_merge['Plays']


# # Run regression of per person global sales on interaction term (genre * year)
# genre_sales_dummies = pd.get_dummies(game_sales_merge['Genre'])
# game_sales_reg = game_sales_merge.copy()
# for genre in genre_sales_dummies.columns:
#     game_sales_reg[f'genre_{genre}_year'] = genre_sales_dummies[genre]\
#         * game_sales_reg['Year_y']
# X = game_sales_reg[[f'genre_{genre}_year' for genre in genre_sales_dummies.
#                     columns]]
# y = game_sales_reg['PP_Sales']

# # Add a constant to the model (intercept)
# X = sm.add_constant(X)

# # Fit the regression model
# model = sm.OLS(y, X).fit()

# # Print the summary of the regression
# print(model.summary())

# no statistically significant difference between genre and per person sales

# %% Section Initial Plots

# set the custom color palette
# http://seaborn.pydata.org/tutorial/color_palettes.html
# sns.color_palette("mako", as_cmap=True)
# custom_palette = ["#ff6700", "#9400d3"]
# sns.set_palette(custom_palette)

# # plot the ratinsg density between activision blizzard and other games
# sns.set(style="white")
# sns.histplot(data=games, x="Rating", hue="actblz_indicator", stat='density',
#              multiple="stack", palette=custom_palette, kde=True)
# plt.show()

# # plot the relationship between rating and plays
# plt.figure(figsize=(10,6))
# sns.regplot(x="Rating", y="Plays", data=games)
# plt.show()

# # plot the relationship between year published and rating
# plt.figure(figsize=(10,6))
# sns.regplot(x="Year", y="Rating", data=games)
# plt.show()


# sns.pairplot(games[['Rating', 'Plays']])
# sns.jointplot(x = "Plays", y = "Rating", data=games, kind="hex")

# # %% Section Genre Analysis
# # %%% Sub-Section Genre plot

# # Bubble plot with year as x-axis, rating as y-axis, bubble size as
# # count per genre, color of the bubble as genre
# fig, ax = plt.subplots(figsize=(15, 10))
# bubble_plot = sns.scatterplot(
#     data=genre_annual_data,
#     x='Year',
#     y='Rating',
#     size='count',  # bubble sizes based on countsof games per genre per year
#     hue='Genres',
#     alpha=0.5,     # transparency of bubbles
#     sizes=(50, 1500)  # range of bubble sizes
# )
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=
#            'Genres and Number of games')
# # plt.xlabel("X", size=16)
# # plt.ylabel("y", size=16)
# plt.title("Avg Annual Game Rating by genre, sized by counts per year", size=18)
# bubbleplot_plot = os.path.join(PATH, "bubbleplot.png")
# fig.savefig(bubbleplot_plot)
# plt.show()


# # Average annual rating per year by genre
# genre_annual_rating = genre_annual_data.pivot(index="Genres", columns="Year",
#                                               values="Rating")
# fig, ax = plt.subplots(figsize=(15, 10))
# sns.heatmap(genre_annual_rating, annot=True)
# plt.title('Genre Counts')
# plt.xlabel('Genre')
# plt.ylabel('Count')
# plt.show()

# # joint plot of count and rating
# g = sns.JointGrid(x="count", y="Rating", data=genre_data)
# g = g.plot(sns.regplot, sns.distplot)

# # plot number of games in each genre
# sorted_count_genre = genre_data.sort_values(by='count', ascending=False)
# sns.set_style('white')
# ax = sns.barplot(x=sorted_count_genre['Genres'], y=sorted_count_genre['count'])
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
# plt.title('Genre Counts')
# plt.xlabel('Genre')
# plt.ylabel('Count')
# plt.show()

# ###### Get sales per genre
# # genre_annual_sales = game_sales_merge_genre_exploded.groupby(["Genres", "Year"])\
# #     [['Global_Sales']].sum().reset_index()
# # genre_annual_sales = genre_annual_sales.pivot(index="Genres", columns="Year",
# #                                               values="Global_Sales")
# # fig, ax = plt.subplots(figsize=(15, 10))
# # sns.heatmap(genre_annual_sales, annot=False)
# # plt.title('Genre Counts')
# # plt.xlabel('Genre')
# # plt.ylabel('Total Sales')
# # plt.show()

# # Plotly treemap that shows top five publishers with their sales per genre
# # Sales per publisher per genre

# genre_pub_sales = sales.groupby(["Genre", "Publisher"])[['Global_Sales']]\
#     .sum().reset_index()


# # Function to get top 5 publishers per genre
# def get_top_publishers(group):
#     """Get top 5 publishers per genre."""
#     return group.nlargest(5, 'Global_Sales')


# # Apply the function to each genre group
# top_pub_per_genre = genre_pub_sales.groupby('Genre').apply(get_top_publishers)
# top_pub_per_genre = top_pub_per_genre.reset_index(drop=True)

# # Plot the treemap
# fig, ax = plt.subplots(figsize=(15, 10))
# fig = px.treemap(top_pub_per_genre,
#                  path=[px.Constant('All Genres'), 'Genre', 'Publisher'],
#                  values='Global_Sales',
#                  color='Publisher',
#                  title='Treemap of Top Publishers per Genre')
# fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
# fig.show()

# # Activision Blizzard is one of the top 5 publishers in acttion, shooter,
# # role-playing, platform, misc, strategy


# # Plotly and Seaborn line plots that shows global video game sales by genre
# # over years

# # Sales per genre by year
# genre_sales = sales.groupby(["Genre", "Year"])[['Global_Sales']]\
#     .sum().reset_index()

# # in plotly
# fig, axes = figsize=(20, 20)
# fig = px.line(genre_sales, x='Year', y='Global_Sales', color='Genre',
#               title='Global Video Game Sales by Genre Over Years')
# fig.update_layout(xaxis_title='Year', yaxis_title='Global_Sales',
#                   legend_title='Genre')
# fig.show()

# # in sns
# fig, ax = plt.subplots()
# sns.lineplot(data=genre_sales, x='Year', y='Global_Sales', hue='Genre')
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=
#            'Genres and Number of games')
# ax.set_title('Global Video Game Sales by Genre Over Years')
# ax.set_ylabel('Global Sales')
# ax.set_xlabel('Year')
# fig.show()


# # get top 5 publishers per popular genre
# genre_pub = sales.groupby(['Genre', 'Publisher'])[['Global_Sales']]\
#     .sum().sort_values(['Genre', 'Global_Sales'], ascending=False)\
#         .reset_index()
# strategy_top5 = genre_pub.loc[genre_pub['Genre'].isin(['Strategy'])]\
#     .reset_index(drop=True).head(5)
# print('Top 5 Strategy:')
# print(strategy_top5['Publisher'])
# print('------------------------------------------------')
