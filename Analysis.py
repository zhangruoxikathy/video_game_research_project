# Regression Analysis

###############################################################################
"""
In this .py file, we will conduct regression analysis towards our
assessment of whether Activision Blizzard brings potential for Microsoft in
the video gaming industry.

There are four regressions:
    1. Regression for Genre Popularity Analysis
    2. Regression for Number of Players based on Game Rating, Review Sentiments
    3. Regression for Twitch Popularity
        a. based on Monthly Average Hours Watched on Twitch
        b. based on Best Monthly Ranking a Game has Obtained on Twitch
"""
###############################################################################

# import packages
import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm


PATH = r'C:\Users\zhang\OneDrive\Documents\GitHub\final-project-our_code_works'
DATAPATH = r'C:\Users\zhang\OneDrive\Documents\GitHub\final-project-our_code_works\data'
IMAGEPATH = r'C:\Users\zhang\OneDrive\Documents\GitHub\final-project-our_code_works\image'
ANALYSISPATH = r'C:\Users\zhang\OneDrive\Documents\GitHub\final-project-our_code_works\regression_table_result'

# Define latex structure for regression output
beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\usepackage[margin=1in, left=0.5in]{geometry}
\\begin{document}"""
endtex = "\\end{document}"

# %% Read Data from DataCleaning, TextProcessing, Plotting Files

sys.path.append(PATH)
import DataCleaning as data
import Plotting as plot
import TextProcessing as textprocess
games = textprocess.games
sales = data.import_and_clean_sales()
twitch_merge_exploded = plot.twitch_merge_exploded
twitch_merge = plot.twitch_merge


# %% Regression for Genre Popularity Analysis, export to latex and pdf

def regress_sales_genres():
    """Regress global sales on genres and years to assess trends in genres.

    For games published from 2005-2017, with sales > 0.05M units until 2020.

    Independent variable:
    --------------------
        Global_Sales: Global sales

    Regressors:
    -----------
        genre_{genre}_year: Game genre in sales dataset*adjusted year
                            interaction terms,
        actblz_indicator: Activision Blizzard indicator
    """
    genre_salaes_dummies = pd.get_dummies(sales['Genre'])
    game_sales_reg = sales.copy()
    game_sales_reg['Year'] = game_sales_reg['Year'] - 2004
    game_sales_reg = game_sales_reg[game_sales_reg['Global_Sales'] > 0.05]
    # Produce interaction terms
    for genre in genre_salaes_dummies.columns:
        game_sales_reg[f'genre_{genre}_year'] = genre_salaes_dummies[genre]\
            * game_sales_reg['Year']
    X = game_sales_reg[[f'genre_{genre}_year' for genre in genre_salaes_dummies
                        .columns] + ['actblz_indicator']]
    y = game_sales_reg['Global_Sales']

    # Add a constant to the model and fit the regression model
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # output the regression result to latex and then pdf
    with open(os.path.join(ANALYSISPATH, 'Regression1_Result.tex'),
              'w') as file:
        file.write(beginningtex)
        file.write("\\section*{Regression 1, Genre Popularity " +
                   "Analysis, for Games Published from 2005-2017}\\bigskip\n")
        file.write("\\subsection*{INDEPENDENT VARIABLE Global " +
                   "sales}\\bigskip\n")
        file.write("\\subsection*{RERGESSORS genre*year interaction terms," +
                   " Activision Blizzard Indicator}\\bigskip\n")
        file.write(model.summary().as_latex())
        file.write(endtex)

    os.chdir(ANALYSISPATH)
    os.system("pdflatex Regression1_Result.tex")
    return model


model1 = regress_sales_genres()


# %%% Regression for Number of Players Analysis based on Rating and Sentiment,
#    export to latex and pdf

def regress_rating_actblz():
    """Regress number of players on relavant variables to assess indicators.

    For Twitch top 200 games viewing statistics  per month from 2016-2023.

    Independent variable:
    --------------------
        N_of_players_first: Number of players for a game

    Regressors:
    -----------
        Rating: Game rating, out of 5,
        Sentiment: Sentimental score of a game's reviews, the positive
                         the higher,
        actblz_indicator_first: Activision Blizzard indicator
    """
    X = games[['Rating', 'Sentiment', 'actblz_indicator']]
    y = games['N_of_players']
    # Add a constant to the model and fit the regression model
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # Output the regression result to latex and then pdf
    with open(os.path.join(ANALYSISPATH, 'Regression2_Result.tex'),
              'w') as file:
        file.write(beginningtex)
        file.write("\\section*{Regression 2, Number of Players for a Game" +
                   " on Game Ratings, Sentiment, for Game Published" +
                   "2005-2023}\\bigskip\n")
        file.write("\\subsection*{INDEPENDENT VARIABLE Number of players " +
                   "for a game}\\bigskip\n")
        file.write("\\subsection*{REGRESSORS Game rating out of 5, " +
                   "sentimental score on reviews, Activision Blizzard " +
                   "indicator}\\bigskip\n")
        file.write(model.summary().as_latex())
        file.write(endtex)

    os.system("pdflatex Regression2_Result.tex")
    return model


model2 = regress_rating_actblz()


# %% Regression Twitch Popularity Impact on Games, export to latex and pdf

def order_twitch_for_regression(df):
    """Perform groupby on twitch_merge to get game-based statistics."""
    twitch_game_merge = df.groupby('Game').agg({
        'Hours_watched': ['mean', 'max'],
        'Rank': ['mean', 'min'],
        'Hours_streamed': 'mean',
        'Peak_viewers': 'mean',
        'Peak_channels': 'mean',
        'Year_y': 'first',
        'N_of_players': 'first',
        'actblz_indicator': 'first',  # or use another suitable function
        'Sentiment': 'first'  # or use another suitable function
    }).reset_index()

    twitch_game_merge.columns = ['_'.join(col).strip() if col[1] else col[0]
                                 for col in twitch_game_merge.columns.values]
    twitch_game_merge['Age'] = 2024 - twitch_game_merge['Year_y_first']
    twitch_game_merge['log_Hours_watched_mean'] = np.log(
        twitch_game_merge['Hours_watched_mean'])
    twitch_game_merge['log_Peak_viewers_mean'] = np.log(
        twitch_game_merge['Peak_viewers_mean'])
    twitch_game_merge['log_Hours_watched_max'] = np.log(
        twitch_game_merge['Hours_watched_max'])

    return twitch_game_merge


twitch_game_merge = order_twitch_for_regression(twitch_merge)


# %%% Sub-Section Regression for Number of Players Analysis based on
#     Monthly Average Hours Watched on Twitch

def regress_players_twitch_watched():
    """Regress number of players on relavant variables to assess indicators.

    For Twitch top 200 games viewing statistics  per month from 2016-2023.

    Independent variable:
    --------------------
        N_of_players_first: Number of players for a game for monthly popular
        Twitch games

    Regressors:
    -----------
        log_Hours_watched_mean: Log of Twitch average hours watched for a game,
        Age: Number of years a game has been out until 2024,
        actblz_indicator_first: Activision Blizzard indicator,
        Sentiment_first: Sentimental score of a game's reviews, the positive
                         the higher
    """
    X = twitch_game_merge[['log_Hours_watched_mean', 'Age',
                           'actblz_indicator_first', 'Sentiment_first']]
    y = twitch_game_merge['N_of_players_first']
    # Add a constant to the model and fit the regression model
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # Output the regression result to latex and then pdf
    with open(os.path.join(ANALYSISPATH, 'Regression3_Result.tex'),
              'w') as file:
        file.write(beginningtex)
        file.write("\\section*{Regression 3, Number of Players for a Game" +
                   " on Averave Watching Hours on Twitch from " +
                   "2016-2023}\\bigskip\n")
        file.write("\\subsection*{INDEPENDENT VARIABLE Number of players " +
                   "for a game}\\bigskip\n")
        file.write("\\subsection*{REGRESSORS Log(Average hours watched for " +
                   "a game on Twitch monthly), game age, sentimental score " +
                   "on reviews, Activision Blizzard indicator}\\bigskip\n")
        file.write(model.summary().as_latex())
        file.write(endtex)

    os.system("pdflatex Regression3_Result.tex")
    return model


model3 = regress_players_twitch_watched()


# %%% Sub-Section Regression for Number of Players Analysis based on
#     Best Monthly Ranking a Game has Obtained on Twitch

def regress_players_twitch_ranks():
    """Regress number of players on relavant variables to assess indicators.

    For Twitch top 200 games viewing statistics  per month from 2016-2023.

    Independent variable:
    --------------------
        N_of_players_first: Number of players for monthly popular
        Twitch games

    Regressors:
    -----------
        Rank_min: Best monthly ranking of a game based on Twitch popularity,
        Age: Number of years a game has been out until 2024,
        actblz_indicator_first: Activision Blizzard indicator,
        Sentiment_first: Sentimental score of a game's reviews, the positive
                         the higher
    """
    X = twitch_game_merge[['Rank_min', 'Age',
                           'actblz_indicator_first', 'Sentiment_first']]
    y = twitch_game_merge['N_of_players_first']
    # Add a constant to the model and fit the regression model
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # Output the regression result to latex and then pdf
    with open(os.path.join(ANALYSISPATH, 'Regression4_Result.tex'),
              'w') as file:
        file.write(beginningtex)
        file.write("\\section*{Regression 4, Number of Players for a game on" +
                   " Twitch Ranking from 2016-2023}\\bigskip\n")
        file.write("\\subsection*{INDEPENDENT VARIABLE Number of players " +
                   "for a game}\\bigskip\n")
        file.write("\\subsection*{REGRESSORS Best monthly ranking of a game" +
                   " based on Twitch popularity, game age, sentimental score" +
                   " on reviews, Activision Blizzard indicator}\\bigskip\n")
        file.write("\\subsection*{Highest ranking of a game on Twitch, " +
                   "Game Age, Sentimental Score on Reviews, Activision " +
                   "Blizzard Indicator}\\bigskip\n")
        file.write(model.summary().as_latex())
        file.write(endtex)

    os.system("pdflatex Regression4_Result.tex")
    return model


model4 = regress_players_twitch_ranks()
