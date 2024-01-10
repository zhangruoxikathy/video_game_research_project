# Shiny

###############################################################################
"""
In this .py file, we will create an interactive Shiny dashboard for three
visualizations:
    1. Bubble chart of top 20 publishers by total views in the user-chosen year
    2. Treemap of top 5 publishers by the user-chosen genre
    3. Bubble chart of annual average game rating by the user-chosen genre
"""
###############################################################################

# import packages
import plotly.express as px
import sys
import matplotlib.pyplot as plt
from shiny import App, render, ui
from shinywidgets import output_widget, render_widget
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go

PATH = r'/Users/bella/Documents/GitHub/final-project-our_code_works'
DATAPATH = r'/Users/bella/Documents/GitHub/final-project-our_code_works/data'
IMAGEPATH = r'/Users/bella/Documents/GitHub/final-project-our_code_works/image'

# %% Read Data from DataWrangling and TextProcessing Files

sys.path.append(PATH)
import DataCleaning as data
import TextProcessing as textprocess
import Plotting as plot

games = textprocess.games
sales = data.import_and_clean_sales()
gamesgenre_exploded, genre_annual_data, genre_data = data.\
    get_games_explode_and_groupby_genre()
twitch = data.import_and_clean_twitch()
twitch_merge, actblz_stats = textprocess.merge_data_for_regression(
    twitch_df=twitch)
twitch_merge_exploded = plot.twitch_merge_exploded

# %% Create an interactive Shiny to display plots

app_ui = ui.page_fluid(
    ui.h2("Final Project", align="middle"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.img(
                src="https://www.appam.org/assets/1/6/harris3.png",
                align='middle', width=250, height=100
            ),
            ui.column(5, ui.output_text("name")),
            ui.column(5, ui.input_select(
                id="year",
                label="Please choose a year",
                choices=[2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
                selected=2016
            )),
            ui.column(5, ui.input_select(id="salesgenre",
                                         label="Please choose a sales genre",
                                         choices=['Strategy', 'Fighting',
                                                  'Adventure', 'Racing',
                                                  'Action', 'Simulation',
                                                  'Puzzle', 'Shooter',
                                                  'Platform', 'Role-Playing',
                                                  'Sports', 'Misc'],
                                         selected='Strategy')),
            ui.column(5, ui.input_select(id="genre",
                                         label="Please choose a genre",
                                         choices=['Strategy', 'Fighting',
                                                  'Music',
                                                  'Arcade',
                                                  'Point-and-Click',
                                                  'RPG',
                                                  'Visual Novel',
                                                  'Racing',
                                                  'Shooter',
                                                  'Sport',
                                                  'Simulator',
                                                  'Brawler',
                                                  'Adventure',
                                                  'Puzzle',
                                                  'Platform',
                                                  'Tactical',
                                                  'Card & Board Game',
                                                  'Indie',
                                                  'Turn Based Strategy',
                                                  'MOBA'],
                                         selected="Strategy"))),
        ui.panel_main(
            ui.navset_card_tab(
                ui.nav("Publishers by Total Views", output_widget("left_plot"),
                       value="panel1"),
                ui.nav("Publishers per Genre", output_widget("middle_plot"),
                       value="panel2"),
                ui.nav("Annual Game Rating", ui.output_plot("right_plot"),
                       value="panel3"),
                id="inTabset",
            ),
        )))

def server(input, output, session):
    """Defines the server function for Shiny."""
    @output
    @render.text
    def name():
        """Displays student and course information."""
        return ('Python Ⅱ, Fall 2023, Kathy Zhang, Bella Huang, Hanbin Kim')

    @output
    @render_widget
    def left_plot():
        """Plot top 20 publishers by total views in the user-chosen year."""
        year = int(input.year())

        def get_twitchstats_by_year(df, year):
            """Select top 20 publishers based on total viewing hours."""
            publisher_year_aggregates = df.groupby(['Team', 'Year_x'])\
                .agg({'Hours_watched': 'sum', 'Avg_viewer_ratio': 'mean'})\
                .sort_values(by='Hours_watched', ascending=False).reset_index()
            top_publishers_year = publisher_year_aggregates[
                publisher_year_aggregates['Year_x'] == year].head(20)
            return top_publishers_year

        def visualize_twitchstats_by_publisher_year_plotly(year, df):
            """Generate bubble chart for top 20 publishers."""
            top_publishers_year = get_twitchstats_by_year(df, year)
            fig = px.scatter(top_publishers_year, x='Avg_viewer_ratio',
                             y='Hours_watched', size='Hours_watched',
                             color='Team',
                             size_max=60, opacity=0.5,
                             title='Top 20 Publishers by Total Views' +
                             f' in {year}')
            actblz_stats = top_publishers_year[top_publishers_year['Team'] ==
                                               'Activision Blizzard']

            fig.add_trace(go.Scatter(x=actblz_stats['Avg_viewer_ratio'],
                                     y=actblz_stats['Hours_watched'],
                                     mode='markers',
                                     marker=dict(size=actblz_stats[
                                         'Hours_watched'], sizemode='area',
                                         sizeref=2.*max(top_publishers_year[
                                             'Hours_watched'])/(60.**2),
                                         sizemin=4, color='red',
                                         line=dict(width=2)),
                                     name='Activision Blizzard'))

            fig.update_layout(xaxis_title='Average Viewer Ratio',
                              yaxis_title='Average Hours Watched',
                              legend=dict(
                                  title='*Activision Blizzard games ' +
                                  'highlighted in red', x=1, y=0.5,
                                  xanchor='left', yanchor='middle'))

            return fig

        ploty_plot = visualize_twitchstats_by_publisher_year_plotly(
            year, twitch_merge_exploded)
        return ploty_plot

    @output
    @render_widget
    def middle_plot():
        """Plot top 5 publishers by the user-chosen genre."""
        selected_genre = input.salesgenre()
        genre_pub_sales = sales.groupby(["Genre", "Publisher"])[[
            'Global_Sales']].sum().reset_index()

        def get_top_publishers(group):
            """Get top 5 publishers per genre based on global sales."""
            return group.nlargest(5, 'Global_Sales')

        def visualize_top_publishers_per_salesgenre(selected_genre):
            """Generate treemap for top 5 publishers within given genre."""
            top_pub_per_genre = genre_pub_sales.groupby('Genre')\
                .apply(get_top_publishers)
            top_pub_per_genre = top_pub_per_genre.reset_index(drop=True)

            genre_pub = sales.groupby(['Genre', 'Publisher'])[[
                'Global_Sales']].sum().sort_values(['Genre', 'Global_Sales'],
                                                   ascending=False)\
                .reset_index()
            genre_top5 = genre_pub.loc[genre_pub['Genre'].isin(
                [selected_genre])]\
                .reset_index(drop=True).head(5)
            print(genre_top5[['Publisher', 'Global_Sales']])

            fig, ax = plt.subplots(figsize=(15, 10))
            fig = px.treemap(top_pub_per_genre[top_pub_per_genre['Genre'] ==
                                               'Action'],
                             path=[px.Constant('Genres'), 'Genre',
                                   'Publisher'],
                             values='Global_Sales',
                             color='Publisher',
                             title='Treemap of Top 5 Publishers per Genre or' +
                             f' for {selected_genre}')
            fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))

            return fig

        top_publisher_plot = visualize_top_publishers_per_salesgenre(
            selected_genre)

        return top_publisher_plot

    @output
    @render.plot(alt="Rating by genre plot")
    def right_plot():
        """Plot annual average game rating by the user-chosen genre."""
        selected_genre = input.genre()

        def visualize_genre_stats_by_genre(selected_genre):
            fig, ax = plt.subplots(figsize=(15, 10))
            sns.scatterplot(data=genre_annual_data[genre_annual_data[
                'Genres'] == selected_genre], x='Year', y='Rating',
                size='N_of_players', hue='Genres', alpha=0.5, sizes=(50, 1000))
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                       title='Genres and Number of games')
            plt.ylabel('Average Game Rating')
            plt.xlabel('Year Published')
            plt.title("Avg Annual Game Rating by Genre, Sized by " +
                      "Counts per Year Published, 2005-2023", size=18)

            return ax

        rating_by_genre_plot = visualize_genre_stats_by_genre(selected_genre)

        return rating_by_genre_plot


app = App(app_ui, server)
