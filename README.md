# Leveling up: A Deep Dive into Microsoft’s Historical $69B Gaming Move
![Image of acquisition](https://techent.tv/wp-content/uploads/2022/12/image.png)
``
## Group Members:
- Kathy Zhang: zhangruoxikathy
- Bella Huang: bellahxy
- Hanbin Kim: hkim803

## File Management (most .py files are connected and run after each other):

- `DataRetrieval.py`: Retrieves data directly from web, which includes web scraping and text processing.

- `DataCleaning.py`: Cleans all gaming datasets for text processing, plotting, etc. 

- `TextProcessing.py`: Includes text processing on game reviews for word cloud and sentimental analysis, with additional cleaning just for text processing.
  
- `Plotting.py`: Includes plots using `games`, `sales`, `twitch` datasets using `DataCleaning.py` and `TextProcessing.py`, with additional cleaning just for plotting.
  
- `FinancialAnalysis.py`: Includes financial data manipulation and plots.
  
- `Shiny.py`: Converts several `Plotting.py` plots into Shiny with user-selected inputs (see three screenshots starts with Shiny in `image` folder).
  
- `Analysis.py`: Contains all regressions used for our analysis.

#### Input and Output Folders

- `data` folder contains all datasets directly downloaded from Kaggle and `*_rev`, `company_financials`, and `video_game_m&a_deals` are data retrieved using `DataRetrieval.py`. 

- `image` folder contains all output plots from `Plotting.py`, `FinancialAnalysis.py`, and screenshots from `Shiny.py`. `Figure5_bubbleplot_masterplot1 for shiny` and `masterplot2 for shiny` are masterplots for Shiny plots 1 & 2 without user inputs and hence show all genres or years data. 

- `regression_table_result` folder contains all printed regression summary tables from `Analysis.py` 
  

### Introduction
Microsoft's acquisition of Activision Blizzard (Nasdaq: ATVI), a prominent publisher known for multi-billion-dollar video game franchises like Call of Duty, represents a historic moment in the industry with a transaction value of $69 billion. This move, marking the largest acquisition in video game history, positions Microsoft to become the third-largest gaming company globally by revenue. This paper examines the strategic rationale and financial justification behind Microsoft's acquisition, particularly considering the high price tag. We conduct comparative financial analysis, and delve into industry trends to evaluate Activision Blizzard(ATVI)'s alignment with industry potentials and Microsoft's strategic goals, with Python-based research method detailed in this write-up.

### Data Retrieval & Cleaning

Our research utilizes 11 datasets, 3 of which downloaded directly from Kaggle, and the rest–the private comps financials, the deal sizes, the public comps financials scraped from Statista, Wikipedia public info, and Yahoo!Finance respectively in the DataRetrieval file. The first Kaggle dataset `games` encompasses statistics for video games published between 1980 and 2023, detailing publishers, reviews, ratings, and player counts. The second dataset `sales` tracks global sales of video games from 1980 to 2020, categorized by genre and publisher, and quantified in units as of 2020. The third dataset `twitch` provides streaming data for the top 200 games monthly from 2016 to 2023. Data cleansing involved removing duplicates, converting date-time formats to years, transforming strings to lists for column expansion, standardizing player count notations (e.g., converting 'xxk' to 'xx000'), filtering post-2005 data, and assigning an actblz indicator to games published by Activision Blizzard.

### Text Processing

Our sentiment and word cloud analyses on game reviews employed the re package for data cleaning, filtering reviews based on game rating percentiles (top and bottom 10%). After removing common nouns like `game` and 'time' for clearer word cloud generation, we noted an unexpectedly positive sentiment even towards bottom 10%-rated games, suggesting a potential bias in the data (`Figures 8 & 9`). 
Additionally, our review sentiment analysis revealed no significant correlation to player count ('Exhibit Regression 2'), despite the sentiment score's high negative skew--gamers mostly left very positive or very negative comments (`Figure 9`). The sentiment split for Activision Blizzard games was 24% negative and 73% positive, compared to the 15% negative and 84% positive for non-Activision Blizzard games, indicating no distinct advantage for ATVI in game reviews.

###  Plotting & Analysis (All of our regression table outputs are printed to pdf with latex)

##### Financial Analysis: The Deal and the Comparables

Across $1bn+ M&A deals in the video game industry that we scraped from wikipedia, Activision Blizzard has a significantly higher deal size/revenue multiple of 9.2x, only lower than the 13x multiple of Tencent’s the 2021 acquisition of Sumo (`Figure 10`). When juxtaposed with leading video game companies like EA, UBSFY, TTTWO, and RBLX between 2019 and 2022, Activision Blizzard exhibits the highest market capitalization and revenue, yet it records the lowest income margin  (`Figure 11`). Notably, the company's minimal R&D investment, a key success factor in the gaming industry, correlates with its profit margins (`Figure 12`). This suggests that while Activision Blizzard commands substantial market shares and potential, it faces operational challenges.

##### The Video Game Industry

Global sales and the number of players for a game are two variables suitable for objective assessment of a game’s performance, and we aim to select variables to explain these variables through the below regressions. Finally, see if Activision Blizzard reveals strength in these regressor variables.

##### Overview of the Market and Activision Blizzard games based on plots

The distribution of number of players for a game has a fat tail with far fewer games having higher plays, and we can observe from Figure 1 that Activision Blizzard has several games with more than 10,000 units of players over the years, indeed an industry leader. In terms of game rating, it is clear that even lowly-rated Activision Blizzard games have a comparatively higher number of players (`Figure 2`). The steeper slope for AVTI in the comparative analysis in `Figure 3` with two regressions reveals that with the same amount of increase in rating, AVTI games have a  higher increase in players. 

Adventure, tactical, brawler, and platform genre games 

The `games` and `sales` datasets feature differing genre categorizations. Our genre analysis on games genre (`Figure 4`) indicates that while categories like Adventure, Strategy, and Indie consistently produce games each year, board games, Sport genres do so less frequently. The heatmap of the average number of players per genre reveals an inverse relationship: more populated genres tend to have lower average player counts. This could suggest a monopolistic presence in certain genres or reflect intense competition in popular ones. Conversely, less populated genres show higher average player numbers. The game rating heatmap exhibits fluctuating or stable trends across genres, with less predictability. Predominantly populated genres attract smaller, lower-cost games. For a major player like Activision Blizzard, the profitability of entering either a crowded or sparse genre is not straightforwardly determinable.

(`Exhibit Regression 1`) Our analysis of the video game sales dataset, employing linear regression on game sales > 50,000 all platforms with games published from 2005 to 2017, revealed significant positive trends in 5 sales genres over time–Action, Platform, Racing, Shooter, and Sports. This was observed through interaction terms combining each genre with its adjusted publication year (each year term = year - 2004), and a dummy variable indicating Activision Blizzard production. The coefficient 'genre_Action_year' at 0.0431 suggests increasing popularity and sales for future Action games, by 0.0431M units on average annually. Additionally, the 'actblz_indicator' being positive indicates that Activision Blizzard games generally outperform others in sales. Notably, in the top genres ranked by the interaction term, Activision Blizzard stands out as a leading publisher in three–Action, Platform, Shooter (`Masterplot1` and `Shiny Plot 1`). We have used Shiny and Plotly to filter and visualize a treemap that displays top 5 publishers with sales in a user-chosen genre. 

(`Exhibit Regression 2`) While the model highlights trends in popular genres, a further regression analysis was conducted to assess individual game popularity, with the number of players regressing on game ratings, review sentiment scores, and an Activision Blizzard dummy indicator. The positive ATVI indicator suggests that ATVI games typically attract more players. However, only the game rating coefficient proved statistically significant (p-value < 0.05), and this lies in Activision Blizzard's weakness, with average rating of 0.31 lower than the market average of 3.38 out of 5. 

(`Exhibit Regression 3&4`) In our final analysis, we merged the games and Twitch datasets, focusing on the average and maximum monthly hours watched, and the average and minimum game ratings. We also calculated each game's 'Age'. Regression 3 explores the relationship between the number of players and the logarithm of average monthly hours watched, while Regression 4 examines this relationship against the best monthly ranking, game age, Activision Blizzard (ATVI) dummy, and review sentiment score. Notably, the coefficients for best ranking (-49.4) and log(average monthly hours watched) (1619.8) are statistically significant. This indicates that a one-rank increase boosts player count by 49.4, and higher streaming popularity similarly increases player numbers. The 'Age' coefficient's significance aligns with the expectation that older games attract more players. Interestingly, ATVI games exhibit similar trends, with the actblz indicator coefficients being insignificant. Our next step is to analyze the recent performance of ATVI games on Twitch. Notably, Model 3 delivers 10.4% and Model 4 delivers 17.8% R-squared, providing better explaining power. 

Activision Blizzard ranks among the top five most-viewed publishers on Twitch from 2016 to 2023, with several games achieving top 15 status in terms of viewership, trailing behind industry leaders like Riot Games with League of Legends and Epic Games with Fortnite (`Figure 5`). Our first interactive Plotly visualization on Shiny enables users to explore the top Twitch publishers by viewership annually. This analysis reveals that Activision Blizzard has previously ascended to the second position in viewership rankings. Additionally, `Figure 6` illustrates the streaming and viewing trends of specific Activision Blizzard titles. Particularly, many of their games enjoy high popularity, with a recent surge in viewership for the newly released Overwatch 2. This trend underscores the significant impact and following that Activision Blizzard games garner upon release.

### Conclusion, Research Risks, and Concerns

Our regression analysis and trend examination reveal that Activision Blizzard exhibits strength in key game genres, an ability to attract a substantial player base, and sustained popularity on Twitch. Financially, the $69 billion acquisition price appears justifiable given the company's substantial market capitalization and revenue. Consequently, we conclude that this acquisition could be a strategically sound investment for Microsoft in the gaming sector. However, concerns arise from the consistently low ratings and negative reviews of Activision Blizzard's games. In the long term, these factors, coupled with operational risks (notably lower margins) and potential integration challenges, ATVI’s ability to succeed on Microsoft’ Xbox, may impact the company's continued success.

In our research, challenges arose in areas such as data retrieval, data matching, variable selection, and model choice. Data retrieval was particularly problematic due to the scarcity of private company data and the absence of API access, which we mitigated through web scraping. Additionally, inconsistencies in genre and publisher categorizations, outdated information, and the absence of detailed financial metrics such as dollar sales or unit prices, complicated our analysis of the financial impact of game sales. To enhance future research, it may be beneficial to directly source data from gaming platforms like Steam and to consider disaggregating data by platform for more comprehensive analysis.
