# Financial Analysis

###############################################################################
"""
In this file, we will conduct all the cleaning and plotting associated with
financial analysis based on data scraped from Statista and Wikipedia and
 extracted from Yahoo!Finance package.


## Input: Data/company_financials.csv, *_rev.csv, video_game_m&a_deals.csv
## Output: Image/financial_analysis_*.png, Data/edit/*_rev.csv

"""
###############################################################################

# import packages and setup
import os
import sys
import random
import pandas as pd
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.api as sm
from statsmodels.formula.api import ols
import datetime
from matplotlib.ticker import MaxNLocator
import plotly.express as px
import plotly.io as pio
import pandas as pd
from wordcloud import WordCloud
import ast
from IPython.core.display import HTML
from fuzzywuzzy import fuzz


PATH = "C:/Users/hkim803/datasci/final-project-our_code_works"
DATAPATH = "C:/Users/hkim803/datasci/final-project-our_code_works/data"
IMAGEPATH = "C:/Users/hkim803/datasci/final-project-our_code_works/image"


# %% 1. Financials summaries for public firms

### Functions for financial summaries and visuals

def financials_invsummary(df):
    '''Creates summary statistics for public firms'''
    df['Year'] = df.apply(lambda row: pd.to_datetime(row['Year']).year,
                          axis=1)
    df['yearno'] = df.groupby('Company')['Year'].transform(lambda x:
                                                           x - x.max()-1)
    df['Income/Revenue'] = df['NetIncome'] / df['TotalRevenue']
    df['ResearchInvestment'] = df['R&DExpense'] / df['TotalRevenue']
    return(df)


def financials_plot(df,path=IMAGEPATH):
    '''Creates plots for public firms financial data, fitted line option available.'''
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    colors = sns.color_palette()

    sns.lineplot(x='Year', y='TotalRevenue', hue='Company',
                 data=df, ax=axes[0, 0]).set_title("Total Revenue")
    sns.lineplot(x='Year', y='Income/Revenue', hue='Company',
                 data=df, ax=axes[0, 1]).set_title("Income/Revenue")
    sns.lineplot(x='Year', y='ResearchInvestment', hue='Company',
                 data=df, ax=axes[1, 0]).set_title(
                     "Research Investment Proportion")
    sns.barplot(x='Company', y='MarketCap',
                data=df[df['yearno'] == -1], ax=axes[1, 1])\
        .set_title("Market Capital for 2023")

    for ax in axes.flatten():
        ax.tick_params(axis='x', rotation=45)

    for ax in axes[0, :]:
        ax.set_xticks(df['Year'].unique())
    axes[1, 0].set_xticks(df['Year'].unique())

    plt.suptitle('Financial Metrics for Public Firms with >$1b M&A Deals')
    plt.tight_layout()

    os.chdir(IMAGEPATH)
    plt.savefig(f'financial_analysis_publicfirms.png')
    plt.show()


# %% 2. Public financials analysis

os.chdir(DATAPATH)
data = pd.read_csv("company_financials.csv", index_col = None)
financials_invsummary(data)
financials_plot(data)

fig, ax = plt.subplots(figsize=(10, 6))
color = sns.color_palette()[0]
sns.regplot(x='ResearchInvestment', y='Income/Revenue', data=data, 
            scatter=True, color=color, ax=ax)
plt.title('Research Investment vs. Income/Revenue')
plt.xlabel('Research Investment Proportion')
plt.ylabel('Income/Revenue Proportion')
os.chdir(IMAGEPATH)
plt.savefig('financial_analyses_researchinvandpm.png')
plt.show()

print("complete with financials analysis for public firms")


# %% 3. Financials analysis for private firms, acquired

### Functions for firm identification and private firm analysis

def file_identification(directory):
    '''find private revenue files in the given data directory'''
    file_names = os.listdir(directory)
    rev_files = [file_name for file_name in file_names if "_rev" in file_name]
    rev_firms = [file_name.replace('_rev.csv', '') for file_name in rev_files
                 if '_rev.csv' in file_name]
    return(rev_files, rev_firms)


def private_mna(data, private_list, threshold=80):
    '''find which firms with m&a transactions records are in the private revenue directory'''
    data['Target'] = data['Target'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
    
    def match_names(name, name_list, threshold):
        '''name match algorithm for private revenue directory and m&a file'''
        name_parts = name.lower().split()
        if len(name_parts) == 1:
            for stripped_name in name_list:
                if fuzz.ratio(name_parts[0],
                              stripped_name.lower()) >= threshold:
                    return stripped_name
            return None
        elif len(name_parts) == 2:
            for stripped_name in name_list:
                if (
                    fuzz.ratio(name_parts[0],
                               stripped_name.lower()) >= threshold or
                    fuzz.ratio(name_parts[1],
                               stripped_name.lower()) >= threshold
                ):
                    return stripped_name
            return None
        else:
            return None

    data['MatchedName'] = data['Target'].apply(
        lambda x: match_names(x, private_list, threshold))
    filtered_data = data[data['MatchedName'].notnull()]

    return filtered_data


def scrape_revenue_mnaprior(data, filelist, gbpval):
    '''combine data for revenue per deal year,
    edit raw datasets to reflect year distance from acquisition
    output: data/edit files'''
    os.chdir(DATAPATH)
    privaterevenue = pd.DataFrame()
    
    for file in filelist:
        privatefirm = pd.read_csv(file, index_col=None)
        firmname = file.replace('_rev.csv', '')  # Fix the variable name
        mask = data[data['MatchedName'] == firmname]
        
        if not mask.empty:
            yearval = mask['Year'].iloc[0] - 1
            privatefirm['yearno'] = privatefirm['Year'] - yearval - 1
            privatefirm.to_csv(f"edit/edit{file}", index = None)
            revn = privatefirm[privatefirm['Year'] == yearval]
            revn = revn.copy()
            revn['MatchedName'] = firmname
            privaterevenue = pd.concat([privaterevenue, revn])
            
    privaterevenue['Revenue'] = privaterevenue['Revenue']* 1000000
    privaterevenue.loc[privaterevenue['MatchedName'] ==
                       'sumo', 'Revenue'] *= gbpval 
    privaterevenue = privaterevenue.drop(columns = 'Year')
    privaterevenue = pd.merge(privaterevenue, data, on='MatchedName',
                              how='inner')
    privaterevenue['DealProp'] = privaterevenue['Deal value (US$)'
                                                ]/privaterevenue['Revenue']
    
    return privaterevenue


def private_preacq_analysis(directory):
    data = pd.DataFrame()
    file_names = os.listdir(directory)

    for file in file_names:
        ndata = pd.read_csv(os.path.join(directory, file), index_col=None)
        ndata = ndata.drop(columns = 'Year')
        firm = file.replace('_rev.csv', '').replace('edit', '')
        ndata = ndata.rename(columns={'Revenue': f'Revenue_{firm}'})
        if data.empty:
            data = ndata
        else:
            data = data.merge(ndata, on='yearno', how='outer')
    return data


## ===========================================================================
### Run analysis on data
os.chdir(DATAPATH)
mna = pd.read_csv("video_game_m&a_deals.csv", index_col = None)

filelist, privatelist = file_identification(DATAPATH)
mna = private_mna(mna, privatelist, threshold=80)
mna_revs = scrape_revenue_mnaprior(mna, filelist, 1.3757)


# %% 4. Plotting
### Create plots for company revenue and deal sizes/proportions

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='MatchedName', y='DealProp', data=mna_revs, ax=ax)
plt.title('Deal Value/Revenue for Acquired Private Firms, Deal Value > $1b')
plt.xlabel('Firm name')
plt.ylabel('Deal/Revenue')
plt.xticks(rotation=45, ha='right')
os.chdir(IMAGEPATH)
plt.savefig('financial_analysis_dealprop.png')
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.scatterplot(x='MatchedName', y='Revenue', size='Deal value (US$)',
                     sizes=(100, 1000), data=mna_revs)
ax.legend().set_visible(False)
plt.title('Revenue with Deal Value for Acquired Private Firms, Deal Value > $1b')
plt.xlabel('Firm name')
plt.ylabel('Revenue')
os.chdir(IMAGEPATH)
plt.savefig('financial_analysis_revenue_dealval.png')
plt.show()


### Create plot for movement of revenue until right before the acquisition

editpath = os.path.join(DATAPATH, 'edit')
editpath = editpath.replace('\\', '/')
joinedprivate = private_preacq_analysis(editpath)

revcolumns = [column for column in joinedprivate.columns if column != 'yearno']
for column in revcolumns:
    plt.plot(joinedprivate['yearno'], joinedprivate[column], label=column)
plt.xlabel('Year')
plt.ylabel('Revenue')
plt.title('Revenue Over Time')
os.chdir(IMAGEPATH)
plt.savefig('financial_analysis_revenueprioracq.png')
plt.legend()
plt.show()

print("complete with financials analysis for private firms")
