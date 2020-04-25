import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as spy
from itertools import combinations_with_replacement
import seaborn as sns
sns.set()


DATA_PATH = 'C:\\Users\\Meir\\PycharmProjects\\Stat_Lab'

df_cities_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per city 2019b.csv'), encoding='iso-8859-8',
                         index_col='שם ישוב').sort_index()
df_cities = df_cities_raw.drop('סמל ועדה', axis=1)
df_cities = df_cities[df_cities.columns[4:]]

# a dictionary mapping each party's letters to its name, for each of the 10 largest parties (by vote count)
party_code_dict = {'פה': 'כחול לבן', 'מחל': 'הליכוד', 'ודעם': 'הרשימה המשותפת', 'ג': 'יהדות התורה', 'שס': 'ש"ס',
                   'ל': 'ישראל ביתנו', 'אמת': 'העבודה גשר', 'טב': 'ימינה', 'מרצ': 'המחנה הדמוקרטי', 'כף': 'עוצמה יהודית',
                   'ז': 'עוצמה כלכלית', 'זכ': 'מפלגת הדמוקראטורה', 'זן': 'יחד', 'זץ': 'צומת', 'י': 'מנהיגות חברתית',
                   'יז': 'אדום לבן', 'ינ': 'התנועה הנוצרית הליבראלית', 'יף': 'כבוד האדם', 'יק': 'מפלגת הגוש התנכ"י',
                   'כ': 'נעם', 'נ': 'מתקדמת', 'נך': 'כבוד ושוויון', 'נץ': 'כל ישראל אחים', 'כי': 'האחדות העממית',
                   'ףז': 'הפיראטים', 'צ': 'צדק', 'צן': 'צפון', 'ץ': 'דעם', 'ק': 'זכויותינו בקולנו', 'קך': 'סדר חדש',
                   'קץ': 'קמ"ה', 'רק': 'הימין החילוני'}


########## Question 1 ##########


def party_party_scatter(df, party1, party2):
    """
    A function that plots a scatter-plot comparing 2 parties' performances across different towns and cities,
    sized proportionally to each city/town's population.
    :param df: The dataframe.
    :param party1: The first party.
    :param party2: The second party.
    :return: None.
    """
    df = df.drop('מעטפות חיצוניות')
    votes_per_city = df.drop('כשרים', axis=1).sum(axis=1)
    party_share1 = df[party1] / votes_per_city
    party_share2 = df[party2] / votes_per_city
    # we found that 1300 is the scalar that 'resizes' the largest city to about 200
    plt.scatter(party_share1, party_share2, s=df['כשרים'] / 1300, c="blue")
    plt.xlabel('Share of voters who voted for {}'.format(party_code_dict[party1][::-1]))
    plt.ylabel('Share of voters who voted for {}'.format(party_code_dict[party2][::-1]))
    plt.title('Scatter plot for {} vs. {}'.format(party_code_dict[party1][::-1], party_code_dict[party2][::-1]),
              fontsize=18)
    plt.show()


########## Question 2 ##########


def heatmapper(df, x_label, y_label, title):
    """
    A function that receives a dataframe and plots a heatmap of it.
    :param df: The dataframe.
    :param x_label: The label for the X axis.
    :param y_label: The label for the Y axis.
    :param title: The title for the heatmap.
    :return: None.
    """
    # fixes the labels so they print correctly in Hebrew
    labels = [party_code_dict[i][::-1] for i in list(df.keys())]
    sns.heatmap(df, square=True, annot=False, cbar=True, xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=13)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


# removes 'מעטפות חיצוניות' from the dataframe
df_cities_parties = df_cities[df_cities.columns[1:]]

# creates a list of the parties ordered by the number of votes received
size_order = list(df_cities[[df_cities.keys()[i] for i in range(1, len(df_cities.keys()))]].sum().sort_values(
    ascending=False).keys())
political_order = ['ודעם', 'מרצ', 'אמת', 'פה', 'ל', 'מחל', 'ג', 'שס', 'טב', 'כף']


def correlation_heatmap(df, corr_method=spy.pearsonr, order=size_order[:10]):
    """
    A function that plots a heatmap of the correlations between the 10 largest parties (by vote count).
    :param df: The dataframe.
    :param corr_method: The correlation method to use.
    :param order: The ordering method to use.
    :return: None.
    """
    df = df.drop('מעטפות חיצוניות')
    # creates an empty 10x10 dataframe
    corr_df = pd.DataFrame(np.zeros((10, 10)), columns=order)
    # this changes the row index values to be identical to the columns
    corr_df.rename(index={i: order[i] for i in range(10)}, inplace=True)
    for (p1, p2) in [i for i in combinations_with_replacement(order, 2)]:
        if p1 == p2:
            corr_df[p1][p2] = 1
        else:
            corr = corr_method(df.loc[:, p1], df.loc[:, p2])[0]
            corr_df[p1][p2], corr_df[p2][p1] = corr, corr
    title = "{} correlation heatmap ordered by {}".format(corr_method.__name__[:-1].capitalize(),
                                                "political leanings" if order == political_order else "size")
    heatmapper(corr_df.reindex(order).reindex(order, axis=1), "Parties", "Parties", title=title)


########## Question 3 ##########


df_ballots_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per ballot 2019b.csv'), encoding='iso-8859-8',
                         index_col=['קלפי', 'שם ישוב']).sort_index()
df_ballots = df_ballots_raw.drop('סמל ועדה', axis=1)
df_ballots = df_ballots[df_ballots.columns[8:]]


def turnout_comparison(df, df_raw, by_ballot=False, top_10=False):
    """
    A function that 'fixes' the voter turnout rate to be 100% and compares the resulting voting patterns with the
    actual voting patterns. We do this under the assumption that the people who did not vote will vote similarly to
    those in their city/precinct.
    :param df: The 'main' dataframe, including only the parties.
    :param df_raw: A raw dataframe containing additional columns.
    :param by_ballot: A boolean parameter whether the scaling is performed at the city or ballot level.
    :param top_10: A boolean parameter whether to show all parties or only the 10 largest by votes received.
    :return: None.
    """
    df_turnout = df_raw['כשרים'] / df_raw['בזב']
    df_scaled = (df.transpose() / df_turnout).transpose()

    # Filters the dataframe to only include the columns representing parties
    parties_only = df[[df.keys()[i] for i in range(len(df.keys()))]]
    # Calculates the percentage of valid votes received by each party
    par = parties_only.sum().div(df_raw['כשרים'].sum()).sort_values(ascending=False)

    # Filters the dataframe to only include the columns representing parties, with scaled values
    scaled_parties_only = df_scaled[[df_scaled.keys()[i] for i in range(len(df_scaled.keys()))]]
    # Calculates the percentage of valid votes received by each party, after scaling
    scaled_par = scaled_parties_only.sum().div(df_raw['בזב'].sum()).sort_values(ascending=False)

    (par, scaled_par) = (par[:10], scaled_par[:10]) if top_10 else (par, scaled_par)

    width = 0.3
    fig, ax = plt.subplots()
    scaled_bar = ax.bar(np.arange(len(scaled_par.keys())), list(scaled_par), width, color='b')
    original_bar = ax.bar(np.arange(len(par.keys())) + width, list(par), width, color='r')

    if top_10:
        for i in range(len(scaled_bar)):
            ax.text(i - width, scaled_bar[i]._height, round(scaled_bar[i]._height, 3), fontsize=6)
            ax.text(i + width / 1.5, original_bar[i]._height, round(original_bar[i]._height, 3), fontsize=6)

    ax.set_xticks(np.arange(len(par.keys())))
    ax.set_xticklabels([party_code_dict[name][::-1] for name in par.keys()], rotation='vertical')
    ax.legend((scaled_bar[0], original_bar[0]), ('scaled', 'original'))
    ax.set_xlabel("Top 10 parties" if top_10 else "Full party list")
    ax.set_ylabel("Vote share per party")
    ax.set_title("Voting pattern comparison scaled by {}".format('ballots' if by_ballot else 'cities'), fontsize=16)
    plt.show()


########## To Run Code: uncomment the relevant function call and run program. ##########


### Question 1 ###

parties_pair_1 = ('פה', 'מחל')
party_party_scatter(df_cities, parties_pair_1[0], parties_pair_1[1])

parties_pair_2 = ('אמת', 'מרצ')
party_party_scatter(df_cities, parties_pair_2[0], parties_pair_2[1])

parties_pair_3 = ('ודעם', 'כף')
party_party_scatter(df_cities, parties_pair_3[0], parties_pair_3[1])


### Question 2 ###

# using the Pearson correlation method and ordered by party size
correlation_heatmap(df_cities_parties)

# using the Pearson correlation method and ordered by party political orientation
correlation_heatmap(df_cities_parties, order=political_order)

# using the Spearman correlation method and ordered by party size
correlation_heatmap(df_cities_parties, spy.spearmanr)

# using the Spearman correlation method and ordered by party political orientation
correlation_heatmap(df_cities_parties, spy.spearmanr, order=political_order)


### Question 3 ###

# performing the comparison by 'fixing' the turnout at the city/town level
turnout_comparison(df_cities.drop('כשרים', axis=1), df_cities_raw)

# performing the comparison by 'fixing' the turnout at the precinct level
turnout_comparison(df_ballots, df_ballots_raw, by_ballot=True)

# performing the comparison by 'fixing' the turnout at the city/town level for the top 10 parties only
turnout_comparison(df_cities.drop('כשרים', axis=1), df_cities_raw, top_10=True)

# performing the comparison by 'fixing' the turnout at the precinct level for the top 10 parties only
turnout_comparison(df_ballots, df_ballots_raw, by_ballot=True, top_10=True)
