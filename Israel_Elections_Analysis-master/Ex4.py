import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as spy


DATA_PATH = 'C:\\Users\\Meir\\PycharmProjects\\Stat_Lab'


df_cities_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per city 2019b.csv'), encoding='iso-8859-8',
                         index_col='שם ישוב').sort_index()
df_cities_raw = df_cities_raw[df_cities_raw.index != 'מעטפות חיצוניות']
df_cities = df_cities_raw.drop('סמל ועדה', axis=1)
df_cities = df_cities[df_cities.columns[4:]]
size_order = list(df_cities[[df_cities.keys()[i] for i in range(1, len(df_cities.keys()))]].sum().sort_values(
    ascending=False).keys())[:10]
df_cities = df_cities[size_order]


df_socio_raw = pd.read_csv(os.path.join(DATA_PATH, r'HevratiCalcaliYeshuvim.csv'), encoding='iso-8859-8',
                         index_col='רשות מקומית').sort_index()


party_code_dict = {'פה': 'כחול לבן', 'מחל': 'הליכוד', 'ודעם': 'הרשימה המשותפת', 'ג': 'יהדות התורה', 'שס': 'ש"ס',
                   'ל': 'ישראל ביתנו', 'אמת': 'העבודה גשר', 'טב': 'ימינה', 'מרצ': 'המחנה הדמוקרטי', 'כף': 'עוצמה יהודית',
                   'ז': 'עוצמה כלכלית', 'זכ': 'מפלגת הדמוקראטורה', 'זן': 'יחד', 'זץ': 'צומת', 'י': 'מנהיגות חברתית',
                   'יז': 'אדום לבן', 'ינ': 'התנועה הנוצרית הליבראלית', 'יף': 'כבוד האדם', 'יק': 'מפלגת הגוש התנכ"י',
                   'כ': 'נעם', 'נ': 'מתקדמת', 'נך': 'כבוד ושוויון', 'נץ': 'כל ישראל אחים', 'כי': 'האחדות העממית',
                   'ףז': 'הפיראטים', 'צ': 'צדק', 'צן': 'צפון', 'ץ': 'דעם', 'ק': 'זכויותינו בקולנו', 'קך': 'סדר חדש',
                   'קץ': 'קמ"ה', 'רק': 'הימין החילוני'}


########## Question 1 ##########


def intersection_comparison(df, df_raw, df_socio, with_print=True):
    """
    A function that returns the actual election result frequencies and the frequencies of the 10 largest parties and
    cities in common with the socio-economic data file.
    :param df: The dataframe containing the election data.
    :param df_raw: The raw dataframe containing the election data.
    :param df_socio: The dataframe containing the socio-economic data.
    :param with_print: A boolean whether to print the cities in common or not.
    :return: The frequencies for the whole country and the frequencies in the cities in common.
    """
    intersect = df_socio.index & df.index
    if with_print:
        print("The number of cities/towns appearing identically in both files is: {}".format(len(intersect)))
        print(list(intersect))
    df_filtered_cities = df.loc[intersect]
    df_filtered_valid = df_filtered_cities.sum(axis=1)
    filtered_par = df_filtered_cities.sum().div(df_filtered_valid.sum())[size_order]

    # Calculates the percentage of valid votes received by each party
    par = df.sum().div(df_raw['כשרים'].sum()).sort_values(ascending=False)

    return par, filtered_par


def plot_bars(df1, df2):
    """
    A function that plots the countrywide election results vs. the election results in cities in common with the
    socio-economic data file.
    :param df1: The dataframe containing the countrywide election results.
    :param df2: The dataframe containing the election data for the common cities only.
    :return: None.
    """
    width = 0.3
    fig, ax = plt.subplots()
    df1_bar = ax.bar(np.arange(len(df1.keys())), list(df1), width, color='b')
    df2_bar = ax.bar(np.arange(len(df2.keys())) + width, list(df2), width, color='r')

    for i in range(len(df1_bar)):
        ax.text(i - width - 0.03, df1_bar[i]._height + 0.001, round(df1_bar[i]._height, 3), fontsize=6)
        ax.text(i + width / 1.5, df2_bar[i]._height + 0.001, round(df2_bar[i]._height, 3), fontsize=6)

    ax.set_xticks(np.arange(len(df1.keys())))
    ax.set_xticklabels([party_code_dict[name][::-1] for name in df2.keys()], rotation='vertical')
    ax.legend((df1_bar[0], df2_bar[0]), ('real', 'intersection'))
    ax.set_xlabel("Top 10 parties")
    ax.set_ylabel("Vote share per party")
    ax.set_title("Voting pattern comparison all vs. filtered cities", fontsize=16)
    plt.show()


########## Question 2 ##########


def plot_bars2(df1, df2_lst):
    """
    A function that plots 10 subplots, one for each of the 10 socio-economic ratings, by the performance of the 10
    largest parties.
    :param df1: The dataframe containing the countrywide election results.
    :param df2_lst: A list of dataframes to plot.
    :return: None.
    """
    width = 0.3
    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, sharex='col', sharey='row')
    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]

    df1_bars = []
    df2_bars = []
    for i in range(len(axs)):
        df1_bars.append(axs[i].bar(np.arange(len(df1.keys())), list(df1), width, color='b'))
        df2_bars.append(axs[i].bar(np.arange(len(df2_lst[i].keys())) + width, list(df2_lst[i]), width, color='r'))

    for i in range(10):
        axs[i].set_xticks(np.arange(len(df1.keys())))
        axs[i].set_xticklabels([party_code_dict[name][::-1] for name in df1.keys()], rotation='vertical',
                               fontdict={'fontsize': 8})
        axs[i].legend((df1_bars[i], df2_bars[i]), ('real', 'rating {}'.format(i + 1)), fontsize=6)

    fig.text(0, 0.5, "Vote share per party", va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.004, "Top 10 parties", ha='center', fontsize=12)
    plt.suptitle("Voting pattern comparison country vs. socio-economic rating", x=0.5, y=1, fontsize=14)
    plt.show()


def socio_economic_comparison(df, df_raw, df_socio, with_plot=True):
    """
    A function that calculates the performance of the 10 largest parties in cities by socio-economic rank.
    :param df: The dataframe containing the election data.
    :param df_raw: The raw dataframe containing the election data.
    :param df_socio: The dataframe containing the socio-economic data.
    :param with_plot: Whether to plot the results or not.
    :return: The calculated countrywide and 10 largest parties dataframes if with_plot is False, None otherwise.
    """
    dfs_by_socio_rank = [df_socio[df_socio['מדד חברתי-'] == str(i)] for i in range(1, 11)]
    filtered_par_lst = []
    for i in range(10):
        par, filtered_par = intersection_comparison(df, df_raw, dfs_by_socio_rank[i], with_print=False)
        filtered_par_lst.append(filtered_par)
    if with_plot:
       plot_bars2(par, filtered_par_lst)
    else:
        return par, filtered_par_lst


########## Question 3 ##########


def party_share_by_socio_rank(df, df_raw, df_socio):
    """
    A function that essentially reformats the data already calculated to divide by party instead of by socio-economic
    rank.
    :param df: The dataframe with the election data.
    :param df_raw: The raw dataframe with the election data.
    :param df_socio: The dataframe containing the socio-economic rank info.
    :return: A list of dataframes per party.
    """
    par, filtered_par = socio_economic_comparison(df, df_raw, df_socio, with_plot=False)
    filtered_par_by_rank = [pd.DataFrame([j[i] for j in filtered_par], columns=[i]) for i in par.keys()]
    filtered_par_by_rank = [df.set_index(pd.Index([i for i in range(1, 11)])) for df in filtered_par_by_rank]
    return filtered_par_by_rank


def plot_bars_by_party(df_lst):
    """
    A function that plots 10 subplots, one for each of the 10 largest parties, by their performance in cities by the 10
    socio-economic ratings.
    :param df_lst: A list of dataframes to plot.
    :return: None.
    """
    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, sharex='col', sharey='row')
    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]

    df_bars = []
    for i in range(len(axs)):
        df_bars.append(axs[i].bar(np.arange(1, len(df_lst[i].index) + 1), list(df_lst[i].values.flatten()), color='b'))

    for i in range(10):
        axs[i].set_xticks(np.arange(1, len(df_lst[i].index) + 1))
        axs[i].set_xticklabels(np.arange(1, len(df_lst[i].index) + 1), fontdict={'fontsize': 7})
        axs[i].legend([party_code_dict[df_lst[i].keys()[0]][::-1]], fontsize=5)

    fig.text(0, 0.5, "Vote share of party", va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.004, "Socio-economic rank", ha='center', fontsize=12)
    plt.suptitle("Voting pattern parties by socio-economic rating", x=0.5, y=1, fontsize=14)
    plt.show()


########## Question 4 ##########


df_ballots_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per ballot 2019b.csv'), encoding='iso-8859-8',
                         index_col='ברזל').sort_index()
df_ballots_raw = df_ballots_raw[df_ballots_raw['שם ישוב'] != 'מעטפות חיצוניות']
df_ballots = df_ballots_raw.drop('סמל ועדה', axis=1)


def calculate_ballot_distance(city, ballot, city_df, ballots_df):
    """
    A function that calculates the distance between a ballot and its city.
    :param city: The city.
    :param ballot: The ballot.
    :param city_df: The dataframe containing the city data.
    :param ballots_df: The dataframe containing the ballot data.
    :return: The distance.
    """
    tmp = sum((city_df.loc[city] - ballots_df.loc[ballot]) ** 2)
    return tmp if tmp >= 0 else 0


def calculate_heterogeneity(df, df_socio):
    """
    A function that calculates the heterogeneity coefficients and retieves the gini index coefficients.
    :param df: The dataframe containing the ballots.
    :param df_socio: The dataframe containing the gini index values.
    :return: The heterogeneity and gini index coefficients and the list of common cities.
    """
    intersect = df_socio.index & df['שם ישוב']
    print("The number of ballots whose cities appear in both files is: {}".format(len(intersect)))
    intersect = list(intersect.unique())
    df_filtered_ballots_raw = df.loc[df['שם ישוב'].isin(intersect)]
    df_filtered_ballots = df_filtered_ballots_raw[size_order]

    # A dict containing a map of cities to ballot numbers.
    city_ballot_dict = {i: list(df_filtered_ballots_raw[df_filtered_ballots_raw['שם ישוב'] == i].index) for i in
                        intersect}

    # A dataframe containing the p_i values.
    df_pi = pd.DataFrame([df_filtered_ballots.loc[city_ballot_dict[i]].sum().div(df_filtered_ballots.loc[
                                city_ballot_dict[i]].sum().sum()) for i in intersect], columns=size_order)
    df_pi.rename(index={i: intersect[i] for i in range(len(intersect))}, inplace=True)

    # A dataframe containing the p_i_b values.
    df_pi_b = pd.DataFrame([df_filtered_ballots.loc[i].div(df_filtered_ballots.loc[i].sum()) for i in
                            df_filtered_ballots.index], columns=size_order)

    # A dict containing the distances of each ballot from its city, key is city, val is list of distances.
    distance_dict = {i: [calculate_ballot_distance(i, j, df_pi, df_pi_b) for j in city_ballot_dict[i]] for i in
                     intersect}

    # Removes cities with only 1 ballot (thus automatically have a distance of 0).
    only_once = [key for key, val in distance_dict.items() if len(val) == 1]
    for i in only_once:
        del distance_dict[i]
        intersect.remove(i)

    # A dataframe containing the average distances for each city from its ballots.
    hetero_dist_df = pd.DataFrame(sum(distance_dict[i]) / len(distance_dict[i]) for i in intersect)
    hetero_dist_df.rename(index={i: intersect[i] for i in range(len(intersect))}, inplace=True)

    # A dataframe containing the gini index for each city (in common).
    df_socio_gini = df_socio.loc[df_socio.index.isin(hetero_dist_df.index)]["מדד ג'יני[2]"]

    return hetero_dist_df, df_socio_gini, intersect


def plot_heterogeneity_comparison(dist_df, gini_df, df_raw):
    """
    A function that plots a scatterplot of the heterogeneity coefficients vs. the gini index coefficients.
    :param dist_df: The dataframe containing the distances.
    :param gini_df: The dataframe containing the gini index coefficients.
    :param df_raw: The dataframe containing the raw election data.
    :return: None.
    """
    plt.scatter(dist_df, gini_df, s=df_raw['כשרים'] / 1300, c='b')
    plt.xlabel('Heterogeneity Coefficient')
    plt.ylabel('Gini Coefficient')
    plt.title('Heterogeneity distance vs. Gini index', fontsize=18)
    plt.show()


########## To Run Code: uncomment the relevant function call and run program. ##########


### Question 1 ###

par, filtered_par = intersection_comparison(df_cities, df_cities_raw, df_socio_raw)
plot_bars(par, filtered_par)


### Question 2 ###

socio_economic_comparison(df_cities, df_cities_raw, df_socio_raw)


### Question 3 ###

filtered_par_by_rank = party_share_by_socio_rank(df_cities, df_cities_raw, df_socio_raw)
plot_bars_by_party(filtered_par_by_rank)


### Question 4 ###

hetero_df, socio_gini_df, common_cities = calculate_heterogeneity(df_ballots, df_socio_raw)
plot_heterogeneity_comparison(hetero_df, socio_gini_df, df_cities_raw.loc[common_cities])
print('The spearman coefficient of heterogeneity and gini across the different cities is {}'.format(spy.spearmanr(
    hetero_df, socio_gini_df)[0]))
