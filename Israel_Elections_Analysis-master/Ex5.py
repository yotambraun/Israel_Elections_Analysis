import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from adjustText import adjust_text


DATA_PATH = 'C:\\Users\\Meir\\PycharmProjects\\Stat_Lab'


def get_election_data(data_path, filename, is_ballots=True, top_10=True, is_april=False):
    """
    A function that receives a filepath and returns the raw and cleaned data, as well as a list of the largest parties.
    :param data_path: The directory path.
    :param filename: The file name.
    :param is_ballots: A boolean whether this is a ballots .csv or a cities .csv file.
    :param top_10: A boolean whether to filter out all but the top parties.
    :param is_april: A boolean whether the data is from the September election or the April one.
    :return: A cleaned dataframe, a raw dataframe and a list of the largest parties by total votes.
    """
    df_raw = pd.read_csv(os.path.join(data_path, filename), encoding='iso-8859-8',
                         index_col='ברזל' if is_ballots else 'שם ישוב').sort_index()
    if is_ballots:
        df_raw = df_raw[df_raw['שם ישוב'] != 'מעטפות חיצוניות']
    else:
        df_raw = df_raw[df_raw.index != 'מעטפות חיצוניות']
    if is_april:
        df = df_raw[df_raw.columns[5:]]
    else:
        df = df_raw.drop('סמל ועדה', axis=1)
        df = df[df_raw.columns[10 if is_ballots else 6:]]
    if top_10:
        if is_april:
            size_order = list(df[[df.keys()[i] for i in range(len(df.keys()))]].sum().
                              sort_values(ascending=False).keys())[:14]
        else:
            size_order = list(df[[df.keys()[i] for i in range(len(df.keys()))]].sum().
                              sort_values(ascending=False).keys())[:10]
        df = df[size_order]
        return df, df_raw, size_order
    return df, df_raw


df_ballots, df_ballots_raw, size_order = get_election_data(DATA_PATH, r'votes per ballot 2019b.csv')
df_cities, df_cities_raw = get_election_data(DATA_PATH, r'votes per city 2019b.csv', False, False)
df_cities_a, df_cities_raw_a, size_order_a = get_election_data(DATA_PATH, r'votes per city 2019a.csv', False, True,
                                                               True)
df_cities_a.rename(index={'נצרת עילית': 'נוף הגליל'}, inplace=True)
df_cities_raw_a.rename(index={'נצרת עילית': 'נוף הגליל'}, inplace=True)
largest_cities = df_cities_raw[df_cities_raw['בזב'] >= 10000].index & df_cities_raw_a[df_cities_raw_a['בזב'] >=
                                                                                      10000].index


party_code_dict = {'פה': 'כחול לבן', 'מחל': 'הליכוד', 'ודעם': 'הרשימה המשותפת', 'ג': 'יהדות התורה', 'שס': 'ש"ס',
                   'ל': 'ישראל ביתנו', 'אמת': 'העבודה גשר', 'טב': 'ימינה', 'מרצ': 'המחנה הדמוקרטי', 'כף': 'עוצמה יהודית',
                   'ז': 'עוצמה כלכלית', 'זכ': 'מפלגת הדמוקראטורה', 'זן': 'יחד', 'זץ': 'צומת', 'י': 'מנהיגות חברתית',
                   'יז': 'אדום לבן', 'ינ': 'התנועה הנוצרית הליבראלית', 'יף': 'כבוד האדם', 'יק': 'מפלגת הגוש התנכ"י',
                   'כ': 'נעם', 'נ': 'מתקדמת', 'נך': 'כבוד ושוויון', 'נץ': 'כל ישראל אחים', 'כי': 'האחדות העממית',
                   'ףז': 'הפיראטים', 'צ': 'צדק', 'צן': 'צפון', 'ץ': 'דעם', 'ק': 'זכויותינו בקולנו', 'קך': 'סדר חדש',
                   'קץ': 'קמ"ה', 'רק': 'הימין החילוני'}


########## Question 1 ##########


def pca_parties_scatter(df):
    """
    A function that performs PCA on the 10 largest parties in the election.
    :param df: The election data.
    :return: None.
    """
    df = df[df.columns[10:]]
    df_par_freq = df.div(df.sum(axis=1), axis=0)
    df_par_freq_nona = df_par_freq.fillna(0).transpose()
    pca = PCA(n_components=2)
    mapped_df = pd.DataFrame(pca.fit_transform(df_par_freq_nona), index=df.keys())
    fig, ax = plt.subplots()
    ax.scatter(mapped_df[0][size_order], mapped_df[1][size_order], label='Large Parties', c='b')
    ax.scatter(mapped_df[0][list(set(df.keys()) - set(size_order))], mapped_df[1][list(set(df.keys()) - set(
        size_order))], label='Small Parties', c='r')
    adjust_text([plt.text(mapped_df[0][i], mapped_df[1][i], party_code_dict[i][::-1], fontsize=9) for i in size_order])
    plt.xlabel("PCA 2 weight per ballot")
    plt.ylabel("PCA 1 weight per ballot")
    plt.title("PCA of parties")
    plt.legend()
    plt.show()


########## Question 2 ##########


def pca_ballots_scatter(df, df_raw):
    """
    A function that performs PCA on the ballots in the election.
    :param df: The election data.
    :param df_raw: The raw election data.
    :return: None.
    """
    df_par_freq = df.div(df.sum(axis=1), axis=0)
    df_par_freq_nona = df_par_freq.fillna(0)
    pca = PCA(n_components=2)
    mapped_df = pd.DataFrame(pca.fit_transform(df_par_freq_nona), index=df.index)
    city_1 = 'חיפה'
    city_2 = 'רהט'
    city_1_ballots = df_raw[df_raw['שם ישוב'] == city_1].index
    city_2_ballots = df_raw[df_raw['שם ישוב'] == city_2].index
    plt.figure(figsize=(5, 5))
    cols = ['r' if mapped_df.index[i] in list(mapped_df.index & city_1_ballots) else ('k' if mapped_df.index[i] in list(
        mapped_df.index & city_2_ballots) else 'y') for i in range(len(mapped_df.index))]
    plt.xlabel("PCA 2 weight per ballot")
    plt.ylabel("PCA 1 weight per ballot")
    plt.title("PCA of ballots")
    all_cols = [cols, list(np.full(len(city_1_ballots), 'r')), list(np.full(len(city_2_ballots), 'k'))]
    plots = (mapped_df, mapped_df.loc[list(mapped_df.index & city_1_ballots)], mapped_df.loc[list(mapped_df.index &
                                                                                                 city_2_ballots)])
    labels = ['all ballots', city_1[::-1], city_2[::-1]]
    for i in range(len(plots)):
        plt.scatter(plots[i][0], plots[i][1], c=all_cols[i], label=labels[i])
    plt.legend()
    plt.show()


########## Question 3 ##########


def pca_cities_scatter(df, df_raw):
    """
    A function that plots a scatterplot of the cities after performing PCA on the data.
    :param df: The data to perform the PCA on.
    :param df_raw: The raw data containing additional information.
    :return: None.
    """
    df_par_freq = df.div(df.sum(axis=1), axis=0)
    df_par_freq_nona = df_par_freq.fillna(0)
    pca = PCA(n_components=2)
    mapped_df = pd.DataFrame(pca.fit_transform(df_par_freq_nona), index=df.index)
    plt.figure(figsize=(5, 5))

    plt.xlabel("PCA 2 weight per city")
    plt.ylabel("PCA 1 weight per city")
    plt.title("PCA of cities")
    plt.scatter(mapped_df[0], mapped_df[1], s=df_raw['כשרים'] / 1300)
    plt.show()


def party_adjustments(df):
    """
    A function that adjusts the parties to enable comparison of the September and April elections.
    :param df: The election data.
    :return: The adjusted data.
    """
    adjusted_df = df
    adjusted_df['מחל'] = df['מחל'] + df['כ'] + 0.5 * df['ז']
    adjusted_df['פה'] = df['פה']
    adjusted_df['ודעם'] = df['ום'] + df['דעם']
    adjusted_df['שס'] = df['שס']
    adjusted_df['ל'] = df['ל']
    adjusted_df['ג'] = df['ג']
    adjusted_df['טב'] = df['נ'] + 0.5 * (df['ז'] + df['טב'])
    adjusted_df['אמת'] = df['אמת'] + df['נר']
    adjusted_df['מרצ'] = df['מרצ']
    adjusted_df['כף'] = 0.5 * df['טב']
    adjusted_df = adjusted_df[size_order]
    return adjusted_df


def election_comparison(df_a, df_b):
    """
    A function that compares the 2 elections and returns the result of applying PCA to each.
    :param df_a: The first election data to compare.
    :param df_b: The first election data to compare.
    :return: The first and second election data after adjusting and applying PCA to them.
    """
    # Filters the dataframe to only include the columns representing parties
    parties_only = df_b[[df_b.keys()[i] for i in range(len(df_b.keys()))]]

    adjusted_par = party_adjustments(parties_only)

    df_par_freq = df_a.div(df_a.sum(axis=1), axis=0)
    df_par_freq_nona = df_par_freq.fillna(0)

    pca = PCA(n_components=2)
    mapped_df = pd.DataFrame(pca.fit_transform(df_par_freq_nona[size_order]), index=df_a.index)

    df_adj_par_freq = adjusted_par.div(adjusted_par.sum(axis=1), axis=0)
    df_adj_par_freq_nona = df_adj_par_freq.fillna(0)
    mapped_adjusted_par = pd.DataFrame(pca.transform(df_adj_par_freq_nona), index=df_b.index)
    return mapped_df, mapped_adjusted_par


def plot_election_comparisons(mapped_df_a, mapped_df_b, df_raw_a, df_raw_b, arrow_points=(), top_3=()):
    """
    A function that plots a scatterplot comparison of the 2 elections.
    :param mapped_df_a: The first election data after applying PCA.
    :param mapped_df_b: The second election data after applying PCA.
    :param df_raw_a: The raw data for the first election.
    :param df_raw_b: The raw data for the second election.
    :param arrow_points: A list/tuple of points to connect by arrow.
    :param top_3: An additional list of points to connect by (a different color) arrow.
    :return: None.
    """
    plt.figure(figsize=(5, 5))
    plt.xlabel("PCA 2 weight per city")
    plt.ylabel("PCA 1 weight per city")
    plt.title("PCA comparison of elections")
    plt.scatter(mapped_df_a[0], mapped_df_a[1], s=df_raw_a['כשרים'] / 1300, c='b', label='September')
    plt.scatter(mapped_df_b[0], mapped_df_b[1], s=df_raw_b['כשרים'] / 1300, c='r', label='April')
    plt.legend()
    for i in arrow_points:
        plt.arrow(mapped_df_b.loc[i][0], mapped_df_b.loc[i][1], mapped_df_a.loc[i][0] - mapped_df_b.loc[i][0],
                  mapped_df_a.loc[i][1] - mapped_df_b.loc[i][1], shape='full', color='k', length_includes_head=True,
                  head_length=0.01, head_width=0.02)
    for i in top_3:
        plt.arrow(mapped_df_b.loc[i][0], mapped_df_b.loc[i][1], mapped_df_a.loc[i][0] - mapped_df_b.loc[i][0],
                  mapped_df_a.loc[i][1] - mapped_df_b.loc[i][1], shape='full', color='g',
                  length_includes_head=True,
                  head_length=0.01, head_width=0.02)

    plt.show()


def city_election_comparison_plot(df, df_a, df_raw, df_raw_a, city):
    """
    A function that plots a bar plot of a given city comparing the voting pattern in the two elections.
    :param df: The first election data.
    :param df_a: The second election data.
    :param df_raw: The first election raw data.
    :param df_raw_a: The second election raw data.
    :param city: The city.
    :return: None.
    """
    sep_res = df.loc[city] / df_raw['כשרים'][city]
    apr_res = df_a.loc[city] / df_raw_a['כשרים'][city]

    width = 0.3
    fig, ax = plt.subplots()
    sep_bar = ax.bar(np.arange(len(sep_res.keys())), list(sep_res), width, color='b')
    apr_bar = ax.bar(np.arange(len(apr_res.keys())) + width, list(apr_res), width, color='r')

    for i in range(len(sep_bar)):
        ax.text(i - width, sep_bar[i]._height, round(sep_bar[i]._height, 3), fontsize=6)
        ax.text(i + width / 1.5, apr_bar[i]._height, round(apr_bar[i]._height, 3), fontsize=6)

    ax.set_xticks(np.arange(len(apr_res.keys())))
    ax.set_xticklabels([party_code_dict[name][::-1] for name in apr_res.keys()], rotation='vertical')
    ax.legend((sep_bar[0], apr_bar[0]), ('September', 'April'))
    ax.set_ylabel("Vote share per party")
    ax.set_title("Voting pattern comparison in {}".format(city[::-1]), fontsize=16)
    plt.show()


########## To Run Code: uncomment the relevant function call and run program. ##########


### Question 1 ###

pca_parties_scatter(df_ballots_raw)


### Question 2 ###

pca_ballots_scatter(df_ballots, df_ballots_raw)


### Question 3 ###

### part a
pca_cities_scatter(df_cities, df_cities_raw)

### part d
actual_election, adjusted_election = election_comparison(df_cities, df_cities_a)
plot_election_comparisons(actual_election, adjusted_election, df_cities_raw, df_cities_raw_a)

### part e
actual_election, adjusted_election = election_comparison(df_cities, df_cities_a)
plot_election_comparisons(actual_election, adjusted_election, df_cities_raw, df_cities_raw_a, largest_cities)

### part f
actual_election, adjusted_election = election_comparison(df_cities, df_cities_a)
differ_election = pd.DataFrame((actual_election[0] - adjusted_election[0]) ** 2 + (actual_election[1] -
                                                                                  adjusted_election[1]) ** 2)
top_3_differ = differ_election.nlargest(n=3, columns=list(differ_election)[0])
plot_election_comparisons(actual_election, adjusted_election, df_cities_raw, df_cities_raw_a, largest_cities,
                           top_3_differ.index)
for city in top_3_differ.index:
    city_election_comparison_plot(df_cities[size_order], party_adjustments(df_cities_a), df_cities_raw,
                                  df_cities_raw_a, city)
