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
    df_raw = pd.read_csv(os.path.join(data_path, filename), encoding='iso-8859-8', index_col='שם ישוב' if not
                                                                                is_ballots else None).sort_index()
    if is_ballots:
        df_raw = df_raw[df_raw['שם ישוב'] != 'מעטפות חיצוניות']
    else:
        df_raw = df_raw[df_raw.index != 'מעטפות חיצוניות']
    if is_april:
        df = df_raw[df_raw.columns[5:]]
    else:
        df = df_raw.drop('סמל ועדה', axis=1)
        df = df[df_raw.columns[11 if is_ballots else 6:]]
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
df_ballots_a, df_ballots_raw_a = get_election_data(DATA_PATH, r'votes per ballot 2019a.csv', True, False, True)
df_ballots_raw_a.loc[df_ballots_raw_a['שם ישוב'] == 'נצרת עילית', 'שם ישוב'] = 'נוף הגליל'


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
    df = df[df.columns[11:]]
    df_par_freq = df.div(np.linalg.norm(df, 2, axis=0)).transpose()
    df_par_freq_nona = df_par_freq.fillna(0)
    pca = PCA(n_components=2)
    mapped_df = pd.DataFrame(pca.fit_transform(df_par_freq_nona), index=df.keys())
    fig, ax = plt.subplots()
    ax.scatter(mapped_df[0][list(set(df.keys()) - set(size_order))],
               mapped_df[1][list(set(df.keys()) - set(size_order))], label='Small Parties', c='r')
    ax.scatter(mapped_df[0][size_order], mapped_df[1][size_order], label='Large Parties', c='b')
    adjust_text([plt.text(mapped_df[0][i], mapped_df[1][i], party_code_dict[i][::-1], fontsize=9) for i in size_order])
    adjust_text([plt.text(mapped_df[0][i], mapped_df[1][i], i[::-1], fontsize=9)
                 for i in list(set(df.keys()) - set(size_order))])
    plt.xlabel("PCA 2 weight per ballot")
    plt.ylabel("PCA 1 weight per ballot")
    plt.title("PCA of parties")
    plt.legend()
    plt.show()


def pca_ballots_scatter(df, df_raw):
    """
    A function that plots a scatterplot of the ballots after performing PCA on the data.
    :param df: The data to perform the PCA on.
    :param df_raw: The raw data containing additional information.
    :return: None.
    """
    df_par_freq = df.transpose().div(np.linalg.norm(df.transpose(), 2, axis=0))
    df_par_freq_nona = df_par_freq.fillna(0).transpose()
    pca = PCA(n_components=2)
    mapped_df = pd.DataFrame(pca.fit_transform(df_par_freq_nona), index=df.index)
    city_1 = 'ירושלים'
    city_2 = 'בני ברק'
    city_3 = 'חיפה'
    city_4 = 'רהט'
    city_1_ballots = df_raw[df_raw['שם ישוב'] == city_1].index
    city_2_ballots = df_raw[df_raw['שם ישוב'] == city_2].index
    city_3_ballots = df_raw[df_raw['שם ישוב'] == city_3].index
    city_4_ballots = df_raw[df_raw['שם ישוב'] == city_4].index
    cols = ['r' if mapped_df.index[i] in list(mapped_df.index & city_1_ballots) else
            'g' if mapped_df.index[i] in list(mapped_df.index & city_2_ballots) else
            'b' if mapped_df.index[i] in list(mapped_df.index & city_3_ballots) else
            'k' if mapped_df.index[i] in list(mapped_df.index & city_4_ballots) else
            'y' for i in range(len(mapped_df.index))]
    all_cols = [cols, list(np.full(len(city_1_ballots), 'r')), list(np.full(len(city_2_ballots), 'g')),
                      list(np.full(len(city_3_ballots), 'b')), list(np.full(len(city_4_ballots), 'k'))]
    plots = (mapped_df, mapped_df.loc[list(mapped_df.index & city_1_ballots)],
                        mapped_df.loc[list(mapped_df.index & city_2_ballots)],
                        mapped_df.loc[list(mapped_df.index & city_3_ballots)],
                        mapped_df.loc[list(mapped_df.index & city_4_ballots)])
    labels = ['all ballots', city_1[::-1], city_2[::-1], city_3[::-1], city_4[::-1]]
    for i in range(len(plots)):
        plt.scatter(plots[i][0], plots[i][1], c=all_cols[i], label=labels[i], s=5)
    plt.xlabel("PCA 2 weight per city")
    plt.ylabel("PCA 1 weight per city")
    plt.title("PCA of ballots")
    plt.legend(markerscale=5)
    plt.show()


########## Questions 2-6 ##########


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
    return adjusted_df


def merge_ballot_dfs(df_a, df_b):
    """
    A function that merges the 2 election data dataframes, matching by city/town name and ballot id.
    :param df_a: The dataframe containing the September election results.
    :param df_b: The dataframe containing the April election results.
    :return: A dataframe containing both of the elections' data.
    """
    df_a.rename(columns={'קלפי': 'מספר קלפי'}, inplace=True)
    df_b.astype({'מספר קלפי': 'float64'})
    merged_df = pd.merge(df_a, df_b, left_on=['שם ישוב', 'מספר קלפי'], right_on=['שם ישוב', 'מספר קלפי'])
    merged_df.set_index('ברזל', inplace=True)
    return merged_df


sep_par_keys = [i + '_x' for i in size_order]
apr_par_keys = [i + '_y' for i in size_order]
merged_ballots_df = merge_ballot_dfs(df_ballots_raw, party_adjustments(df_ballots_raw_a))


def get_special_ballots(df, s_keys, a_keys, num_special=10, s_type='quadratic distances'):
    """
    A function that finds num_largest special ballots, according the s_type criterion.
    :param df: The dataframe containing both elections' data.
    :param s_keys: The keys for the parties in the September election.
    :param a_keys: The keys for the parties in the April election.
    :param num_special: The number of special ballots to find.
    :param s_type: The criterion by which to find the special ballots.
    :return: The voting patterns for each of the elections and a dictionary containing relevant info about the
    special ballots.
    """
    s_freq = df[s_keys].div(df[s_keys].sum(axis=1), axis=0)
    a_freq = df[a_keys].div(df[a_keys].sum(axis=1), axis=0)
    if s_type == 'quadratic distances':
        special = pd.DataFrame(((s_freq.values - a_freq.values) ** 2).sum(axis=1), index=s_freq.index)
    elif s_type == 'average turnout':
        special = pd.DataFrame((df['כשרים_x'].div(df['בזב_x']) + df['כשרים_y'].div(df['בזב_y']) / 2), index=df.index)
    elif s_type == 'turnout difference':
        special = pd.DataFrame(abs(df['כשרים_x'].div(df['בזב_x']) - df['כשרים_y'].div(df['בזב_y'])), index=df.index)
    elif s_type == 'turnout difference five parties':
        five_parties = ['מחל', 'טב', 'שס', 'ג', 'כף']
        s_freq = df[[i + '_x' for i in five_parties]].transpose().div(df['כשרים_x'], axis=1).transpose()
        a_freq = df[[i + '_y' for i in five_parties]].transpose().div(df['כשרים_y'], axis=1).transpose()
        special = pd.DataFrame(abs(s_freq.sum(axis=1) - a_freq.sum(axis=1)), index=df.index)
    elif s_type == 'invalid ballot difference':
        special = pd.DataFrame(abs(df['פסולים_x'].div(df['מצביעים_x']) - df['פסולים_y'].div(df['מצביעים_y'])))
    special_ballots_dict = [(i, {'city': df['שם ישוב'][i], 'ballot': df['מספר קלפי'][i], 'eligible_s': df['בזב_x'][i],
                            'voted_s': df['כשרים_x'][i], 'eligible_a': df['בזב_y'][i], 'voted_a': df['כשרים_y'][i],
                            'turnout_s': round(df['כשרים_x'][i] / df['בזב_x'][i], 2),
                            'turnout_a': round(df['כשרים_y'][i] / df['בזב_y'][i], 2),
                            '5_parties_s': round(s_freq.loc[i].sum(), 2), '5_parties_a': round(a_freq.loc[i].sum(), 2),
                            'invalid_s': round((df['פסולים_x'][i] / df['מצביעים_x'][i]) * 100, 2),
                            'invalid_a': round((df['פסולים_y'][i] / df['מצביעים_y'][i]) * 100, 2)})
                            for i in list(special.nlargest(num_special, 0).index)]
    return s_freq, a_freq, special_ballots_dict


def plot_bars(df1, df2, ballots_list, s_type='quadratic distances'):
    """
    A function that plots 10 subplots, one for each of the 10 special ballots.
    :param df1: The dataframe containing the September election results.
    :param df2: The dataframe containing the April election results.
    :param ballots_list: A list of the special ballots to plot.
    :param s_type: The criterion by which the ballots in ballots_list are special.
    :return: None.
    """
    width = 0.3
    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, sharex='col', sharey='row')
    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]

    df1_bars = []
    df2_bars = []
    for i in range(len(axs)):
        df1_bars.append(axs[i].bar(np.arange(len(df1.keys())), list(df1.loc[ballots_list[i][0]]), width, color='b'))
        df2_bars.append(axs[i].bar(np.arange(len(df2.keys())) + width, list(df2.loc[ballots_list[i][0]]), width,
                                                                                                        color='r'))

    for i in range(10):
        axs[i].set_ylim(0, 1)
        axs[i].set_xticks(np.arange(len(df1.keys())))
        axs[i].set_xticklabels([party_code_dict[name[:-2]][::-1] for name in df1.keys()], rotation='vertical',
                                                                                        fontdict={'fontsize': 8})
        if s_type == 'quadratic distances':
            axs[i].legend([], [], title_fontsize=6, title="{} {}\nElg. A.: {}\nVt. A.: {}\nElg. S.: {}\nVt. S.: {}"
                          .format(ballots_list[i][1]['ballot'], ballots_list[i][1]['city'][::-1],
                                  ballots_list[i][1]['eligible_a'], ballots_list[i][1]['voted_a'],
                                  ballots_list[i][1]['eligible_s'], ballots_list[i][1]['voted_s']))
        elif s_type == 'average turnout':
            axs[i].legend([], [], title_fontsize=6, title="{}\n{}\nTo. A.:\n{}\nTo. S.:\n{}"
                          .format(ballots_list[i][1]['city'][::-1], ballots_list[i][1]['ballot'],
                                  ballots_list[i][1]['turnout_a'], ballots_list[i][1]['turnout_s']))
        elif s_type == 'turnout difference':
            axs[i].legend([], [], title_fontsize=6, title="{} {}\nTo. A.: {}\nTo. S.: {}\nDiff.: {}"
                          .format(ballots_list[i][1]['ballot'], ballots_list[i][1]['city'][::-1],
                                  ballots_list[i][1]['turnout_a'], ballots_list[i][1]['turnout_s'],
                                  round(abs(ballots_list[i][1]['turnout_a'] - ballots_list[i][1]['turnout_s']), 2)))
        elif s_type == 'turnout difference five parties':
            axs[i].legend([], [], title_fontsize=6, title="{}\n{}\nTo. A.: {}\nTo. S.: {}\nDiff.: {}"
                          .format(ballots_list[i][1]['city'][::-1], ballots_list[i][1]['ballot'],
                                  ballots_list[i][1]['5_parties_a'], ballots_list[i][1]['5_parties_s'],
                                  round(abs(ballots_list[i][1]['5_parties_a'] - ballots_list[i][1]['5_parties_s']), 2)))
        elif s_type == 'invalid ballot difference':
            axs[i].legend([], [], title_fontsize=6, title="{}\n{}\nInv. A.:\n{}%\nInv. S.:\n{}%\nDiff.:\n{}%"
                          .format(ballots_list[i][1]['city'][::-1], ballots_list[i][1]['ballot'],
                                  ballots_list[i][1]['invalid_a'], ballots_list[i][1]['invalid_s'],
                                  round(abs(ballots_list[i][1]['invalid_a'] - ballots_list[i][1]['invalid_s']), 2)))

    fig.text(0, 0.5, "Vote share per party", va='center', rotation='vertical', fontsize=14)
    fig.text(0.5, 0.004, "Top 10 parties", ha='center', fontsize=12)
    plt.suptitle("Largest {} between elections".format(s_type), x=0.5, y=1, fontsize=14)
    fig.legend((df1_bars[0], df2_bars[0]), ('Sep.', 'Apr.'), fontsize=7, loc=3)
    plt.show()


########## To Run Code: uncomment the relevant function call and run program. ##########


### Question 1 ###

pca_parties_scatter(df_ballots_raw)

pca_ballots_scatter(df_ballots, df_ballots_raw)


### Question 2 ###

sep_freq, apr_freq, quad_distances = get_special_ballots(merged_ballots_df, sep_par_keys, apr_par_keys,
                                                         s_type='quadratic distances')
plot_bars(sep_freq, apr_freq, quad_distances, s_type='quadratic distances')


### Question 3 ###

sep_freq, apr_freq, avg_turnouts = get_special_ballots(merged_ballots_df, sep_par_keys, apr_par_keys,
                                                       s_type='average turnout')
plot_bars(sep_freq, apr_freq, avg_turnouts, s_type='average turnout')


### Question 4 ###

sep_freq, apr_freq, turnout_diff = get_special_ballots(merged_ballots_df, sep_par_keys, apr_par_keys,
                                                       s_type='turnout difference')
plot_bars(sep_freq, apr_freq, turnout_diff, s_type='turnout difference')


### Question 5 ###

sep_freq, apr_freq, turnout_diff_5_parties = get_special_ballots(merged_ballots_df, sep_par_keys, apr_par_keys,
                                                                 s_type='turnout difference five parties')
plot_bars(sep_freq, apr_freq, turnout_diff_5_parties, s_type='turnout difference five parties')


### Question 6 ###

sep_freq, apr_freq, turnout_diff_5_parties = get_special_ballots(merged_ballots_df, sep_par_keys, apr_par_keys,
                                                                 s_type='invalid ballot difference')
plot_bars(sep_freq, apr_freq, turnout_diff_5_parties, s_type='invalid ballot difference')
