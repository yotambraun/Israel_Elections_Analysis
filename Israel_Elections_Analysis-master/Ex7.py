import os
import pandas as pd
import numpy as np
import copy
from matplotlib import pyplot as plt
from scipy import optimize as s_opt
import plotly.graph_objects as go
import cvxpy as cvx
import seaborn as sns
sns.set()


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
        df = df_raw[df_raw.columns[7:]]
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
df_ballots_a, df_ballots_raw_a, size_order_a = get_election_data(DATA_PATH, r'votes per ballot 2019a.csv', True, True,
                                                             True)
df_ballots_raw_a.loc[df_ballots_raw_a['שם ישוב'] == 'נצרת עילית', 'שם ישוב'] = 'נוף הגליל'


party_code_dict = {'פה': 'כחול לבן', 'מחל': 'הליכוד', 'ודעם': 'הרשימה המשותפת', 'ג': 'יהדות התורה', 'שס': 'ש"ס',
                   'ל': 'ישראל ביתנו', 'אמת': 'העבודה גשר', 'טב': 'ימינה', 'מרצ': 'המחנה הדמוקרטי', 'כף': 'עוצמה יהודית',
                   'ז': 'עוצמה כלכלית', 'זכ': 'מפלגת הדמוקראטורה', 'זן': 'יחד', 'זץ': 'צומת', 'י': 'מנהיגות חברתית',
                   'יז': 'אדום לבן', 'ינ': 'התנועה הנוצרית הליבראלית', 'יף': 'כבוד האדם', 'יק': 'מפלגת הגוש התנכ"י',
                   'כ': 'נעם', 'נ': 'מתקדמת', 'נך': 'כבוד ושוויון', 'נץ': 'כל ישראל אחים', 'כי': 'האחדות העממית',
                   'ףז': 'הפיראטים', 'צ': 'צדק', 'צן': 'צפון', 'ץ': 'דעם', 'ק': 'זכויותינו בקולנו', 'קך': 'סדר חדש',
                   'קץ': 'קמ"ה', 'רק': 'הימין החילוני', 'לה': 'לא הצביעו'}

party_code_dict_a = {'מחל': 'הליכוד', 'פה': 'כחול לבן', 'שס': 'ש"ס', 'ג': 'יהדות התורה', 'ום': 'חד"ש תע"ל',
                     'אמת': 'מפלגת העבודה', 'ל': 'ישראל ביתנו', 'טב': 'איחוד מפלגות הימין', 'מרצ': 'מרצ',
                     'כ': 'כולנו', 'דעם': 'רע"ם בל"ד', 'נ': 'הימין החדש', 'ז': 'זהות', 'נר': 'גשר', 'לה': 'לא הצביעו'}


def heatmapper(df, x_label, y_label, title, party_order_s, party_order_a):
    """
    A function that receives a dataframe and plots a heatmap of it.
    :param df: The dataframe.
    :param x_label: The label for the X axis.
    :param y_label: The label for the Y axis.
    :param title: The title for the heatmap.
    :param party_order_s: The order for the parties from the September election.
    :param party_order_a: The order for the parties from the April election.
    :return: None.
    """
    df = df[party_order_s]
    df = df.T[party_order_a].T
    # fixes the labels so they print correctly in Hebrew
    labels_s = [party_code_dict[i][::-1] for i in party_order_s]
    labels_a = [party_code_dict_a[i][::-1] for i in party_order_a]
    sns.heatmap(df, square=True, annot=False, cbar=True, xticklabels=labels_s, yticklabels=labels_a)
    plt.title(title, fontsize=13)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


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
apr_par_keys = [i + '_y' for i in size_order_a]
sep_parties_for_print = [party_code_dict[i][::-1] for i in size_order]
apr_parties_for_print = [party_code_dict_a[i][::-1] for i in size_order_a]
merged_ballots_df = merge_ballot_dfs(df_ballots_raw, df_ballots_raw_a)
merged_ballots_df.rename(columns={'ודעם': 'ודעם_x', 'כף': 'כף_x'}, inplace=True)
merged_ballots_df.rename(columns={'דעם': 'דעם_y', 'ום': 'ום_y', 'נר': 'נר_y'}, inplace=True)
n_sep = merged_ballots_df[sep_par_keys]
n_apr = merged_ballots_df[apr_par_keys]


def sankey(party_changes, apr_parties, sep_parties, title):
    """
    A function that creates a sankey graph for visualizing vote changes between the April and September elections.
    :param party_changes: The voting changes.
    :param apr_parties: A list of the largest parties in the April election.
    :param sep_parties: A list of the largest parties in the September election.
    :param title: Text to append to the title of the plot.
    :return: None.
    """
    apr_vals = np.tile(np.arange(len(apr_parties)), len(sep_parties))
    sep_vals = np.repeat(np.arange(len(apr_parties), len(apr_parties) + len(sep_parties)), len(apr_parties))
    colors = ['lightcoral', 'lightskyblue', 'darkgrey', 'palegreen', 'gold', 'cyan', 'orchid', 'orange',
              'tan', 'plum', 'gray']
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=10,
            thickness=8,
            label=apr_parties + sep_parties,
            color=['lightgray'] * len(apr_parties) + colors
        ),
        link=dict(
            source=apr_vals,
            target=sep_vals,
            value=party_changes.flatten(),
            color=[colors[col - len(apr_parties)] for col in sep_vals],
        ))])
    fig.update_layout(title_text='מעבר קולות בין מפלגות בין מערכת הבחירות באפריל לספטמבר - ' + title, font_size=8)
    fig.show()


########## Question 1 ##########


def transfer_matrix_no_constraints(apr_data, sep_data):
    """
    A function that calculates the transfer matrix M* by minimizing the Frobenius norm on the September and April
    election results.
    :param apr_data: The September election data.
    :param sep_data: The April election data.
    :return: The resulting M* matrix.
    """
    m_mat = cvx.Variable((len(apr_data.keys()), len(sep_data.keys())))
    cvx.Problem(cvx.Minimize(cvx.norm((apr_data.values @ m_mat) - sep_data, 'fro'))).solve(solver='SCS')
    return m_mat.value


party_order_sep = ['מרצ', 'אמת', 'פה', 'מחל', 'ודעם', 'שס', 'ל', 'ג', 'כף', 'טב']
party_order_apr = ['מרצ', 'אמת', 'נר', 'פה', 'כ', 'מחל', 'דעם', 'ום', 'שס', 'ל', 'ג', 'ז', 'טב', 'נ']


########## Question 2 ##########

n_sep_nv = copy.copy(n_sep)
n_sep_nv['לה'] = merged_ballots_df['בזב_x'] - merged_ballots_df['כשרים_x']
n_apr_nv = copy.copy(n_apr)
n_apr_nv['לה'] = merged_ballots_df['בזב_y'] - merged_ballots_df['כשרים_y']
so_nv = size_order + ['לה']
so_a_nv = size_order_a + ['לה']
po_s_nv = ['מרצ', 'אמת', 'פה', 'מחל', 'ודעם', 'שס', 'ג', 'כף', 'טב', 'ל', 'לה']
po_a_nv = ['מרצ', 'אמת', 'נר', 'פה', 'כ', 'מחל', 'דעם', 'ום', 'שס', 'ג', 'ז', 'טב', 'נ', 'ל', 'לה']
sep_parties_for_print_nv = [party_code_dict[i][::-1] for i in so_nv]
apr_parties_for_print_nv = [party_code_dict_a[i][::-1] for i in so_a_nv]


def get_ols_matrix(sep_df, apr_df, sep_order, apr_order):
    """
    A function that calculates the multivariate linear regression on the September and April election data.
    :param sep_df: The September election data.
    :param apr_df: The April election data.
    :param sep_order: The September party order.
    :param apr_order: The April party order.
    :return: The dataframe containing the results of the regression.
    """
    regression_df = pd.DataFrame((sep_df.T.values @ apr_df.values @ np.linalg.pinv(apr_df.T.values @ apr_df.values)).T,
                                 index=apr_order, columns=sep_order)
    return regression_df


########## Question 3 ##########


def get_nnls_mat(sep_data, apr_data):
    """
    A function that calculates and returns the results of performing non-negative least squares on the September and
    April election data.
    :param sep_data: The September election data.
    :param apr_data: The April election data.
    :return: The matrix containing the results of the nnls method.
    """
    return np.array([s_opt.nnls(apr_data.values, sep_data.values[:, i])[0] for i in range(len(sep_data.keys()))])


########## Question 4 ##########


def plot_mse_residuals_party_bars(sep_df, apr_df, sep_order, apr_order):
    """
    A function that calculates the residuals of the multivariate linear regression performed in question 2 and plots
    the mse by party.
    :param sep_df: The September election data.
    :param apr_df: The April election data.
    :param sep_order: The September party order.
    :param apr_order: The April party order.
    :return: None.
    """
    m_df_reg = pd.DataFrame((sep_df.T.values @ apr_df.values @ np.linalg.pinv(apr_df.T.values @ apr_df.values)).T,
                            index=apr_order, columns=sep_order)
    res_mat = pd.DataFrame(apr_df.values @ m_df_reg - sep_df.values, index=apr_df.index)
    mses = [(res_mat[i] ** 2).mean() for i in res_mat.keys()]
    plt.bar(np.arange(len(mses)), mses, color='b')
    plt.xticks(np.arange(len(mses)), [party_code_dict[i][::-1] for i in res_mat.keys()], rotation='vertical')
    for i in range(len(res_mat.keys())):
        plt.text(i - 0.25, mses[i] + 0.001, int(round(mses[i])))
    plt.suptitle('MSE for parties residuals after OLS')
    plt.show()


########## To Run Code: uncomment the relevant function call and run program. ##########


### Question 1 ###

### Part a ###
text_1a = 'Squared Loss Frobenius Norm'
m_df = pd.DataFrame(transfer_matrix_no_constraints(n_sep, n_apr).T, index=size_order_a, columns=size_order)
heatmapper(m_df, 'September Parties', 'April Parties', text_1a, party_order_sep, party_order_apr)
sankey(m_df.T.values * n_apr.sum(axis=0).values, apr_parties_for_print, sep_parties_for_print, text_1a)

### Part b ###
text_1b = 'Squared Loss Frobenius Norm w/ zeroing'
m_df = pd.DataFrame(transfer_matrix_no_constraints(n_sep, n_apr).T, index=size_order_a, columns=size_order)
filtered_m_df = m_df.where(m_df >= 0.005, 0)
heatmapper(filtered_m_df, 'September Parties', 'April Parties', text_1b, party_order_sep, party_order_apr)
sankey(filtered_m_df.T.values * n_apr.sum(axis=0).values, apr_parties_for_print, sep_parties_for_print, text_1b)


### Question 2 ###

### Part a ###
text_2a = 'OLS'
m_df_regression = get_ols_matrix(n_sep_nv, n_apr_nv, so_nv, so_a_nv)
heatmapper(m_df_regression, 'September Parties', 'April Parties', text_2a, po_s_nv, po_a_nv)
sankey(m_df_regression.T.values * n_apr_nv.sum(axis=0).values, apr_parties_for_print_nv, sep_parties_for_print_nv, text_2a)

### Part b ###
text_2b = 'OLS w/ zeroing'
m_df_regression = get_ols_matrix(n_sep_nv, n_apr_nv, so_nv, so_a_nv)
filt_m_df_reg = m_df_regression.where(m_df_regression >= 0.005, 0)
heatmapper(filt_m_df_reg, 'September Parties', 'April Parties', text_2b, po_s_nv, po_a_nv)
sankey(filt_m_df_reg.T.values * n_apr_nv.sum(axis=0).values, apr_parties_for_print_nv, sep_parties_for_print_nv, text_2b)


### Question 3 ###

### Part a ###
text_3a = 'NNLS'
m_mat_nnls = pd.DataFrame(get_nnls_mat(n_sep, n_apr), index=size_order, columns=size_order_a)
heatmapper(m_mat_nnls.T, 'September Parties', 'April Parties', text_3a, party_order_sep, party_order_apr)
sankey(m_mat_nnls.values * n_apr.sum(axis=0).values, apr_parties_for_print, sep_parties_for_print, text_3a)

### Part b ###
text_3b = 'NNLS w/ zeroing'
m_mat_nnls = pd.DataFrame(get_nnls_mat(n_sep, n_apr), index=size_order, columns=size_order_a)
filt_m_df_nnls = m_mat_nnls.where(m_mat_nnls >= 0.005, 0)
heatmapper(filt_m_df_nnls.T, 'September Parties', 'April Parties', text_3b, party_order_sep, party_order_apr)
sankey(filt_m_df_nnls.values * n_apr.sum(axis=0).values, apr_parties_for_print, sep_parties_for_print, text_3b)


### Question 4 ###

plot_mse_residuals_party_bars(n_sep_nv, n_apr_nv, so_nv, so_a_nv)
