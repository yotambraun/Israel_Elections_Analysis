import os
import pandas as pd
import numpy as np
import copy
from matplotlib import pyplot as plt
from scipy import optimize as s_opt
from scipy.stats import norm
import plotly.graph_objects as go
import cvxpy as cvx
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
import statsmodels.api as sm
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


########## Question 2 ##########


def perform_bootstrap(df, num_iters=100):
    """
    A function that performs bootstrapping on a dataframe and returns the results.
    :param df: The dataframe to peform the boostrapping on.
    :param num_iters: The number of bootstrap samples to produce.
    :return: The resulting bootstrap samples.
    """
    return [df.iloc[np.random.randint(len(df.index), size=len(df.index))] for i in range(num_iters)]


def find_low_p_val_pairs(res_mat, true_res_mat, a_keys, b_keys, with_params=False):
    """
    A function that finds and prints the party pairs with low p_values for them in when comparing two given matrices.
    :param res_mat: The approximating matrix.
    :param true_res_mat: The 'true' matrix.
    :param a_keys: The keys from the April election.
    :param b_keys: The keys from the September election.
    :param with_params: A boolean whether or not we need to call '.params' on the matrix elements to retrieve the
    relevant values.
    :return: None.
    """
    cnt = 0
    for index_a, party_a in enumerate(a_keys):
        for index_b, party_b in enumerate(b_keys):
            if with_params:
                res_a_b = [res_mat[b_iter][index_b][index_a] for b_iter in range(len(res_mat))]
            else:
                res_a_b = [res_mat[b_iter][index_b].params[index_a] for b_iter in range(len(res_mat))]
            std_a_b = np.std(res_a_b)
            if with_params:
                z_stat = ((sum(res_a_b) / len(res_a_b)) - true_res_mat.T[party_b][party_a]) / std_a_b
            else:
                z_stat = ((sum(res_a_b) / len(res_a_b)) - true_res_mat[index_b][index_a]) / std_a_b
            p_value = norm.sf(z_stat)
            if p_value < 0.0001:
                print(party_a.split('_')[0], party_b.split('_')[0], std_a_b, p_value)
                cnt += 1

            if party_a == party_b:
                z_stat_2 = ((sum(res_a_b) / len(res_a_b)) - 1) / std_a_b
                p_value_2 = norm.sf(z_stat_2)
                if 1 - p_value_2 < 0.05:
                    print(party_a.split('_')[0], party_b.split('_')[0], std_a_b, p_value)

    print(cnt)


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


common_ballot_indices = list(n_sep.index)
train, test = train_test_split(merged_ballots_df, test_size=0.2, train_size=0.8, shuffle=True)
train_sep = train[sep_par_keys]
train_apr = train[apr_par_keys]
test_sep = test[sep_par_keys]
test_apr = test[apr_par_keys]
train_sep_nv = copy.copy(train_sep)
train_sep_nv['לה'] = train['בזב_x'] - train['כשרים_x']
train_apr_nv = copy.copy(train_apr)
train_apr_nv['לה'] = train['בזב_y'] - train['כשרים_y']
so_nv = size_order + ['לה']
so_a_nv = size_order_a + ['לה']


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


########## Question 5 ##########


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
    fig.update_layout(title_text='מעבר קולות בין מפלגות בין מערכת הבחירות באפריל לספטמבר - ' + title, font_size=10)
    fig.show()


df_socio_raw = pd.read_csv(os.path.join(DATA_PATH, r'HevratiCalcaliYeshuvim.csv'), encoding='iso-8859-8',
                         index_col='רשות מקומית').sort_index()


def divide_by_socio(df, df_socio):
    """
    A function that returns the actual election result frequencies and the frequencies of the 10 largest parties and
    cities in common with the socio-economic data file. This is split by those cities with low socio-economic rating,
    those with high rating and those with no rating.
    :param df: The dataframe containing the election data.
    :param df_socio: The dataframe containing the socio-economic data.
    :return: The frequencies for the cities in low and high socio-economic brackets and those not in any.
    """
    intersect = df_socio.index & np.unique(df['שם ישוב'].values)
    df_com_ballots = df[df['שם ישוב'].isin(intersect)]
    df_other_ballots = df[df['שם ישוב'].isin([x for x in np.unique(df['שם ישוב'].values) if x not in intersect])]
    low_5 = df_socio[df_socio['מדד חברתי-'].isin([str(i) for i in range(1, 6)])]
    high_5 = df_socio[df_socio['מדד חברתי-'].isin([str(i) for i in range(6, 11)])]
    df_low_5 = df_com_ballots[df_com_ballots['שם ישוב'].isin(low_5.index & np.unique(df_com_ballots['שם ישוב'].values))]
    df_high_5 = df_com_ballots[df_com_ballots['שם ישוב'].isin(high_5.index & np.unique(df_com_ballots['שם ישוב'].values))]
    return df_low_5, df_high_5, df_other_ballots


low_df, high_df, other_df = divide_by_socio(merged_ballots_df, df_socio_raw)
low_train = low_df[low_df.index.isin(train.index)]
low_train_s = low_train[sep_par_keys]
low_train_a = low_train[apr_par_keys]
low_test = low_df[low_df.index.isin(test.index)]
low_test_s = low_test[sep_par_keys]
low_test_a = low_test[apr_par_keys]
high_train = high_df[high_df.index.isin(train.index)]
high_train_s = high_train[sep_par_keys]
high_train_a = high_train[apr_par_keys]
high_test = high_df[high_df.index.isin(test.index)]
high_test_s = high_test[sep_par_keys]
high_test_a = high_test[apr_par_keys]
other_train = other_df[other_df.index.isin(train.index)]
other_train_s = other_train[sep_par_keys]
other_train_a = other_train[apr_par_keys]
other_test = other_df[other_df.index.isin(test.index)]
other_test_s = other_test[sep_par_keys]
other_test_a = other_test[apr_par_keys]

party_order_sep = ['מרצ', 'אמת', 'פה', 'מחל', 'ודעם', 'שס', 'ל', 'ג', 'כף', 'טב']
party_order_apr = ['מרצ', 'אמת', 'נר', 'פה', 'כ', 'מחל', 'דעם', 'ום', 'שס', 'ל', 'ג', 'ז', 'טב', 'נ']


########## To Run Code: uncomment the relevant function call and run program. ##########


### Question 1 ###

ols_res = [sm.OLS(n_sep[i].values, n_apr.values).fit() for i in sep_par_keys]
ols_res_bse = [ols_res[i].bse / np.linalg.norm(ols_res[i].params) for i in range(len(ols_res))]
p_val_mat = [(size_order[i], size_order_a[j], norm.sf(ols_res[i].params[j] / ols_res_bse[i][j]), ols_res_bse[i][j])
             for i in range(len(size_order)) for j in range(len(size_order_a))]
cnt = 0
for i in p_val_mat:
    if i[2] < 0.0001:
        print(i)
        cnt += 1
print(cnt)


### Question 2 ###

t_ols_res = [sm.OLS(n_sep[i].values, n_apr.values).fit() for i in sep_par_keys]
true_ols_res_normalized = [t_ols_res[i].params / np.linalg.norm(t_ols_res[i].params) for i in range(len(t_ols_res))]
boot_mats = perform_bootstrap(n_apr)
ols_res_bs = [[sm.OLS(n_sep[i].values, boot_mats[j].values).fit() for i in sep_par_keys] for j in range(len(boot_mats))]
find_low_p_val_pairs(ols_res_bs, true_ols_res_normalized, apr_par_keys, sep_par_keys)


### Question 3 ###

true_nnls = pd.DataFrame(get_nnls_mat(n_sep, n_apr), index=size_order, columns=size_order_a)
true_filt_m_df_nnls = true_nnls.where(true_nnls >= 0.005, 0)
true_filt_m_df_nnls = true_filt_m_df_nnls.divide(np.linalg.norm(true_filt_m_df_nnls, axis=1), axis=0)
boot_ms = perform_bootstrap(n_apr)
nnls_res_bs = [get_nnls_mat(n_sep, boot_ms[i]) for i in range(len(boot_ms))]
find_low_p_val_pairs(nnls_res_bs, true_filt_m_df_nnls, size_order_a, size_order, with_params=True)


### Question 4 ###

m_df_fro = pd.DataFrame(transfer_matrix_no_constraints(train_sep, train_apr).T, index=size_order_a, columns=size_order)
print("MSE using Frobenius norm: {}".format(((test_apr.values @ m_df_fro.values - test_sep.values) ** 2).mean()))

m_df_regression = get_ols_matrix(train_sep_nv, train_apr_nv, so_nv, so_a_nv)
print("MSE using OLS: {}".format(((test_apr.values @ m_df_regression.drop('לה', axis=1).drop('לה', axis=0).values -
                                   test_sep.values) ** 2).mean()))

m_mat_nnls = pd.DataFrame(get_nnls_mat(train_sep, train_apr), index=size_order, columns=size_order_a)
print("MSE using NNLS: {}".format(((test_apr.values @ m_mat_nnls.values.T - test_sep.values) ** 2).mean()))


### Question 5 ###

m_mat_nnls_l = pd.DataFrame(get_nnls_mat(low_train_s, low_train_a), index=size_order, columns=size_order_a)
print("MSE using NNLS low: {}".format(((low_test_a.values @ m_mat_nnls_l.values.T - low_test_s.values) ** 2).mean()))

m_mat_nnls_h = pd.DataFrame(get_nnls_mat(high_train_s, high_train_a), index=size_order, columns=size_order_a)
print("MSE using NNLS high: {}".format(((high_test_a.values @ m_mat_nnls_h.values.T - high_test_s.values) ** 2).mean()))

m_mat_nnls_o = pd.DataFrame(get_nnls_mat(other_train_s, other_train_a), index=size_order, columns=size_order_a)
print("MSE using NNLS other: {}".format(((other_test_a.values @ m_mat_nnls_o.values.T - other_test_s.values) **
                                         2).mean()))

heatmapper(m_mat_nnls_l.T, 'September Parties', 'April Parties', 'Cities with low socio-economic ratings',
           party_order_sep, party_order_apr)
sankey(m_mat_nnls_l.values * low_test_a.sum(axis=0).values, apr_parties_for_print, sep_parties_for_print,
       'Cities with low socio-economic ratings')
heatmapper(m_mat_nnls_h.T, 'September Parties', 'April Parties', 'Cities with high socio-economic ratings',
           party_order_sep, party_order_apr)
sankey(m_mat_nnls_h.values * high_test_a.sum(axis=0).values, apr_parties_for_print, sep_parties_for_print,
       'Cities with high socio-economic ratings')
heatmapper(m_mat_nnls_o.T, 'September Parties', 'April Parties', 'Cities without socio-economic ratings',
           party_order_sep, party_order_apr)
sankey(m_mat_nnls_o.values * other_test_a.sum(axis=0).values, apr_parties_for_print, sep_parties_for_print,
       'Cities without socio-economic ratings')


### Question 6 (Bonus) ###

# Ridge
ridge_models = [Ridge().fit(train_apr, train_sep[i]) for i in sep_par_keys]
m_ridge = np.array([ridge_models[i].coef_ for i in range(len(sep_par_keys))])
print("MSE using Ridge: {}".format(((test_apr.values @ m_ridge.T - test_sep.values) ** 2).mean()))

ridge_models_l = [Ridge().fit(low_train_a, low_train_s[i]) for i in sep_par_keys]
m_ridge_l = np.array([ridge_models_l[i].coef_ for i in range(len(sep_par_keys))])
print("MSE using Ridge low: {}".format(((low_test_a.values @ m_ridge_l.T - low_test_s.values) ** 2).mean()))

ridge_models_h = [Ridge().fit(high_train_a, high_train_s[i]) for i in sep_par_keys]
m_ridge_h = np.array([ridge_models_h[i].coef_ for i in range(len(sep_par_keys))])
print("MSE using Ridge high: {}".format(((high_test_a.values @ m_ridge_h.T - high_test_s.values) ** 2).mean()))

ridge_models_o = [Ridge().fit(other_train_a, other_train_s[i]) for i in sep_par_keys]
m_ridge_o = np.array([ridge_models_o[i].coef_ for i in range(len(sep_par_keys))])
print("MSE using Ridge other: {}".format(((other_test_a.values @ m_ridge_o.T - other_test_s.values) ** 2).mean()))


# Lasso
lasso_models = [Lasso().fit(train_apr, train_sep[i]) for i in sep_par_keys]
m_lasso = np.array([lasso_models[i].coef_ for i in range(len(sep_par_keys))])
print("MSE using Lasso: {}".format(((test_apr.values @ m_lasso.T - test_sep.values) ** 2).mean()))

lasso_models_l = [Lasso(alpha=0.0001, positive=True).fit(low_train_a, low_train_s[i]) for i in sep_par_keys]
m_lasso_l = np.array([lasso_models_l[i].coef_ for i in range(len(sep_par_keys))])
print("MSE using Lasso low: {}".format(((low_test_a.values @ m_lasso_l.T - low_test_s.values) ** 2).mean()))

lasso_models_h = [Lasso(alpha=0.0001, positive=True).fit(high_train_a, high_train_s[i]) for i in sep_par_keys]
m_lasso_h = np.array([lasso_models_h[i].coef_ for i in range(len(sep_par_keys))])
print("MSE using Lasso high: {}".format(((high_test_a.values @ m_lasso_h.T - high_test_s.values) ** 2).mean()))

lasso_models_o = [Lasso(alpha=0.0001, positive=True).fit(other_train_a, other_train_s[i]) for i in sep_par_keys]
m_lasso_o = np.array([lasso_models_o[i].coef_ for i in range(len(sep_par_keys))])
print("MSE using Lasso other: {}".format(((other_test_a.values @ m_lasso_o.T - other_test_s.values) ** 2).mean()))
