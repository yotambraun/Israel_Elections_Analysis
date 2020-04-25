import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm


DATA_PATH = 'C:\\Users\\Meir\\PycharmProjects\\Stat_Lab'

df_ballots_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per ballot 2019b.csv'), encoding='iso-8859-8',
                         index_col='ברזל').sort_index()
df_ballots_raw = df_ballots_raw[df_ballots_raw['שם ישוב'] != 'מעטפות חיצוניות']
df_ballots = df_ballots_raw.drop('סמל ועדה', axis=1)
df_ballots = df_ballots[df_ballots.columns[8:]]
size_order = list(df_ballots[[df_ballots.keys()[i] for i in range(1, len(df_ballots.keys()))]].sum().sort_values(
    ascending=False).keys())[:10]
df_ballots = df_ballots[size_order]


party_code_dict = {'פה': 'כחול לבן', 'מחל': 'הליכוד', 'ודעם': 'הרשימה המשותפת', 'ג': 'יהדות התורה', 'שס': 'ש"ס',
                   'ל': 'ישראל ביתנו', 'אמת': 'העבודה גשר', 'טב': 'ימינה', 'מרצ': 'המחנה הדמוקרטי', 'כף': 'עוצמה יהודית',
                   'ז': 'עוצמה כלכלית', 'זכ': 'מפלגת הדמוקראטורה', 'זן': 'יחד', 'זץ': 'צומת', 'י': 'מנהיגות חברתית',
                   'יז': 'אדום לבן', 'ינ': 'התנועה הנוצרית הליבראלית', 'יף': 'כבוד האדם', 'יק': 'מפלגת הגוש התנכ"י',
                   'כ': 'נעם', 'נ': 'מתקדמת', 'נך': 'כבוד ושוויון', 'נץ': 'כל ישראל אחים', 'כי': 'האחדות העממית',
                   'ףז': 'הפיראטים', 'צ': 'צדק', 'צן': 'צפון', 'ץ': 'דעם', 'ק': 'זכויותינו בקולנו', 'קך': 'סדר חדש',
                   'קץ': 'קמ"ה', 'רק': 'הימין החילוני'}

########## Question 1 ##########


participation = df_ballots_raw['כשרים'] / df_ballots_raw['בזב']
participation[participation > 1] = 1
p_v_i = dict(zip(list(df_ballots.index), participation))
p_alpha_j = np.arange(0.5, 1, 0.05)
np.random.shuffle(p_alpha_j)
dict_party_value = {size_order[i]: round(p_alpha_j[i], 2) for i in range(len(size_order))}


def nij_t_calculator(df):
    """
    A function that performs the calculation for the nij tilde correction we leared in class.
    :param df: The dataframe to perform the correction on.
    :return: The dataframe containing the corrections.
    """
    nij_scalar = df_ballots_raw['בזב'].sum() / df_ballots_raw['כשרים'].sum()
    nij_t_df = pd.DataFrame(round(df * nij_scalar), columns=size_order)
    nij_t_df.rename(index={i + 1: list(df.index)[i] for i in range(len(df_ballots))}, inplace=True)
    return nij_t_df


def perform_simulation(df, pij):
    """
    A function that performs a single simulation using the binomial distribution.
    :param df: The dataframe to simulate from.
    :param pij: The probabilities for the binomial distribution samplings.
    :return: A dataframe containing the simulation results.
    """
    if len(pij) == len(df.keys()):
        nij_df = pd.DataFrame(np.array([np.random.binomial(df[j], pij[j]) for j in df.keys()]).transpose(),
                              columns=size_order)
    else:
        nij_df = pd.DataFrame(np.array([np.random.binomial(df[df.keys()][j].values.astype(int), list(pij.values()))
                                        for j in df.keys()]).transpose(), columns=size_order)
    nij_df.rename(index={i: list(df.index)[i] for i in range(len(df_ballots))}, inplace=True)
    return nij_df


def perform_simulations(nij_t, pij, with_errs=False, num_sims=25):
    """
    A function that performs multiple simulations and averages them.
    :param nij_t: The dataframe on which to perform the simulations.
    :param pij: The probabilities for the binomial distribution samplings.
    :param with_errs: A boolean whether to calculate the standard deviation errors or not.
    :param num_sims: The number of simulations to run.
    :return: The dataframes representing the original, estimated and scaled percentages, and the standard deviation
    errors if with_errs is True, else returns None for this parameter.
    """
    nij_turnout = df_ballots_raw['כשרים'] / df_ballots_raw['בזב']

    nij = perform_simulation(nij_t, pij)
    nij_scaled = (nij[~nij.index.duplicated()].transpose() / nij_turnout).transpose().replace(np.inf, 0)

    party_std_dict = {j: [[nij[j].sum()], [nij_scaled[j].sum()]] for j in nij_t.keys()}
    for i in range(num_sims - 1):
        nij_tmp = perform_simulation(nij_t, pij)
        nij_scaled_tmp = (nij_tmp[~nij_tmp.index.duplicated()].transpose() / nij_turnout).transpose().replace(np.inf, 0)
        for j in nij_t.keys():
            party_std_dict[j][0].append(nij_tmp[j].sum())
            party_std_dict[j][1].append(nij_scaled_tmp[j].sum())
        nij += nij_tmp
        nij_scaled += nij_scaled_tmp
    nij /= num_sims
    nij_scaled /= num_sims
    nij_par = nij.sum().div(df_ballots_raw['כשרים'].sum())

    # Filters the dataframe to only include the columns representing parties
    parties_only = df_ballots[[df_ballots.keys()[i] for i in range(len(df_ballots.keys()))]]
    # Calculates the percentage of valid votes received by each party
    par = parties_only.sum().div(df_ballots_raw['כשרים'].sum()).sort_values(ascending=False)

    # Filters the dataframe to only include the columns representing parties, with scaled values
    scaled_parties_only = nij_scaled[[nij_scaled.keys()[i] for i in range(len(nij_scaled.keys()))]]
    # Calculates the percentage of valid votes received by each party, after scaling
    scaled_par = scaled_parties_only.sum().div(df_ballots_raw['בזב'].sum()).sort_values(ascending=False)

    (par, scaled_par) = (par[:10], scaled_par[:10])

    if with_errs:
        stds = [[np.std(party_std_dict[j][0]) / parties_only[j].sum() for j in party_std_dict.keys()],
                [np.std(party_std_dict[j][1]) / parties_only[j].sum() for j in party_std_dict.keys()]]
    return nij_par, par, scaled_par, stds if with_errs else None


def plot_bars(nij_par, par, scaled_par, stds, title, legend_values=('nij scaled', 'original', 'nij')):
    """
    A function that receives 3 dataframes and plots them side by side as bar charts for comparison.
    :param nij_par: The dataframe containing the estimated values.
    :param par: The dataframe containing the original (actual) values.
    :param scaled_par: The dataframe containing the scaled (estimated) values.
    :param stds: The standard deviation errors (could be None if we don't want to show it).
    :param title: The title for the graph.
    :param legend_values: The strings to put inside the legend.
    :return: None.
    """
    width = 0.3
    fig, ax = plt.subplots()

    std, scaled_std = (stds[0], stds[1]) if stds else (None, None)
    scaled_bar = ax.bar(np.arange(len(scaled_par.keys())), list(scaled_par), width, color='b', yerr=scaled_std)
    original_bar = ax.bar(np.arange(len(par.keys())) + width, list(par), width, color='r')
    nij_bar = ax.bar(np.arange(len(nij_par.keys())) - width, list(nij_par), width, color='g', yerr=std)

    ax.set_xticks(np.arange(len(par.keys())))
    ax.set_xticklabels([party_code_dict[name][::-1] for name in par.keys()], rotation='vertical')
    ax.legend((scaled_bar[0], original_bar[0], nij_bar[0]), legend_values)
    ax.set_xlabel("Top 10 parties")
    ax.set_ylabel("Vote share per party")
    ax.set_title(title, fontsize=12)
    plt.show()


########## Question 2 ##########


def calculate_regression_df(df, df_raw):
    """
    A function that calculates the OLS params (which for us are also known as alpha j inverse values).
    :param df: The dataframe with only the relevant data.
    :param df_raw: The raw dataframe containing additional columns.
    :return: The alpha j inverse values.
    """
    ols_model = sm.OLS(df_raw['בזב'], df)
    res = ols_model.fit()
    return res.params


def perform_regression_simulations(df, df_raw, pij, num_sims=25):
    """
    A function that simulates the voting distributions using the OLS method.
    :param df: The dataframe with only the relevant data.
    :param df_raw: The raw dataframe containing additional columns.
    :param pij: The probabilities for the binomial distribution samplings.
    :param num_sims: The number of simulations to run.
    :return: The results of the perform_simulations() function.
    """
    ols_res = calculate_regression_df(df, df_raw)
    nij_est = pd.DataFrame(np.array([ols_res * df.loc[i] for i in df.index]).astype(int), columns=size_order)
    nij_est.rename(index={i: list(df.index)[i] for i in range(len(df_ballots))}, inplace=True)
    return perform_simulations(nij_est, pij, num_sims=num_sims)


########## Question 3 ##########


def comparison_of_corrections(df, df_raw, pij, num_sims=25):
    """
    A function performs a comparison of different correction methods we learned.
    :param df: The dataframe with only the relevant data.
    :param df_raw: The raw dataframe containing additional columns.
    :param pij: The probabilities for the binomial distribution samplings.
    :param num_sims: The number of simulations to run.
    :return: The dataframes representing the original, estimated and scaled percentages.
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

    ols_res = calculate_regression_df(df, df_raw)

    nij_est = pd.DataFrame(np.array([ols_res * df.loc[i] for i in df.index]).astype(int), columns=size_order)
    nij_est.rename(index={i: list(df.index)[i] for i in range(len(df_ballots))}, inplace=True)

    nij = perform_simulation(nij_est, pij)
    for i in range(num_sims - 1):
        nij += perform_simulation(nij_est, pij)
    nij /= num_sims
    nij_par = nij.sum().div(df_ballots_raw['כשרים'].sum())

    return nij_par, par, scaled_par


########## To Run Code: uncomment the relevant function call and run program. ##########


### Question 1 ###

nij_t = nij_t_calculator(df_ballots)

nij_par1, par1, scaled_par1, stds = perform_simulations(nij_t, p_v_i, with_errs=True)
title_1 = "Correction using binomial distribution using v_i values."
plot_bars(nij_par1, par1, scaled_par1, stds, title_1)

nij_par2, par2, scaled_par2, stds = perform_simulations(nij_t, dict_party_value, with_errs=True)
title_2 = "Correction using binomial distribution using alpha_j values."
plot_bars(nij_par2, par2, scaled_par2, stds, title_2)


### Question 2 ###

nij_par3, par3, scaled_par3, stds = perform_regression_simulations(df_ballots, df_ballots_raw, p_v_i)
title_3 = "Correction using OLS regression using v_i values."
plot_bars(nij_par3, par3, scaled_par3, stds, title_3)

nij_par4, par4, scaled_par4, stds = perform_regression_simulations(df_ballots, df_ballots_raw, dict_party_value)
title_4 = "Correction using OLS regression alpha_j values."
plot_bars(nij_par4, par4, scaled_par4, stds, title_4)


### Question 3 ###

legend_values = ('scaled', 'original', 'ols')

nij_par5, par5, scaled_par5 = comparison_of_corrections(df_ballots, df_ballots_raw, p_v_i)
title_5 = "OLS regression vs. Lab 2 corrections using v_i values."
plot_bars(nij_par5, par5, scaled_par5, None, title_5, legend_values)

nij_par6, par6, scaled_par6 = comparison_of_corrections(df_ballots, df_ballots_raw, dict_party_value)
title_6 = "OLS regression vs. Lab 2 corrections using alpha_i values."
plot_bars(nij_par6, par6, scaled_par6, None, title_6, legend_values)
