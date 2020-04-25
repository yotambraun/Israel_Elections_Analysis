import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as s_opt
from sklearn.model_selection import KFold
from sklearn.utils import shuffle


DATA_PATH = 'C:\\Users\\Meir\\PycharmProjects\\Stat_Lab'
SPLITS = [i for i in range(1, 51)]
NUM_FOLD = 10
NUM_REPS = 50


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


########## Questions 1 and 2 ##########


def divide_into_groups(df, num_splits):
    """
    A function that receives a dataframe and splits into num_splits equally sized groups.
    :param df: The dataframe to split.
    :param num_splits: The number of equally sized splits.
    :return: A list of the splits in the dataframe.
    """
    # the indices to divide the dataframe at
    batch_dividers = [round(len(df) / num_splits) * i for i in range(num_splits)]
    batch_dividers.append(len(df))
    divided_dfs = [df.iloc[batch_dividers[i]:batch_dividers[i + 1]] for i in range(len(batch_dividers[:-1]))]
    return divided_dfs


def get_nnls_mat(sep_data, apr_data):
    """
    A function that calculates and returns the results of performing non-negative least squares on the September and
    April election data.
    :param sep_data: The September election data.
    :param apr_data: The April election data.
    :return: The matrix containing the results of the nnls method.
    """
    return np.array([s_opt.nnls(apr_data.values, sep_data.values[:, i])[0] for i in range(len(sep_data.keys()))])


def get_train_test_mses(apr_freq_df, apr_df, sep_df, split_idxs=SPLITS, num_folds=NUM_FOLD):
    """
    A function that calculates the train and test MSEs for the different splits using k-fold cross validation.
    :param apr_freq_df: The dataframe containing the frequencies for the April election.
    :param apr_df: The dataframe containing the election results for the April election.
    :param sep_df: The dataframe containing the election results for the September election.
    :param split_idxs: An array for the different numbers of splits.
    :param num_folds: The number of cross validations to perform.
    :return: The train MSEs and the test MSEs.
    """
    total_train_mses = []
    total_test_mses = []
    for j in split_idxs:
        df_divisions = divide_into_groups(apr_freq_df, num_splits=j)
        apr_divided_ballots = [apr_df.loc[df_divisions[i].index] for i in range(len(df_divisions))]
        sep_divided_ballots = [sep_df.loc[df_divisions[i].index] for i in range(len(df_divisions))]
        train_mses = []
        test_mses = []
        for k in range(j):
            kf = KFold(num_folds, shuffle=True).split(apr_divided_ballots[k])
            for train_apr_ind, test_apr_ind in kf:
                train_apr = apr_divided_ballots[k].iloc[train_apr_ind]
                test_apr = apr_divided_ballots[k].iloc[test_apr_ind]
                train_sep = sep_divided_ballots[k].loc[train_apr.index]
                test_sep = sep_divided_ballots[k].loc[test_apr.index]
                train_mses.append(((train_apr.values @ get_nnls_mat(train_sep, train_apr).T - train_sep.values) ** 2).mean())
                test_mses.append(((test_apr.values @ get_nnls_mat(train_sep, train_apr).T - test_sep.values) ** 2).mean())
        total_train_mses.append(np.mean(train_mses))
        total_test_mses.append(np.mean(test_mses))
    return total_train_mses, total_test_mses


def plot_train_test_mses(train_mses, test_mses):
    """
    A function that plots the given train and test MSEs.
    :param train_mses: The train MSEs.
    :param test_mses: The test MSEs.
    :return: None.
    """
    plt.plot(np.arange(1, len(train_mses) + 1), train_mses, label='Train', c='b')
    plt.plot(np.arange(1, len(test_mses) + 1), test_mses, label='Test', c='r')
    plt.legend()
    plt.xlabel('Number of Splits')
    plt.ylabel('Average MSE')
    plt.title('Train and Test average MSE as function of number of splits')
    plt.show()


right_wing_parties = ['מחל_y', 'שס_y', 'ג_y', 'נ_y', 'טב_y', 'ז_y']
n_apr_freq = n_apr.divide(merged_ballots_df['כשרים_y'], axis=0)


########## Question 3 ##########


def mse_sample_comparison(df, df_raw, num_reps=NUM_REPS):
    """
    A function that randomly samples different number of ballots num_reps times and calculates the averages and
    standard deviations of the MSEs and plots them.
    :param df: The dataframe containing the election data.
    :param df_raw: The raw dataframe containing the election data.
    :param num_reps: The number of repititions for each sample size.
    :return: None.
    """
    par = df[size_order].sum().div(df_raw['כשרים'].sum())
    r_vals = [i * 5 for i in range(1, 21)]
    avg_mses = np.zeros(len(r_vals))
    std_mses = np.zeros(len(r_vals))
    for i, r in enumerate(r_vals):
        rep_mses = np.zeros(NUM_REPS)
        for rep in range(num_reps):
            samp = df.sample(r)[size_order]
            samp = samp.sum().divide(df_raw['כשרים'][samp.index].sum())
            rep_mses[rep] = ((par - samp) ** 2).mean()
        avg_mses[i] = rep_mses.mean()
        std_mses[i] = rep_mses.std()
    plt.errorbar(r_vals, avg_mses, yerr=std_mses, c='b', fmt='-o')
    plt.xlabel('Size of Samples')
    plt.ylabel('Average of MSEs for Samples')
    plt.title('Average of MSEs as Function of Size of Samples w/ Error Bars')
    plt.show()


########## To Run Code: uncomment the relevant function call and run program. ##########


### Question 1 ###

n_apr_freq['pct_right'] = n_apr_freq[right_wing_parties].sum(axis=1)
n_apr_freq_sorted = n_apr_freq.sort_values('pct_right', ascending=False)
train_mse, test_mse = get_train_test_mses(n_apr_freq_sorted, n_apr, n_sep)
plot_train_test_mses(train_mse, test_mse)
print("Num splits that minimizes avg test MSE: {} with avg MSE value of: {}".
      format(test_mse.index(min(test_mse)) + 1, min(test_mse)))
print("Num splits that minimizes avg train MSE: {} with avg MSE value of: {}".
      format(train_mse.index(min(train_mse)) + 1, min(train_mse)))


### Question 2 ###

# similar to question 1, but now we are choosing the different ballot groups at random
n_apr_freq_shuffled = shuffle(n_apr_freq)
train_mse, test_mse = get_train_test_mses(n_apr_freq_shuffled, n_apr, n_sep)
plot_train_test_mses(train_mse, test_mse)
print("Num splits that minimizes avg test MSE: {} with avg MSE value of: {}".
      format(test_mse.index(min(test_mse)) + 1, min(test_mse)))
print("Num splits that minimizes avg train MSE: {} with avg MSE value of: {}".
      format(train_mse.index(min(train_mse)) + 1, min(train_mse)))


### Question 3 ###

mse_sample_comparison(df_ballots, df_ballots_raw)
