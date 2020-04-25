import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

DATA_PATH = 'C:\\Users\\Meir\\PycharmProjects\\Stat_Lab\\Week_1'

df_sep_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per city 2019b.csv'), encoding='iso-8859-8',
                         index_col='שם ישוב').sort_index()


def invalid_percentages(df):
    """
    A function that plots the percentages of invalid votes of the total votes by city
    :param df: The dataframe containing the election data.
    :return: None.
    """
    part_1_1 = df[['פסולים',  'מצביעים']]
    part_1_1['pct_invalid'] = part_1_1['פסולים'] / part_1_1['מצביעים']

    # Find the city with the largest percentage of disqualified votes.
    city = part_1_1.nlargest(1, 'pct_invalid')
    print("The city with the largest percentage of disqualified votes is: " + city.index[0] + " with: " +
          str(round(city['pct_invalid'][city.index[0]] * 100, 3)) + "% of its total votes disqualified.")

    plt.hist(part_1_1['pct_invalid'], 100, color='b')
    plt.xlabel("Share of invalid votes")
    plt.ylabel("Number of cities with X percentage of invalid votes")
    plt.title("Bar chart of invalid votes")
    plt.show()


def two_city_hist(df, city1, city2, above_threshold=True, threshold=0.0325):
    """
    A function that compares voting patterns in 2 cities.
    :param df: The dataframe containing the election data.
    :param city1: The first city.
    :param city2: The second city.
    :param above_threshold: A boolean parameter whether to include all parties or only those that passed the threshold.
    :param threshold: The threshold to pass if above_threshold is True. Default is 3.25%.
    :return: None.
    """
    # Filters the dataframe to only include the columns representing parties
    parties_only = df[[df.keys()[i] for i in range(6, len(df.keys()))]]
    # Calculates the percentage of valid votes received by each party
    par = parties_only.sum().div(df['כשרים'].sum().sum()).sort_values(ascending=False)
    if above_threshold:
        # Selects only those that passed the threshold
        par = par[par > threshold]

    # Calculates the percentages of votes per party in each of the chosen cities
    city1_res = df.loc[city1, par.keys()] / df['כשרים'][city1]
    city2_res = df.loc[city2, par.keys()] / df['כשרים'][city2]

    width = 0.3
    fig, ax = plt.subplots()
    city1_bar = ax.bar(np.arange(len(par.keys())), list(city1_res), width, color='b')
    city2_bar = ax.bar(np.arange(len(par.keys())) + width, list(city2_res), width, color='r')
    ax.set_xticks(np.arange(len(par.keys())))
    ax.set_xticklabels(name[::-1] for name in par.keys())
    ax.legend((city1_bar[0], city2_bar[0]), (city1[::-1], city2[::-1]))
    ax.set_xlabel("Votes for each party" + ("(that passed the threshold)" if above_threshold else ""))
    ax.set_ylabel("Share of votes each party received in each city")
    ax.set_title("Voting pattern comparison of {} and {}".format(city1[::-1], city2[::-1]))
    plt.show()


def multinomial_distance(df, is_min=True, closed_envelopes=True):
    """
    A function that calculates the distance between each city/town and the whole country, according the the
    quadratic function provided in the exercise description.
    :param df: A dataframe containing the election data.
    :param is_min: A boolean parameter whether to plot the city with the min or max distance.
    :param closed_envelopes: A boolean parameter whether to include the 'מעטפות חיצוניות' or not.
    :return: None.
    """
    # Filters the dataframe to only include the columns representing parties
    parties_only = df[[df.keys()[i] for i in range(6, len(df.keys()))]]
    # Calculates the percentage of valid votes received by each party
    par = parties_only.sum().div(df['כשרים'].sum().sum()).sort_values(ascending=False)

    # Creates a dictionary where the key is the city name and the value is the distance between said city and the
    # whole country, as per the provided function in the exercise description
    distance_dict = {city: ((parties_only.loc[city] / df_sep_raw['כשרים'][city] - parties_only.sum() /
                             df_sep_raw['כשרים'].sum()) ** 2).sum() for city in df_sep_raw.index}

    # Since the smallest 'distance' is achieved by the 'מעטפות חיצוניות', we choose the actual city/town with the
    # smallest distance
    if not closed_envelopes:
        distance_dict.pop('מעטפות חיצוניות')

    if is_min:
        city = min(distance_dict, key=distance_dict.get)
    else:
        city = max(distance_dict, key=distance_dict.get)

    print("The city/town with the {} 'distance score' from the countrywide voting pattern is: {}".format(
        "minimum" if is_min else "maximum", city))
    print("The 'distance score' of {} from the country as a whole is {}".format(city, distance_dict[city]))

    # Calculates the percentages of votes per party in each of the chosen city
    city_res = df.loc[city, par.keys()] / df['כשרים'][city]

    width = 0.3
    fig, ax = plt.subplots()
    city_bar = ax.bar(np.arange(len(par.keys())), list(city_res), width, color='b')
    all_bar = ax.bar(np.arange(len(par.keys())) + width, list(par), width, color='r')
    ax.set_xticks(np.arange(len(par.keys())))
    ax.set_xticklabels([name[::-1] for name in par.keys()], rotation='vertical')
    ax.legend((city_bar[0], all_bar[0]), (city[::-1], 'ישראל'[::-1]))
    ax.set_xlabel("Full party list")
    ax.set_ylabel("Share of per party in {} and {}".format(city[::-1], 'ישראל'[::-1]))
    ax.set_title("Voting pattern comparison of {} and {}".format(city[::-1], 'ישראל'[::-1]))
    plt.show()


### To Run Code: uncomment the relevant function call and run program. ###

### Question 1 ###
invalid_percentages(df_sep_raw)

### Question 2 ###
two_city_hist(df_sep_raw, 'ירושלים', 'חיפה')
two_city_hist(df_sep_raw, 'רמת גן', 'גבעתיים')
two_city_hist(df_sep_raw, 'תל אביב - יפו', 'בני ברק')

### Question 3 ###
multinomial_distance(df_sep_raw)                          # for the min distance
multinomial_distance(df_sep_raw, closed_envelopes=False)  # for the min distance without 'מעטפות חיצוניות'
multinomial_distance(df_sep_raw, is_min=False)            # for the max distance
