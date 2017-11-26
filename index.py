import csv
import numpy as np
import scipy.stats
from scipy import integrate
import math

social_state = []
state_1 = 0
state_2 = 0
state_3 = 0
state_4 = 0
state_5 = 0
p1 = 0
p2 = 0
p = 0
p0 = 0.5
q = 0
alpha = 0.05


def reader_csv():
    with open('Analytic.csv') as file_csv:
        reader = csv.DictReader(file_csv)
        for row in reader:
            social_state.append(row['SE'])


def count_of_state():
    global state_1, state_2, state_3, state_4, state_5, p1, p2, q, p
    for state in social_state:
        if int(state) == 1:
            state_1 += 1
        elif int(state) == 2:
            state_2 += 1
        elif int(state) == 3:
            state_3 += 1
        elif int(state) == 4:
            state_4 += 1
        elif int(state) == 5:
            state_5 += 1
    p = state_3 / len(social_state)
    q = 1 - p


def binomial_dist(prob, state, len_all_state):
    global p1, p2
    for i in range(state, len_all_state):
        p1 += scipy.stats.binom.pmf(i, len_all_state, prob)

    for i in range(0, state):
        p2 += scipy.stats.binom.pmf(i, len_all_state, prob)

    print(p1, "+", p2, "=", p1 + p2, end='\n')
    P = 2 * min(p1, p2)
    if P < alpha and P != alpha:
        print("alternative hypothesis is not accepted")
    else:
        print("alternative hypothesis is accepted")
    print(P)


def z_crit(p0, p, len_soc_state):
    z0 = (p - p0) / math.sqrt((p0 * (1 - p0)) / len_soc_state)
    integ = integrate.quad(lambda x: math.exp(-((x ** 2) / 2)), abs(z0), np.inf)
    print(integ[0] / math.sqrt(2 * math.pi))


def chi_square(prob, state, len_all_state):
    chi = ((state - len_all_state * prob) ** 2) / (len_all_state * prob)
    P = integrate.quad((lambda x: scipy.stats.chi2.pdf(x, 1)), 2 * chi, np.inf)[0]
    print(chi * 2, " ", P, end='\n')


def second_chi_square(prob_array, state_array, len_all_astate):
    chi = 0
    for i in range(0, len(prob_array)):
        chi += ((state_array[i] - (len_all_astate * prob_array[i])) ** 2) / (len_all_astate * prob_array[i])
    P = integrate.quad((lambda x: scipy.stats.chi2.pdf(x, 5)), chi, np.inf)[0]
    print(chi, " ", P, end='\n')


def main():
    reader_csv()
    count_of_state()
    binomial_dist(0.5, state_3, len(social_state))
    z_crit(p0, p, len(social_state))
    chi_square(0.5, state_3, len(social_state))
    all_probs = [0.1, 0.2, 0.5, 0.1, 0.1]
    all_states = [state_1, state_2, state_3, state_4, state_5]
    second_chi_square(all_probs, all_states, len(social_state))


if __name__ == '__main__':
    main()
