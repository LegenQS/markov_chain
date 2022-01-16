# markov_chain
This file ranks 769 college football teams based on the scores of every game in the 2019 season by markov chains, and updates transition matrix M with the following rule:

M[A, A] = M[A, A] + l{A wins} + point_A / (point_A + point_B)

M[B, A] = M[B, A] + l{A wins} + point_A / (point_A + point_B)

M[B, B] = M[B, B] + l{B wins} + point_B / (point_A + point_B)

After that, the file achieves the following two functions:
1. List top "rank" (user defined number) teams with matrix M by calculating w_t with given iteration time t, ranking by decreasing value of w_t;
2. Calculate the stationary distribution by calculate first eigenvector of matrix M and compare the difference between each state under t and the stationary distribution w_infty 
by first-order norm.
