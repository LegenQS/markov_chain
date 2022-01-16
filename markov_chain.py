import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

# minimum value used to calculate the first eigenvalue
minValue = 10 ** (-13)

class Markov:
    def __init__(self, file_name):
        # class object initialization
        # paramter list:
        # --------------------------------------------
        # file_name: local path of file to be read
        # Object attributes:
        # --------------------------------------------
        # refer: dict type paramter to judge the dtype of reading file with its sufix
        # name: data containing names of college soccer team
        # length: total number of teams on name_file
        # result: total score after calculation on data file
        # w_0: initialized with uniform distribution
        # data: data file containing data of competition records with format as:
        # teamA    pointA    teamB    pointB

        self.refer = {'csv': float, 'txt': str}
        self.data = self.read_file(file_name[0])
        self.name = self.read_file(file_name[1])
        self.length = len(self.name)
        self.result = self.random_walk()
        self.w_0 = [1 / self.length for i in range(self.length)]

    def read_file(self, file_name):
        # read file from local path with specific file type, with "txt" to str and "csv" to float
        # parameter list:
        # --------------------------------------------
        # file_name: local path of file to be read

        file_type = file_name.split('.')[-1]
        return np.loadtxt(open(file_name, "r"), delimiter=",", skiprows=0, dtype=self.refer[file_type])

    def random_walk(self):
        # calculate final scores with given data

        M = np.zeros((self.length, self.length))

        for record in self.data:
            teamA, pointA, teamB, pointB = [int(i) for i in record]
            teamA -= 1
            teamB -= 1

            M[teamA, teamA] += pointA / (pointA + pointB)
            M[teamB, teamA] += pointA / (pointA + pointB)
            M[teamA, teamB] += pointB / (pointA + pointB)
            M[teamB, teamB] += pointB / (pointA + pointB)

            if pointA > pointB:
                M[teamA, teamA] += 1
                M[teamB, teamA] += 1
            elif pointA < pointB:
                M[teamA, teamB] += 1
                M[teamB, teamB] += 1

        return self.normalize(M)

    def normalize(self, M):
        # normalize input matrix, enables each row of matrix X to sum to 1
        # parameter list:
        # --------------------------------------------
        # M: input matrix

        for i in range(M.shape[0]):
            if sum(M[i] != 0):
                M[i] /= sum(M[i])

        return M

    def state_distribution(self, M, w, t, norm_sign=False, w_infty=None, decimals=5):
        # with input transition matrix and initialized position, calculate final position with given iteration time, or
        # calculate the difference between current state and the final state by iteration time
        # parameter list:
        # --------------------------------------------
        # M: transition matrix
        # w: first initialized state
        # t: iteration times
        # norm_sign: calculate difference between w and w_infty when True, and calculate final state when False by default
        # w_infty: stationary distribution by calculation on the first eigenvector of M, None by default
        # decimals: number of decimal places to keep, 5 by default

        norm = []

        if norm_sign:
            for i in range(t):
                w = w @ M
                norm.append(np.linalg.norm(w - w_infty, ord=1))
            return norm
        else:
            for i in range(t):
                w = w @ M

        return np.round(w, decimals)

    def myplot(self, x, title='', xlabel='x', ylabel='y', figname='1.png'):
        # plot given parameter with details
        # parameter list:
        # --------------------------------------------
        # x: data to be plotted
        # title: plot title on fig, '' by default
        # xlabel: plot label of x on fig, 'x' by default
        # ylabel: plot lable of y on fig, 'y' by default
        # figname: fig name to save to loocal path, '1.png' by default

        plt.figure(figsize=(30, 30))
        plt.plot(x, linewidth=3)
        if title:
            plt.title(title, fontsize=50)
        if xlabel:
            plt.xlabel(xlabel, fontsize=50)
        if ylabel:
            plt.ylabel(ylabel, fontsize=50)
        plt.tick_params(labelsize=50)
        plt.savefig(figname, dpi=500)
        plt.show()
        plt.close()

    def rank_top(self, t=[10, 100, 1000, 10000], rank=25):
        # calculate and show top "rank" teams with given iteration time t
        # parameter list:
        # --------------------------------------------
        # t: iteration times by list format
        # rank: number of teams to be shown

        seq = [i for i in range(self.length)]

        f = open("./log.txt", "w+")
        for i in t:
            w_t = self.state_distribution(self.result, self.w_0, i)
            dic = dict(zip(w_t, seq))

            dic = dict(sorted(dic.items(), key=lambda x: (x[0], x[1]), reverse=True))
            final = dict(zip(self.name[list(dic.values())[:rank]], list(dic.keys())[:rank]))
            length = len(max(list(final.keys()), key=len))

            print("-------------------------------", file=f,flush=True)
            print("Table {}".format(t.index(i) + 1), file=f,flush=True)
            print("{:<5} {:<{}} {:<15}".format('Rank', 'Team Name', length + 1, 'w_t value'), file=f,flush=True)
            for i in range(rank):
                print("{:<5} {:<{}} {:<15}".format(i + 1, self.name[list(dic.values())[i]],
                                                   length + 1, list(dic.keys())[i]), file=f,flush=True)
        f.close()

    def weight_iter(self, t=10000):
        # calculate and show the difference between current state under t and the stationary distribution along with t
        # parameter list:
        # --------------------------------------------
        # t: given total iteration times

        c = np.linalg.eig(self.result.T)
        for i in range(len(c[0])):
            if abs(c[0][i] - 1) < minValue:
                u_1 = c[1][:, i]
                break

        w_infty = u_1 / sum(u_1)
        norm = self.state_distribution(self.result, self.w_0, t, norm_sign=True, w_infty=w_infty)

        self.myplot(norm, title='Relationship between t and $||w_t - w_\infty||_1$', xlabel='t',
               ylabel='Value of $||w_t - w_\infty||_1$', figname='1.png')

def main():
    file_name = ['./CFB2019_scores.csv', './TeamNames.txt']
    mark = Markov(file_name)
    mark.rank_top()
    mark.weight_iter()

if __name__ == '__main__':
    main()