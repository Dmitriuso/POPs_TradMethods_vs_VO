import ctypes
import numpy as np
import pandas as pd
from scipy import optimize
from vitaoptimum.voplus.ccs import Ccs
import os
import pickle

# tolerance
tol = 1.0e-16

def MaximizeSharpeRatioOptmzn(MeanReturns, CovarReturns, RiskFreeRate, PortfolioSize):
    # define maximization of Sharpe Ratio using principle of duality
    def f(x, MeanReturns, CovarReturns, RiskFreeRate, PortfolioSize):
        funcDenomr = np.sqrt(np.matmul(np.matmul(x, CovarReturns), x.T))
        funcNumer = np.matmul(np.array(MeanReturns), x.T) - RiskFreeRate
        func = -(funcNumer / funcDenomr)
        return func

    # define equality constraint representing fully invested portfolio
    def constraintEq(x):
        A = np.ones(x.shape)
        b = 1
        constraintVal = np.matmul(A, x.T) - b
        return constraintVal

    # define bounds and other parameters
    xinit = np.repeat(0.33, PortfolioSize)
    cons = ({'type': 'eq', 'fun': constraintEq})
    lb = 0.0
    ub = 1.0
    bnds = tuple([(lb, ub) for x in xinit])

    # invoke minimize solver
    opt = optimize.minimize(f, x0=xinit, args=(MeanReturns, CovarReturns, \
                                               RiskFreeRate, PortfolioSize), method='SLSQP', \
                            bounds=bnds, constraints=cons, tol=tol)
    return opt


# global maximum Sharpe Ratio optimization solver wrapper function
def GlobalMaxSharpeRatioOptmzn(MeanReturns, CovarReturns, RiskFreeRate, PortfolioSize):
    # define maximization of Sharpe Ratio using principle of duality

    def fobj(x, g, h):
        h[0] = sum(x) - 1.0
        funcDenomr = np.sqrt(np.matmul(np.matmul(x, CovarReturns), x.T))
        funcNumer = np.matmul(np.array(MeanReturns), x.T) - RiskFreeRate
        func = -(funcNumer / funcDenomr)
        return func

    # dimension of the problem and constraints
    dim = PortfolioSize  # problem dimension
    # qmeasures = np.zeros(4, dtype=ctypes.c_double)  # std init
    qmeasures = np.array([tol, tol, tol, 0.0], dtype=ctypes.c_double)  # machine precision init
    nh = 1  # equality constraint dimension [ h(x) = 0 ]
    ng = 0  # non equality constraint dimension [ g(x) <= 0 ]

    # set boundary constraints to [0, 1]
    low = np.zeros(dim, dtype=ctypes.c_double) # + [0.001]
    high = np.ones(dim, dtype=ctypes.c_double)

    solver = Ccs(fobj=fobj,  # objective function with constraints
                 dim=dim, ng=ng, nh=nh,  # dimensions
                 low=low, high=high, qmeasures=qmeasures)  # boundary constraints

    results = solver.run()

    return results


# function computes asset returns
def StockReturnsComputing(StockPrice, Rows, Columns):
    StockReturn = np.zeros([Rows - 1, Columns])
    for j in range(Columns):  # j: Assets
        for i in range(Rows - 1):  # i: Daily Prices
            StockReturn[i, j] = ((StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]) * 100
    return StockReturn


def totalcontribrisk(covReturns, AssetWeights, PortfolioRisk):
    mcr = np.matmul(covReturns, np.transpose(AssetWeights)) / PortfolioRisk
    tcr = np.around(((np.multiply(mcr, AssetWeights) / PortfolioRisk)), decimals=3)
    percenttcr = np.asfarray(tcr, float) * 100
    return (percenttcr)


if __name__ == '__main__':

    # set precision for printing results
    np.set_printoptions(precision=16, suppress=True)

    # input portfolio dataset comprising 15 stocks
    StockFileName = 'DJIA_Apr112014_Apr112019_kpf1.csv'
    Rows = 1259  # excluding header
    Columns = 15  # excluding date

    # read stock prices
    df = pd.read_csv(StockFileName, nrows=Rows)

    # extract asset labels
    assetLabels = df.columns[1:Columns + 1].tolist()
    print('Asset labels of portfolio: \n', assetLabels)

    # read asset prices data
    StockData = df.iloc[0:, 1:]

    # compute asset returns
    arStockPrices = np.asarray(StockData)
    [Rows, Cols] = arStockPrices.shape
    arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

    # compute mean returns and variance covariance matrix of returns
    meanReturns = np.mean(arReturns, axis=0)
    covReturns = np.cov(arReturns, rowvar=False)
    print('\nMean Returns:\n', meanReturns)
    print('\nVariance-Covariance Matrix of Returns:\n', covReturns)

    # obtain maximal Sharpe Ratio for  the portfolio  of Dow stocks

    # set portfolio size
    portfolioSize = Columns

    # set risk free asset rate of return
    Rf = 3  # April 2019 average risk  free rate of return in USA approx 3%
    annRiskFreeRate = Rf / 100

    # compute daily risk free rate in percentage
    r0 = (np.power((1 + annRiskFreeRate), (1.0 / 360.0)) - 1.0) * 100
    print('\nRisk free rate (daily %): ', end="")
    print("{0:.3f}".format(r0))

    # initialization
    xOptimal = []
    minRiskPoint = []
    expPortfolioReturnPoint = []
    maxSharpeRatio = 0

    # saving results
    sharpe_list = []
    risk_list = []
    return_list = []
    diver_list = []
    weights_list = []

    def saving_results(sharp, risk, returns, diver, sol):
        sharpe_list.append(sharp)
        risk_list.append(risk)
        return_list.append(returns)
        diver_list.append(diver)
        weights_list.append(sol)

    def print_results():
        print('\n\nRESULTS =================')
        n = len(sharpe_list)
        for i in range(n):
            if i == 0:
                print('\n >> TRADITIONAL:')
            elif i == 1:
                print('\n >> ADVANCED:')
            print(' sharpe = {}'.format(sharpe_list[i]))
            print(' risk =   {}'.format(risk_list[i]))
            print(' return = {}'.format(return_list[i]))
            print(' divers = {}'.format(diver_list[i]))
            print(' weights= {}\n'.format(weights_list[i]))
        # save to file
        to_save = (sharpe_list, risk_list, return_list, diver_list, weights_list)
        pickle.dump(to_save, open('saved.bin', 'wb'))

    def load_results(file_name: str = 'saved.bin'):
        sharpe, risk, returns, divers, weights = pickle.load(open(file_name, 'rb'))
        return sharpe, risk, returns, divers, weights
        # USE: sharp, risk, returns, divers, weights = load_results('saved_std.bin')

    print('\nTRADITIONAL ALGORITHM\n')

    # compute maximal Sharpe Ratio and optimal weights
    result = MaximizeSharpeRatioOptmzn(meanReturns, covReturns, r0, portfolioSize)
    print(result)
    xOptimal.append(result.x)

    # compute risk returns and max Sharpe Ratio of the optimal portfolio
    xOptimalArray = np.array(xOptimal)
    Risk = np.matmul((np.matmul(xOptimalArray, covReturns)), np.transpose(xOptimalArray))
    expReturn = np.matmul(np.array(meanReturns), xOptimalArray.T)
    annRisk = np.sqrt(Risk * 251)
    annRet = 251 * np.array(expReturn)
    maxSharpeRatio = (annRet - Rf) / annRisk

    # display results
    print('Maximal Sharpe Ratio: ', maxSharpeRatio, '\nAnnualized Risk (%):  ', \
          annRisk, '\nAnnualized Expected Portfolio Return(%):  ', annRet)
    print('\nOptimal weights (%):\n', xOptimalArray.T * 100)

    # compute diversification ratio of the optimal portfolio
    AssetWeights = xOptimalArray
    PortfolioRisk = np.sqrt(np.matmul((np.matmul(AssetWeights, \
                                                 covReturns)), np.transpose(AssetWeights)))
    AssetRisk = np.sqrt(np.diagonal(covReturns))
    PortfolioDivRatio = sum(np.multiply(AssetRisk, AssetWeights.squeeze())) / PortfolioRisk.squeeze()
    print("\n Diversification Ratio %4.2f" % PortfolioDivRatio)

    saving_results(maxSharpeRatio.squeeze().squeeze(),
                   annRisk.squeeze().squeeze(), annRet.squeeze(), PortfolioDivRatio, xOptimal)

    # ==================================================================================

    print('\nVITAOPTIMUM PLUS ALGORITHM\n')

    os.environ['OMP_NUM_THREADS'] = '1'

    restarts = 100

    for i in range(restarts):
        print('\n  >> restart {}'.format(i + 1))

        # compute optimal weights and the globally maximal Sharpe Ratio
        globlOptimumresult = GlobalMaxSharpeRatioOptmzn(meanReturns, covReturns, r0, portfolioSize)
        globlOptimumresult.print()
        xOptimal = globlOptimumresult.solution
        fitness = globlOptimumresult.best_fobj
        print("optimum: ", xOptimal)

        # compute risk returns and max Sharpe Ratio of the VitaOptimum Plus optimal portfolio
        xOptimalArray = np.array(xOptimal)
        Risk = np.matmul((np.matmul(xOptimalArray, covReturns)), np.transpose(xOptimalArray))
        expReturn = np.matmul(np.array(meanReturns), xOptimalArray.T)
        annRisk = np.sqrt(Risk * 251)
        annRet = 251 * np.array(expReturn)
        maxSharpeRatio = (annRet - Rf) / annRisk

        # display results
        print('\n Maximal Sharpe Ratio: %4.2f' % maxSharpeRatio)
        print('\n Annualized Risk %5.2f' % (annRisk), '%')
        print('\n Annualized Expected Portfolio Return %5.2f' % (annRet), '%')
        print('\n Optimal weights (%):\n', xOptimalArray.T * 100)

        # compute diversification ratio of the optimal portfolio
        AssetWeights = xOptimalArray
        PortfolioRisk = np.sqrt(np.matmul((np.matmul(AssetWeights, \
                                                     covReturns)), np.transpose(AssetWeights)))

        AssetRisk = np.sqrt(np.diagonal(covReturns))
        PortfolioDivRatio = sum(np.multiply(AssetRisk, AssetWeights)) / PortfolioRisk
        print("\n Diversification Ratio %4.2f" % PortfolioDivRatio)

        saving_results(maxSharpeRatio, annRisk, annRet, PortfolioDivRatio, xOptimal)

    print_results()

