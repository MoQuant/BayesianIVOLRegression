import numpy as np
import pandas as pd
import json 
import time
import datetime
import warnings
import matplotlib.pyplot as plt

# Filter out runtime warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

class Tree:

    def __init__(self, S, K, r, q, t, steps=50, optype='call'):
        self.S = S
        self.K = K
        self.r = r
        self.q = q
        self.t = t
        self.N = steps
        self.dt = t/steps
        self.optype = optype
        self.row = 4*steps + 2
        self.col = steps + 1
        self.center = int(self.row / 2 - 1)
        self.tree = [[0 for j in range(self.col)] for i in range(self.row)]

    def params(self, v):
        self.up = np.exp(v*np.sqrt(2.0*self.dt))
        self.down = 1.0/self.up
        self.m = 1.0
        
        A = np.exp((self.r - self.q)*self.dt/2.0)
        B = np.exp(-v*np.sqrt(self.dt/2.0))
        C = np.exp(v*np.sqrt(self.dt/2.0))

        self.pu = pow((A - B)/(C - B), 2)
        self.pd = pow((C - A)/(C - B), 2)
        self.pm = 1.0 - (self.pu + self.pd)

    def optionTree(self):
        self.tree[self.center][0] = self.S
        for j in range(self.col):
            for i in range(1, self.col - j):
                self.tree[self.center - 2*i][i + j] = self.tree[self.center - 2*(i-1)][i-1+j]*self.up
                self.tree[self.center + 2*i][i + j] = self.tree[self.center + 2*(i-1)][i-1+j]*self.down
                self.tree[self.center][i +j] = self.tree[self.center][i - 1 + j]*self.m    

        for i in range(self.row):
            if i % 2 != 0:
                if self.optype == 'call':
                    self.tree[i][-1] = np.max([self.tree[i - 1][-1] - self.K, 0.0])
                else:
                    self.tree[i][-1] = np.max([self.K - self.tree[i - 1][-1], 0.0])

        inc = 2
        for j in range(2, self.col+1):
            for i in range(inc, self.row - inc):
                if i % 2 != 0:
                    A = self.tree[i - 2][-j+1]
                    B = self.tree[i][-j+1]
                    C = self.tree[i + 2][-j+1]
                    cash = self.pu*A + self.pm*B + self.pd*C
                    cash = np.exp(-self.r*self.dt)*cash
                    if np.isnan(cash):
                        return 0
                    if self.optype == 'call':
                        self.tree[i][-j] = np.max([self.tree[i - 1][-j] - self.K, cash])
                    else:
                        self.tree[i][-j] = np.max([self.K - self.tree[i - 1][-j], cash])
            inc += 2
        
        return self.tree[self.center + 1][0]

    def optionVega(self, v):
        dV = 0.01
        self.params(v+dV)
        c1 = self.optionTree()
        self.params(v-dV)
        c0 = self.optionTree()
        vega = (c1 - c0)/(2.0*dV)
        return vega

    def impliedVol(self, option_price):
        v0 = 0.1
        v1 = 0.2
        for i in range(12):
            self.params(v0)
            try:
                v1 = v0 - 0.01*(self.optionTree() - option_price)/self.optionVega(v0)
            except:
                return False
            if abs(v1 - v0) < 0.0001:
                break
            v0 = v1
        if v1 < 0:
            return False
        return v1

# --------------------------------

def QuantVol(f):
    def Solve(*a, **b):
        K, oP, IV, expire = f(*a, **b)
        ivol = {e:0 for e in expire}
        for ep in expire:
            total = sum(oP[ep])
            for price, iv in zip(oP[ep], IV[ep]):
                ivol[ep] += (price/total)*iv
        return ivol, expire
    return Solve

@QuantVol
def Options(S, rf, q):
    chain = open('SPYChain.json','r').read()
    chain = json.loads(chain)['options']
    expiration = list(chain.keys())
    expire = [expiration[i] for i in (2, 3, 5, 8, 13, 21)]
    strikes = {e:[] for e in expire}
    prices = {e:[] for e in expire}
    impvol = {e:[] for e in expire}
    for expiry in expire:
        T1 = time.mktime(datetime.datetime.strptime(expiry, '%Y-%m-%d').timetuple())
        T0 = int(time.time())
        T = (T1 - T0)/(60*60*24*365)
        stk = list(chain[expiry]['c'].keys())
        price = [float(chain[expiry]['c'][sk]['l']) for sk in stk]
        strike = list(map(float, stk))
        for K, oP in zip(strike, price):
            if K >= S*0.98 and K <= S*1.02:
                tree = Tree(S, K, rf, q, T)
                iv = tree.impliedVol(oP)
                if iv:
                    print(T, oP, iv)
                    strikes[expiry].append(K)
                    prices[expiry].append(oP)
                    impvol[expiry].append(iv)
    return strikes, prices, impvol, expire


def Regression(x, y, beta):
    x0, x1 = np.min(x), np.max(x)
    nn = 50
    dx = (x1 - x0)/(nn - 1)
    rx, ry = [], []
    for i in range(nn):
        xi = x0 + i*dx
        rx.append(xi)
        ry.append(beta[0] + beta[1]*xi)
    return rx, ry


# --------------------------------

days = 150

sp = pd.read_csv('SPY.csv')['adjClose'].values[-days:]
ts = pd.read_csv('TSLA.csv')['adjClose'].values[-days:]

x = sp[1:]/sp[:-1] - 1.0
y = ts[1:]/ts[:-1] - 1.0

S = sp[-1]
rf = 0.0355
q = 0.0122

iv, expiry = Options(S, rf, q)

X = np.array([[1, i] for i in x])
Y = y

XTX = X.T.dot(X)
XTY = X.T.dot(Y)

beta = np.linalg.inv(XTX).dot(XTY)
rx, ry = Regression(x, y, beta)

lb = 0.03

E0 = lb*np.linalg.inv(XTX)
W0 = 0.001*np.ones(2)

E0W0 = E0.dot(W0)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(x, y, color='gray')
ax.plot(rx, ry, color='black')

colors = ['red','orange','gold','green','blue','purple']

for i, option in enumerate(expiry):
    ivol = iv[option]
    factor = 1.0 / pow(ivol, 2)
    En = np.linalg.inv(E0 + factor*XTX)
    sbeta = En.dot(E0W0 + factor*XTY)
    
    sx, sy = Regression(x, y, sbeta)
    ax.plot(sx, sy, color=colors[i], label='Expire: ' + option)

ax.legend()
plt.show()
    
    








