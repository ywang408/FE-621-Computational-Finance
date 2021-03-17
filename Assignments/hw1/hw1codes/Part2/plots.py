import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from impliedVol import DATA1
import numpy as np
import pandas as pd

# 2d plot
temp1 = DATA1[1][5]
temp2 = DATA1[1][7]
temp3 = DATA1[1][10]
temp1 = temp1[temp1.type == 'c']
temp2 = temp2[temp2.type == 'c']
temp3 = temp3[temp3.type == 'c']

exp = [temp1.expiry.iloc[0], temp2.expiry.iloc[0], temp3.expiry.iloc[0]]
strike1 = temp1.strike
strike2 = temp2.strike
strike3 = temp3.strike
vol1 = temp1.bisec_Root
vol2 = temp2.bisec_Root
vol3 = temp3.bisec_Root

plt.figure(1)
plt.plot(strike1, vol1, label=exp[0].date())
plt.plot(strike2, vol2, label=exp[1].date())
plt.plot(strike3, vol3, label=exp[2].date())
plt.xlabel('strike')
plt.ylabel('vol')
plt.title("SPY call options volatility")
plt.legend()

# 3d plot
# merge strike and vol of call options
info = ['strike', 'bisec_Root']
delta_T = ['strike', 1]
new_df = DATA1[1][0][info]
for df in DATA1[1]:
    delta_T.append(df['delta_t'].iloc[0]*365)
    call_df = df[df.type == 'c']
    new_df = pd.merge(new_df, call_df[info], on='strike', how='outer')
    new_df.columns = delta_T

# drop useless data
new_df = new_df.set_index('strike')
new_df = new_df.drop(new_df.columns[0:4], axis=1)
new_df = new_df.dropna()

# 3d plot
plt.figure(2)
x = new_df.columns
y = new_df.index
X,Y = np.meshgrid(x,y)
Z = new_df
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Y,X,Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_title('Implied volatility surface')
ax.set_ylabel('time to maturity')
ax.set_xlabel('strike')
ax.set_zlabel('volatility')
plt.show()


