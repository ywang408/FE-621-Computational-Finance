from BS_formula import BS_formula
from impliedVol import DATA1, DATA2

r = 0.07 / 100
for i in range(2):
    for j in range(len(DATA1[i])):
        DATA2[i][j]['vol'] = DATA1[i][j].bisec_Root
        DATA2[i][j]['BS_price'] = DATA2[i][j].apply(lambda x:
                                                    BS_formula(x.type, x.spotPrice,
                                                               x.strike, x.delta_t,
                                                               x.vol, r), axis=1)
print(DATA2[1][7].head())
