

import numpy as np
import pandas as pd


# | SA       |  0.965314 |  0.957999 |  0.886267 |  0.956583 |         1 |    0.953233 |
# | GA       |  0.968617 |  0.957763 |  0.915526 |  0.96319  |         1 |    0.961019 |
# | PS       |  0.830109 |  0.830109 |  0.830345 |  0.830109 |         1 |    0.864134 |
# | AC       |  0.963426 |  0.95446  |  0.893582 |  0.958471 |         1 |    0.953988 |
# | DE       |  0.965314 |  0.955168 |  0.882964 |  0.957291 |         1 |    0.952147 |


ary0 = np.load(open('acc/score.npy', 'rb')).T
ary1 = np.load(open('acc/score_seed1.npy', 'rb')).T
ary2 = np.load(open('acc/score_seed2.npy', 'rb')).T
ary3 = np.load(open('acc/score_seed3.npy', 'rb')).T
ary_stack = np.stack([ary0,ary1,ary2,ary3],axis=0)
print(ary_stack)
ary_stack = np.transpose(ary_stack, (1, 0, 2))
print(ary_stack)

for i_class in range(5):
    ary = ary_stack[i_class,:,:]
    df = pd.DataFrame(ary,columns=['SA','GA','PS','AC','DE'])
    # print(df)
    df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=['SA','GA','PS','AC','DE'])
    df_melt.columns = ['index', 'treatments', 'value']
    # print(df_melt)

    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # perform multiple pairwise comparison (Tukey HSD)

    m_comp = pairwise_tukeyhsd(endog=df_melt['value'], groups=df_melt['treatments'], alpha=0.05)
    print(m_comp)