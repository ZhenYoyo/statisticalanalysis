import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# score1 = []
# score2 = []
# score3 = []
score1 = [6, 4, 5, 6, 4, 6, 7, 5, 7, 5, 4, 6]
score2 = [5, 6, 5, 4, 5, 7, 6, 2, 7, 6, 6, 6]
score3 = [6, 7, 5, 6, 3, 3, 7, 6, 6, 5, 5, 2]
# 数据准备
data = {
    'Participant No': range(1, 13),
    'Q1O': score1,
    'Q1A': score2,
    'Q1T': score3
}

df = pd.DataFrame(data)

# 将数据转换为长格式
long_df = pd.melt(df, id_vars=['Participant No'], value_vars=['Q1O', 'Q1A', 'Q1T'], 
                  var_name='Condition', value_name='Score')

# 进行单因素方差分析
model = ols('Score ~ Condition', data=long_df).fit()
anova_results = anova_lm(model)

# 输出 ANOVA 结果
print(anova_results)

# 进行 Tukey 检验
#tukey post hoc pairwise comparison test
tukey = pairwise_tukeyhsd(endog=long_df['Score'], groups=long_df['Condition'], alpha=0.05)
print(tukey)
