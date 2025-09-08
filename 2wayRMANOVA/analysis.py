import pandas as pd
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Step 1: 导入数据
data = pd.read_csv('Q1.csv')


# 进行重复测量方差分析
anova_results = AnovaRM(data, 'Score', 'subID', within=['methods', 'collective']).fit()
print("\nANOVA 结果：")
print(anova_results)

# Step 10: 进行事后比较
long_data = pd.melt(data, id_vars=['subID', 'methods', 'collective'], value_vars=['Score'])
long_data['group'] = long_data['methods'] + "_" + long_data['collective']
tukey_results = pairwise_tukeyhsd(long_data['value'], long_data['group'])
print("\nTukey 事后比较结果：")
print(tukey_results)