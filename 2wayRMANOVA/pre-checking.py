import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Step 1: 导入数据
data = pd.read_csv('Q1.csv')

# Step 2: 检查缺失值
print("缺失值检查：")
print(data.isnull().sum())

# Step 3: 描述性统计
print("\n描述性统计：")
print(data.describe())

# Step 4: 正态性检验
shapiro_results = stats.shapiro(data['Score'])
print("\nShapiro-Wilk p-value:", shapiro_results.pvalue)

# # Step 5: 方差齐性检验
# group1 = data['Score'][data['methods'] == 'graph']  # 替换为您的组名
# group2 = data['Score'][data['methods'] == 'AI']  # 替换为您的组名
group1 = data['Score'][data['collective'] == 'Collective']  # 替换为您的组名
group2 = data['Score'][data['collective'] == 'Personal']  # 替换为您的组名
levene_results = stats.levene(group1, group2)
print("Levene p-value:", levene_results.pvalue)


# if not go here to transform: https://runzecai.com/tools/art_tool/

# Step 6: 检查离群值
# 使用 IQR 方法
Q1 = data['Score'].quantile(0.25)
Q3 = data['Score'].quantile(0.75)
IQR = Q3 - Q1

# 定义上下限
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 识别离群值
outliers = data[(data['Score'] < lower_bound) | (data['Score'] > upper_bound)]
print("\n离群值检查：")
print(outliers)

# # Step 7: 可视化检查
# # 箱形图
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='methods', y='Score', data=data)
# plt.title('Boxplot of Response Variable by Method')
# plt.show()

# # QQ 图
# plt.figure(figsize=(8, 8))
# sm.qqplot(data['Score'], line='s')
# plt.title('QQ Plot')
# plt.show()