# Table-of-Contents
Machine Learning
#整个代码在阿里云推出的AI大模型开源社区--魔塔社区运行
#代码是在：https://www.bilibili.com/video/BV11F411u7Ck/?share_source=copy_web&vd_source=4c824fb53899b83ce1207a3200de7b9d的基础上进行优化修改
# 环境设置
import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 数据加载
train_df = pd.read_csv('./titanic/train.csv')
test_df = pd.read_csv('./titanic/test.csv')
combine = [train_df, test_df]  # 组合数据集方便统一处理

# ==== 数据探索与预处理 ====
def data_exploration():
    # 显示数据摘要
    print("训练集形状:", train_df.shape)
    print("\n前5条数据:")
    print(train_df.head())
    
    # 缺失值统计
    print("\n训练集缺失值统计:")
    print(train_df.isnull().sum())
    print("\n测试集缺失值统计:")
    print(test_df.isnull().sum())
    
    # 数值型特征统计
    print("\n数值特征统计:")
    print(train_df.describe().T)
    
    # 分类特征统计
    print("\n分类特征统计:")
    print(train_df.describe(include=['O']).T)

data_exploration()

# ==== 特征工程 ====
# 删除无关特征
for dataset in combine:
    dataset.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

# 提取称呼特征
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# 填充缺失年龄
for dataset in combine:
    dataset['Age'] = dataset.groupby(['Sex', 'Pclass', 'Title'])['Age'].transform(lambda x: x.fillna(x.median()))

# 创建新特征
for dataset in combine:
    # 家庭规模
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    # 是否独自一人
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    # 年龄等级
    dataset['AgeClass'] = pd.cut(dataset['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4])
    # 票价等级
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    dataset['FareClass'] = pd.qcut(dataset['Fare'], 4, labels=[0,1,2,3])

# 转换分类特征
for dataset in combine:
    # 性别转换
    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)
    # 登船港口填充和转换
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
    # 称呼映射
    title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
    dataset['Title'] = dataset['Title'].map(title_mapping).fillna(0)

# 删除冗余特征
for dataset in combine:
    dataset.drop(['Name', 'SibSp', 'Parch', 'Age', 'Fare'], axis=1, inplace=True)

# ==== 数据准备 ====
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.copy()

# 独热编码
categorical_features = ['Pclass', 'Embarked', 'Title']
X_train = pd.get_dummies(X_train, columns=categorical_features)
X_test = pd.get_dummies(X_test, columns=categorical_features)

# 对齐列
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==== 模型训练与评估 ====
models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('SVM', SVC(probability=True)),
    ('KNN', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Gradient Boosting', GradientBoostingClassifier())
]

# 交叉验证评估
results = []
for name, model in models:
    scores = cross_val_score(model, X_train_scaled, Y_train, cv=5, scoring='accuracy')
    results.append((name, scores.mean(), scores.std()))
    print(f"{name:20} | 平均准确率: {scores.mean():.4f} | 标准差: {scores.std():.4f}")

# ==== 超参数调优 ====
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, Y_train)

print("\n最佳参数组合:", grid_search.best_params_)
print("最佳准确率:", grid_search.best_score_)

# ==== 生成预测 ====
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, Y_train)
predictions = best_model.predict(X_test_scaled)

# 保存结果
output = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})
output.to_csv('submission.csv', index=False)
print("\n预测结果已保存为 submission.csv")
