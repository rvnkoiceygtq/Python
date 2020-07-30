# Load in our libraries
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt


cwd = os.getcwd()
print(cwd)


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.describe()

print(pd.crosstab(train.Sex, train.Embarked, normalize=True ,margins=True, margins_name="Total"))
print(train.groupby(['Pclass','Sex'])['Survived'].agg(['mean','std','count','median','sum']))

train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

print(train.head(5))
print(train['Embarked'].head(3))
print(train.Sex.head(5))
print(train.Sex.dtypes)


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)

g = sns.FacetGrid(train, col='Sex')
g.map(plt.hist, 'Age', bins=20)


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train, col='Sex', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

train.head(5)

import seaborn as sns
sns.set(style="ticks")
df=train[['Age','Fare','Pclass','Sex']]
df.head(5)
sns.pairplot(df, hue="Sex")


sns.violinplot(x='Embarked',y='Age',hue='Sex',data=train, palette="muted", split=True, inner="quart")
#sns.swarmplot(x='Embarked',y='Age',hue='Sex',data=train,color='w',alpha=0.2)
plt.title("Violin and Swarm Plot to compare fare distribution among Pclass groups")


train['FirstName'] = train['Name'].str.extract('(Mr\. |Miss\. |Master. |Mrs\.[A-Za-z ]*\()([A-Za-z]*)', expand=False)[1]
print(train['FirstName'].head(5))

train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print(train['Title'].head(5))

train['LastName'] = train['Name'].str.extract('([A-Za-z]+),', expand=False)

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
print(train['Has_Cabin'])

print(train.head(6))



train['Pclass_12'] = train["Pclass"].apply(lambda x: 1 if x in (1,2) else 0)

##if any(x in str for x in a):

##找出已婚太太
##Mrs(已婚太太-英) ##Mme(已婚太太-法) ##Lady(貴婦) ##the Countess(伯爵夫人)
##Ms(女士,婚姻不明或不願提及婚姻狀況)
train['Name_Mrs_v2'] = train["Name"].apply(lambda x: 1 if x in ("Mrs","Mme","Ms","Countess","Lady") else 0)
print(train.head(6))

train['Mrs'] = train["Name"].apply(lambda x: 1 if any(y in x for y in ("Mrs","Mme","Ms","Countess","Lady")) else 0)

##Miss(小姐-英) ##Mlle(小姐-法)
train['Miss'] = train["Name"].apply(lambda x: 1 if any(y in x for y in ("Miss","Mlle")) else 0)
##train['Pclass_1'] = train["Pclass"].apply(lambda x: 1 if x == 1 else 0)
print(train.head(6))

##Master(小男孩)
train['Master'] = train["Name"].apply(lambda x: 1 if any(y in x for y in ("Master")) else 0)

#Col(陸軍上校Colonel) #Major(少校) #Capt(上尉Captain)
train['Military_officer'] = train["Name"].apply(lambda x: 1 if any(y in x for y in ("Capt","Col","Major")) else 0) 

##Dr(醫生,博士)
train['Dr'] = train["Name"].apply(lambda x: 1 if any(y in x for y in ("Dr")) else 0)

##貴族['Nobleman'] #Sir(爵士) ##Lady(貴婦) ##the Countess(伯爵夫人)
#上流人士
train['Nobleman'] = train["Name"].apply(lambda x: 1 if any(y in x for y in ("Sir","Countess","Lady")) else 0)

##Rev(牧師Reverend)
train['Rev'] = train["Name"].apply(lambda x: 1 if any(y in x for y in ("Rev")) else 0)

##Mr(先生男士)
train['Mr'] = train["Name"].apply(lambda x: 1 if any(y in x for y in ("Mr")) else 0)


#train['Royal'] = train["Name"].apply(lambda x: 1 if any(y in x for y in ("Sir","Capt","Col","Major")) else 0)
#train['Name_Prestige'] = train["Name"].apply(lambda x: 1 if any(y in x for y in ("Sir","Capt","Col","Major")) else 0)



#.search(' ([A-Za-z]+)\.', name)




'''

Embarked_df=pd.get_dummies(train['Embarked'],prefix='Embarked')
Sex_df=pd.get_dummies(train.Sex,prefix='Sex')

train_new = pd.concat([train,Embarked_df,Sex_df],axis=1)

#train_new = pd.concat([train, pd.get_dummies(train['Embarked'],prefix='Embarked')], axis=1)

print(Embarked_df.head(5))
print(Sex_df.head(5))
print(train_new.head(6))

#pd.info()



print(pd.crosstab(train["Sex"],train["Embarked"],margins=True))

print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

print(pd.crosstab(train.Sex, train.Embarked, normalize=True ,margins=True, margins_name="Total"))
print(pd.crosstab(train.Sex, train.Embarked, normalize='columns',margins=True, margins_name="Total"))
print(pd.crosstab(train.Sex, train.Embarked, normalize='index',margins=True, margins_name="Total"))

def percConvert(ser):
  return ser/float(ser[-1])
 
print(pd.crosstab(train["Sex"],train["Embarked"],margins=True).apply(percConvert, axis=1))

print(pd.crosstab(train["Sex"],train["Embarked"],margins=True).apply(percConvert, axis=0))

#print(train["Sex"].mean())


sns.violinplot(x='Sex',y='Fare',data=train,inner=None)
#sns.swarmplot(x='Sex',y='Fare',data=train,color='w',alpha=0.5)
plt.title("Violin and Swarm Plot to compare fare distribution among Pclass groups")


# Creaet violin and swarm plots
sns.violinplot(x='Embarked',y='Age',hue='Sex',data=train,inner=None)
sns.swarmplot(x='Embarked',y='Age',hue='Sex',data=train,color='w',alpha=0.5)
plt.title("Violin and Swarm Plot to compare fare distribution among Pclass groups")
# Note: This takes a long time to run


# Group fare into bins to analyze survival rate across brackets. The brackets are informed by the dist plot above
bins = [0,20,40,60,80,100,200,400,800]
train['Fare_Groups'] = pd.cut(train['Fare'],bins)
train['Fare_Groups2'] = train['Fare_Groups'].astype("object") # Need this conversion for heatmap to work
sns.heatmap(pd.crosstab(train['Pclass'],train['Fare_Groups2'],values=train['Survived'],aggfunc=np.mean).T,annot=True,cmap="Blues")
plt.title("Crosstab Heatmap of Pclass x Fares")
'''
'''

# Analyzing cross-tab of age and sex on survival
bins = [0,12,18,30,40,50,60,70,100]  # General age group breakdown
train['Age_Groups'] = pd.cut(train['Age'],bins)
train['Age_Groups2'] = train['Age_Groups'].astype("object") # Need this conversion for heatmap to work
sns.heatmap(pd.crosstab(train['Sex'],train['Age_Groups2'],values=train['Survived'],aggfunc=np.mean).T,annot=True,cmap="Blues")
plt.title("Crosstab Heatmap of Sex x Age: Children (<12yo) seems prioritized, but elderly were not")

'''

'''
full_data = [train, test]

train_info=train.info()
print (train.info())
train_describe=train.describe()

train[['Pclass','Survived']].head()

train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
train[['Pclass','Sex', 'Survived']].groupby(['Pclass','Sex'], as_index=False).mean()

train[['Pclass','Sex', 'Survived']].groupby(['Pclass','Sex'], as_index=False).mean().sort_values(by='Survived',ascending=False)

train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=True)
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Sex', ascending=True)

train.describe(include=['O'])


for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())

#PassengerId = test['PassengerId']

#train.head()

train.groupby('Pclass').count()
train.groupby('Pclass')['Sex'].count()
train.groupby('Pclass')['Sex'].value_counts()
train.groupby('Pclass')['Sex'].value_counts().unstack().fillna(0)
train.groupby('Pclass')['Parch'].value_counts().unstack().fillna(0)



train.groupby('Pclass').mean()
train.groupby('Pclass').median()


train.groupby(['Pclass','Sex']).agg('mean')
train.groupby(['Pclass','Sex'])['Survived'].agg('mean')
train.groupby(['Pclass','Sex'])['Survived'].agg(['mean','std','count','median','sum'])
train.groupby(['Pclass','Sex'])['Fare'].agg(['mean','std','count','median','sum'])
train.groupby(['Pclass','Sex'])['Age'].agg(['mean','std','count','median','sum','max','min'])

t_g_P_S_A=train.groupby(['Pclass','Sex'])['Age'].agg(['mean','std','count','median','sum','max','min'])
t_g_P_S_F=train.groupby(['Pclass','Sex'])['Fare'].agg(['mean','std','count','median','sum','max','min'])
t_g_P_S_S=train.groupby(['Pclass','Sex'])['Survived'].agg(['mean','std','count','median','sum','max','min'])

train.groupby(['Pclass','Sex'])['Age'].agg(['mean','std','count','median','sum','max','min'])
train.groupby(['Pclass','Sex'])['Fare'].agg(['mean','std','count','median','sum','max','min'])
train.groupby(['Pclass','Sex'])['Survived'].agg(['mean','std','count','median','sum','max','min'])


grouped=train.groupby(['Pclass','Sex'])
grouped_pct=grouped['Survived'] #tip_pct列
grouped_pct.agg('mean')#对与9-1图标中描述的统计，可以将函数名直接以字符串传入
grouped_pct.agg(['mean','count'])#对与9-1图标中描述的统计，可以将函数名直接以字符串传入



for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 6)
print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())

train['CategoricalAge'] = pd.qcut(train['Age'], 5)
print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).agg(['mean','count']))
#print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())

train['CategoricalAge'] = pd.cut(train['Age'], 5)
print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).agg(['mean','count']))
#print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())

train['CategoricalAge'] = pd.qcut(train['Age'], 2)
print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).agg(['mean','count']))

'''
