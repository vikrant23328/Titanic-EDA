#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # <centre> Learning EDA from Historic Diaster "The Titanic Wreck" <centre/>
# 

# <img src="https://raw.githubusercontent.com/insaid2018/Tableau/master/Data/titanic11.jpg"/>

# ## Table of Contents
# 
# 1. [Objective](#section1)<br>
# 2. [Importing Packages and Collecting Data](#section2)
# 3. [Data Profiling & Preprocessing](#section3)
#     - 3.1 [Pre Profiling](#section301)<br/>
#     - 3.2 [Preprocessing](#section302)<br/>
#     - 3.3 [Post Profiling](#section303)<br/>
# 4. [Analysis Through Data Visualization](#section4)
#     - 4.1 [What is Total Count of Survivals and Victims?](#section401)<br/>
#     - 4.2 [Which Gender has more Survival rate?](#section402)<br/>
#     - 4.3 [What is Survival rate based on Person type(Male,female,Child)](#section403)<br/>
#     - 4.4 [Did Economy Class had an impact on Survival?](#section404)<br/>
#     - 4.5 [What is the Survival probaility based on Embarkment of Passengers?](#section405)<br/>
#     - 4.6 [How is Fare distributed for the Passengers?](#section406)<br/>
#     - 4.7 [What was Average Fare by Pclass & Embark location?](#section407)<br/>
#     - 4.8 [ Segment Age in bins with size of 20.Also Correlate Age with Survival.](#section408)<br/>
#     - 4.9 [ Did Solo Traveller has less chances of Survival?](#section409)<br/>
#     - 4.10 [How did Total family size affected Survival Count?](#section410)<br/>
#     - 4.11 [How can you correlate Pclass/Age/fare with Survival rate?](#section411)<br/>
#     - 4.12 [Which features had most Impact on Survival rate? ](#section412)<br/>
# 5. [Conclusions](#section5)<br/>  

# # Objective

# 
# 
# The objective here is to conduct Exploratory data analysis **(EDA)** on the Titanic Dataset in order to gather insights and evenutally predicting survior on basics of factors like Class ,Sex , Age , Gender ,Pclass etc.
# 
# **Why EDA?**
#    - An approach to summarize, visualize, and become intimately familiar with the important characteristics of a data set.
#    - Defines and Refines the selection of feature variables that will be used for machine learning.
#    - Helps to find hidden Insights
#    - It provides the context needed to develop an appropriate model with minimum errors
# 
# 
# **About Event**
# 
# The RMS Titanic was a British passenger liner that **sank** in the **North Atlantic Ocean** in the early morning hours of **15 April 1912**, after it collided with an iceberg during its maiden voyage from **Southampton** to **New York City**. There were an estimated **2,224** passengers and crew aboard the ship, and more than **1,500** died, making it one of the deadliest commercial peacetime maritime disasters in modern history.
# This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# 
# 

# <img src="https://raw.githubusercontent.com/insaid2018/Tableau/master/Data/titanic-maiden-voyage-route.gif" width="400" height="460"/>

# ### 2. Data  Description

# The dataset consists of the information about people boarding the famous RMS Titanic. Various variables present in the dataset includes data of age, sex, fare, ticket etc.
# The dataset comprises of 891 observations of 12 columns. Below is a table showing names of all the columns and their description.

# 
# | Column Name   | Description                                               |
# | ------------- |:-------------                                            :| 
# | PassengerId   | Passenger Identity                                        | 
# | Survived      | Survival (0 = No; 1 = Yes)                                |  
# | Pclass        | Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)               | 
# | Name          | Name of passenger                                         |   
# | Sex           | Sex of passenger                                          |
# | Age           | Age of passenger                                          |
# | SibSp         | Number of sibling and/or spouse travelling with passenger |
# | Parch         | Number of parent and/or children travelling with passenger|
# | Ticket        | Ticket number                                             |
# | Fare          | Price of ticket                                           |
# | Cabin         | Cabin number                                              |
# |Embarkment     | Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)|

# In[52]:


pip install pandas_profiling


# In[155]:


import numpy as np               # For linear algebra
import pandas as pd              # For data manipulation
import matplotlib.pyplot as plt  # For 2D visualization
import pandas_profiling ##comment it....
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# **Importing Data**

# titanic_train=pd.read_csv("https://raw.githubusercontent.com/mohittomar2008/Titanic-Project/main/titanic_train.csv")
# titanic_test=pd.read_csv("https://raw.githubusercontent.com/mohittomar2008/Titanic-Project/main/test.csv")
# 
# combine = [titanic_train, titanic_test]

# In[156]:


titanic_data = pd.read_csv("C:/Users/ajha2/Downloads/titanic.csv")


# In[157]:


titanic_data.columns


# In[160]:


titanic_data.head(15) ## 1st 5 records or you can specify the number


# In[161]:


titanic_data.tail(10) ## last 10 data 


# # **Examining Data**

# In[162]:


titanic_data.shape #shows total number of rows and columns in data set


# In[163]:


titanic_data.describe()


# In[164]:


titanic_data.info()


# In[165]:


titanic_data.isnull().sum() ## to test the number of data points in particular column for which data is unavailable


# In[60]:


titanic_data['Age'].isnull().sum()


# In[61]:


titanic_data['Age'].unique()


# In[68]:


titanic_data['Embarked'].unique()


# In[63]:


titanic_data['Pclass'].unique()


# In[69]:


titanic_data['Cabin'].value_counts().sort_values(ascending = False)


# In[70]:


titanic_data['Age'].value_counts().sort_values(ascending = False) ## count for each distinct values under that column


# In[71]:


titanic_data['Embarked'].value_counts().sort_values(ascending = False)


# In[72]:


titanic_data.groupby('Sex')['Survived'].value_counts()


# In[73]:


titanic_data.groupby('Age')['Survived'].value_counts().sort_values(ascending = False)


# In[74]:


titanic_data.groupby('Pclass')['Survived'].value_counts().sort_values(ascending = False)


# 
# **Insights**:
# 
# 1.Class 3 has the highest death ratio which is 41% and Class 1 has the lowest Death Ratio 9%
# 
# 2.Class 1 has the highest survival rate which is 15% and Class 2 has the lowest survival rate 9%
# 
# 2.Survived is a categorical feature with 0 or 1 values
# 
# 3.Around **38%** samples survived representative of the actual survival rate at **32%**
# 
# 4.Fares varied significantly with few passengers (<1%) paying as high as $512.
# 
# 5.Few elderly passengers (<1%) within age range **65-80**.
# 

# # Data Profiling

# By pandas profiling, an interactive **HTML report** gets generated which contains all the information about the columns of the dataset, like the counts and type of each column. 
# 
# 1.Detailed information about each column, coorelation between different columns and a sample of dataset
# 
# 2.It gives us visual interpretation of each column in the data
# 
# 3.Spread of the data can be better understood by the distribution plot
# 
# 4.Grannular level analysis of each column.

# In[22]:


import pandas_profiling as pp


# In[23]:


profile = pp.ProfileReport(titanic_data)

profile.to_file(output_file="Titanic_before_preprocessing.html")


# In[24]:


import os


# In[25]:


os.getcwd()


# # Treating Outliers

# In[26]:


Q1 = titanic_data['Fare'].quantile(0.25)
Q3 = titanic_data['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - (3 * IQR)
upper_limit = Q3 + (3 * IQR)


# In[27]:


Q1,Q3,IQR,lower_limit,upper_limit


# In[28]:


titanic_data_with_outlier = titanic_data[(titanic_data['Fare'] <= lower_limit) | (titanic_data['Fare'] >= upper_limit)]


# In[29]:


titanic_data_with_outlier.shape


# In[30]:


titanic_data_with_outlier


# In[75]:


sns.set(style='darkgrid')
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True,gridspec_kw={'height_ratios': (.15, .85)})
sns.boxplot(data=titanic_data, x='Fare', ax=ax_box)
sns.histplot(data=titanic_data, x= 'Fare', ax=ax_hist, kde=True)
ax_box.set(xlabel='')
plt.show()


# # Data Preprocessing

# 1. Check for Errors and Null Values
# 
# 2. Replace Null Values with appropriate values
# 
# 3. Drop down features that are incomplete and are not too relevant for analysis
# 
# 4. Create new features that can would help to improve prediction 

# **Check for null or empty values in Data**

# In[76]:


miss_value = titanic_data.isnull().sum()
print(miss_value)
data_len = len(titanic_data)
print(data_len)


# In[77]:


miss_perc = (miss_value/data_len)*100

miss_data = pd.concat([miss_value,miss_perc],axis=1,keys=['Total','%'])

print(miss_data)


# The Age, Cabin and Embarked have null values.Lets fix them

# **Filling missing age by median**

# In[78]:


new_age = titanic_data.Age.median()  
print(new_age)


# In[79]:


new_age = titanic_data.Age.median()  
titanic_data.Age.fillna(new_age, inplace = True)


# In[80]:


titanic_data.isnull().sum()


# **Filling missing Embarked by mode**

# In[81]:


new_embarked = titanic_data['Embarked'].mode()[0]
print(new_embarked)


# In[82]:


titanic_data.Embarked.fillna(new_embarked, inplace = True)


# In[83]:


titanic_data.isnull().sum()


# **Cabin feature may be dropped as it is highly incomplete or contains many null values**

# In[84]:


titanic_data.drop('Cabin', axis = 1,inplace = True)


# In[85]:


titanic_data


# **PassengerId  Feature may be dropped from training dataset as it does not contribute to survival**

# In[86]:


titanic_data.drop('PassengerId', axis = 1,inplace = True)


# **Ticket feature may be dropped down**

# In[87]:


titanic_data.drop('Ticket', axis = 1,inplace = True)


# In[88]:


titanic_data


# # Creating New Fields

# 1. Create New Age Bands to improve  prediction Insights
# 
# 2. Create a new feature called Family based on Parch and SibSp to get total count of family members on board
# 
# 3. Create a Fare range feature if it helps our analysis

# **AGE-BAND**

# In[92]:


titanic_data['Age_band']=0

titanic_data.loc[titanic_data['Age']<=1,'Age_band']="Infant"

titanic_data.loc[(titanic_data['Age']>1)&(titanic_data['Age']<18),'Age_band']="Children"

titanic_data.loc[titanic_data['Age']>=18,'Age_band']="Adults"


# In[95]:


titanic_data


# **Fare-Band**

# In[96]:


titanic_data['FareBand']=0

titanic_data.loc[(titanic_data['Fare']>=0)&(titanic_data['Fare']<=10),'FareBand'] =1

titanic_data.loc[(titanic_data['Fare']>10)&(titanic_data['Fare']<=15),'FareBand'] = 2

titanic_data.loc[(titanic_data['Fare']>15)&(titanic_data['Fare']<=35),'FareBand'] = 3

titanic_data.loc[titanic_data['Fare']>35,'FareBand'] = 4


# In[97]:


titanic_data


# We want to analyze if Name feature can be engineered to extract titles and test correlation between titles and survival, before dropping Name and PassengerId features.
# 
# - In the following code we extract Title feature using regular expressions. The RegEx pattern (\w+\.) matches the first word which ends with a dot character within Name feature. The expand=False flag returns a DataFrame.

# In[104]:


s = pd.Series(['a1', 'b2', 'c3'])
s.str.extract(r'([ab])(\d)')


# In[108]:


titanic_data['Title'] = titanic_data['Name'].str.extract('([A-Za-z]+)\.', expand=False)


# In[109]:


titanic_data


# In[110]:


pd.crosstab(titanic_data['Title'], titanic_data['Sex'])


# In[113]:


a = np.array(["foo", "foo", "foo", "foo", "bar", "bar",
             "bar", "bar", "foo", "foo", "foo"], dtype=object)
b = np.array(["one", "one", "one", "two", "one", "one",
              "one", "two", "two", "two", "one"], dtype=object)
c = np.array(["dull", "dull", "shiny", "dull", "dull", "shiny",
              "shiny", "dull", "shiny", "shiny", "shiny"],dtype=object)


# In[114]:


a,b,c


# In[115]:


pd.crosstab(a, [b, c], rownames=['a'], colnames=['b', 'c'])


# - We can replace many titles with a more common name or classify them as Rare.

# In[117]:


titanic_data['Title'] = titanic_data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

titanic_data['Title'] = titanic_data['Title'].replace('Mlle', 'Miss')
titanic_data['Title'] = titanic_data['Title'].replace('Ms', 'Miss')
titanic_data['Title'] = titanic_data['Title'].replace('Mme', 'Mrs')
    
titanic_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# - We can convert the categorical titles to ordinal.
# - Mr: 1, 
# - Miss: 2, 
# - Mrs: 3, 
# - Master: 4, 
# - Rare: 5

# In[118]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

titanic_data['Title'] = titanic_data['Title'].map(title_mapping)
titanic_data['Title'] = titanic_data['Title'].fillna(0)


# In[119]:


titanic_data


# 
# **Insights**
# 
# - Most titles band Age groups accurately. For example: Master title has Age mean of 5 years.
# - Survival among Title Age bands varies slightly.
# - Certain titles mostly survived (Mme, Lady, Sir) or did not (Don, Rev, Jonkheer).
# 
# ### Decision
# #### We decide to retain the new Title feature for model training

# 
# ### Now we can convert features which contain strings to numerical values. This is required by most model algorithms. Doing so will also help us in achieving the feature completing goal.
# 
# - Converting Sex feature to a new feature called Gender where female=1 and male=0.

# In[120]:


titanic_data['Sex']= titanic_data['Sex'].map({'female': 1, 'male': 0}).astype(int)

titanic_data.head()


# **Extracting Titles Now we can drop down Name feature**

# In[121]:


titanic_data.drop('Name', axis = 1,inplace = True)


# In[122]:


titanic_data['Embarked'] = titanic_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

titanic_data.head()


# - We can also create an artificial feature combining Pclass and Age.

# In[123]:


titanic_data['Age*Class'] = titanic_data.Age * titanic_data.Pclass

titanic_data.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# In[124]:


titanic_data


# # Post Pandas Profiling : Checking Data after data preparation

# In[127]:


import pandas_profiling
profile = pandas_profiling.ProfileReport(titanic_data)
profile.to_file(output_file="Titanic_after_preprocessing.html")


# # Data Visualization

# 4.1 **What is Total Count of Survivals and Victims?**

# In[129]:


titanic_data.groupby(['Survived'])['Survived'].count()# similar functions unique(),sum(),mean() etc


# In[131]:


plt = titanic_data.Survived.value_counts().plot(kind='bar')
plt.set_xlabel('DIED OR SURVIVED')
plt.set_ylabel('Passenger Count')


# **Insights** 
# - Only 342 Passengers Survived out of 891
# - Majority Died which conveys there were less chances of Survival

# --------------------------------------------------------------------------------------------------

# 4.2 **Which gender has more survival rate?**

# In[132]:


titanic_data.groupby(['Survived', 'Sex']).count()["Age"]


# In[133]:


sns.countplot('Survived',data=titanic_data,hue='Sex')


# In[134]:


titanic_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()


# **Insights**
# 
# -  Female has better chances of Survival "LADIES FIRST"
# -  There were more males as compared to females ,but most of them died.

# 4.3 **What is Survival rate based on Person type?**

# In[136]:


titanic_data.groupby(['Survived', 'Age_band']).count()['Sex']


# In[140]:


titanic_data[titanic_data['Age_band'] == 'Adults'].Survived.groupby(titanic_data.Survived).count().plot(kind='pie', figsize=(6, 6),explode=[0,0.05],autopct='%1.1f%%')
plt.axis('equal')
plt.legend(["Died","Survived"])
plt.set_title("Adult survival rate")


# ------------------------------------------**ADULT-SURVIVAL RATE**--------------------------------------------------------------

# In[142]:


titanic_data[titanic_data['Age_band'] == 'Children'].Survived.groupby(titanic_data.Survived).count().plot(kind='pie', figsize=(6, 6),explode=[0,0.05],autopct='%1.1f%%')
plt.axis('equal')
plt.legend(["Died","Survived"])
plt.set_title("Child survival rate")
#plt.show()


# ------------------------------------------**CHILD-SURVIVAL RATE**--------------------------------------------------------------

# In[143]:


titanic_data[titanic_data['Age_band'] == 'Infant'].Survived.groupby(titanic_data.Survived).count().plot(kind='pie', figsize=(6, 6),explode=[0,0.05],autopct='%1.1f%%')
plt.axis('equal')
plt.legend(["Died","Survived"])
plt.set_title("Infant survival rate")
#plt.show()


# **Insights** 
# 
# - Majority Passengers  were  Adults
# 
# - Almost half of the total number of children survived.
# 
# - Most of the Adults failed to Survive
# 
# - More than 85percent of Infant Survived
# 

# 4.4 **Did Economy Class had an impact on survival rate?**

# In[144]:


titanic_data.groupby(['Pclass', 'Survived'])['Survived'].count()


# In[145]:


sns.barplot('Pclass','Survived', data=titanic_data)


# In[146]:


sns.barplot('Pclass','Survived',hue='Sex', data=titanic_data)


# **Insights**
# 
# - Most of the passengers travelled in Third class but only 24per of them survived
# 
# - If we talk about survival ,more passengers in First class survived and again female given more priority
# 
# - Economic Class affected Survival rate and Passengers travelling with First Class had higher ratio of survival as compared to Class 2 and 3.

# 4.5 **What is Survival Propability based on Embarkment of passengers?**

# Titanic’s first voyage was to New York before sailing to the Atlantic Ocean it picked passengers from three ports Cherbourg(C), Queenstown(Q), Southampton(S). Most of the Passengers in Titanicic embarked from the port of Southampton.Lets see how embarkemt affected survival probability.

# In[148]:


sns.countplot('Embarked',data=titanic_data)


# In[150]:


plt = titanic_data[['Embarked', 'Survived']].groupby('Embarked').mean().Survived.plot(kind='bar')
plt.set_xlabel('Embarked')
plt.set_ylabel('Survival Probability')


# **Gender Survival based on Embarkment and Pclass**

# In[152]:


pd.crosstab([titanic_data.Sex, titanic_data.Survived,titanic_data.Pclass],[titanic_data.Embarked], margins=True)


# In[153]:


sns.violinplot(x='Embarked',y='Pclass',hue='Survived',data=titanic_data,split=True)


# In[154]:


sns.catplot(x="Embarked", y="Survived", hue="Sex",
            col="Pclass", aspect=.8,kind='bar',
             data=titanic_data);


# **Insights:**
# 
# - Most Passengers from port C Survived.
# 
# - Most Passengers were from  Southampton(S).
# 
# - Exception in Embarked=C where males had higher survival rate. This could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct correlation between Embarked and Survived.
# 
# - Males had better survival rate in Port C when compared for  S and Q ports.
# - Females had least Survival rate in Pclass 3
# 
# 

# 4.6 **How is Fare distributed for  Passesngers?**

# In[42]:


Titanic_data['Fare'].min()


# In[43]:


Titanic_data['Fare'].max()


# In[44]:


Titanic_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[45]:


Titanic_data.groupby(['FareBand', 'Survived'])['Survived'].count()


# In[45]:


sns.swarmplot(x='Survived', y='Fare', data=Titanic_data);


# **Insights**
# 
# - Majority Passenger's fare lies in 0-100 dollars range
# - Passengers who paid more Fares had more chances of Survival
# - Fare as high as 514 dollars was purcharsed by very few.(Outlier)

# 4.7 **What was Average fare by Pclass & Embark location?**

# In[49]:


sns.boxplot(x="Pclass", y="Fare", data=Titanic_data,hue="Embarked")


# In[48]:


sns.boxplot(x="Embarked", y="Fare", data=Titanic_data)


# **Insights**
# 
# - First Class Passengers paid major part of total Fare.
# - Passengers who Embarked from Port C paid Highest Fare

# 4.8 **Segment Age in bins with size of 10**

# In[49]:


plt=Titanic_data['Age'].hist(bins=20)
plt.set_ylabel('Passengers')
plt.set_xlabel('Age of Passengers')
plt.set_title('Age Distribution of Titanic Passengers',size=17, y=1.08)


# Insights:
# - The youngest passenger on the Titanic were toddlers under 6 months
# - The oldest were of 80 years of age. 
# - The mean for passengers was a bit over 29 years i.e there were more young passengers in the ship.

# **Lets see how Age has correlation with Survival**

# In[50]:



sns.distplot(Titanic_data[Titanic_data['Survived']==1]['Age'])


# In[51]:


sns.distplot(Titanic_data[Titanic_data['Survived']==0]['Age'])


# In[52]:


sns.violinplot(x='Sex',y='Age',hue='Survived',data=Titanic_data,split=True)


# **Insights**
# - Most of the passengers died.
# - Majority of passengers were between 25-40,most of them died
# - Female are more likely to survival 

# 4.9 **Did Solo Passenger has less chances of Survival ?**

# In[52]:


Titanic_data['FamilySize']=0
Titanic_data['FamilySize']=Titanic_data['Parch']+Titanic_data['SibSp']
Titanic_data['SoloPassenger']=0
Titanic_data.loc[Titanic_data.FamilySize==0,'SoloPassenger']=1


# In[53]:


sns.factorplot('SoloPassenger','Survived',data=Titanic_data)


# In[55]:


sns.violinplot(y='SoloPassenger',x='Sex',hue='Survived',data=Titanic_data,split=True)


# In[56]:


sns.factorplot('SoloPassenger','Survived',hue='Pclass',col="Embarked",data=Titanic_data)


# **Insights**
# 
# - Most of the Passengers were travelling Solo and most of them died
# - Solo Females were more likely to Survive as compared to males
# - Passengers Class have a positive correlation with Solo Passenger Survival
# - Passengers Embarked from Port Q had Fifty -Fifty  Chances of Survival 
# 

# 4.10 **How did total family size affected Survival Count**?

# In[57]:


for i in Titanic_data:
    Titanic_data['FamilySize'] = Titanic_data['SibSp'] + Titanic_data['Parch'] + 1

Titanic_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[58]:


sns.barplot(x='FamilySize', y='Survived', hue='Sex', data=Titanic_data)


# **Insights**
# - Both men and women had a massive drop of survival with a FamilySize over 4. 
# - The chance to survive as a man increased with FamilySize until a size of 4
# - Men are not likely to Survive with FamilySize 5 and 6
# - Big Size Family less likihood of Survival

# 4.11 **How can you correlate Pclass/Age/Fare with Survival rate?**

# In[55]:


sns.pairplot(Titanic_data[["FareBand","Age","Pclass","Survived"]],vars= ["FareBand","Age","Pclass"],hue="Survived", dropna=True,markers=["o", "s"])
#plt.set_title('Pair Plot')


# Insights:
# 
# - Fare and Survival has positive correlation
# 
# - We cannt relate age and Survival as majority of travellers were of mid age
# 
# - Higher Class Passengers had more likeihood of Survival
# 

# 4.12 **Which features had most impact on Survival rate?**

# In[60]:


sns.heatmap(Titanic_data.corr(),annot=True)


# **Insights**:
# 
# - Older women have higher rate of survival than older men . Also, older women has higher rate of survival than younger women; an opposite trend to the one for the male passengers.
# - All the features are not necessary to predict Survival
# - More Features creates Complexitity 
# - Fare has positive Correlation
# - For Females major Survival Chances , only for port C males had more likeihood of Survival.
# 

# # Conclusion : "If you were young female travelling in First Class and embarked from port -C then you have best chances of Survival in Titanic"
# 
# -  Most of the Passengers Died
# - "Ladies & Children First" i.e **76% of Females and 16% of Children** Survived
# -  Gender , Passenger type & Classs are mostly realted to Survival.
# -  Survival rate diminishes significantly for Solo  Passengers
# -  Majority of Male Died
# -  Males with Family had better Survival rate as compared to Solo Males

# ******************************************************************************************************************************

# # Part -2

# # <center>Machine Learning </center>       

# ## Table of Contents
# 1. [Logistic Regression](#section1)<br/>
# 2. [KNN or k-Nearest Neighbors](#section2)<br/>
# 3. [Support Vector Machines](#section3)<br/>
# 4. [Naive Bayes classifier](#section4)<br/>
# 5. [Decision Tree](#section5)<br/>
# 6. [Random Forrest](#section6)<br/>
# 7. [Perceptron](#section7)<br/>
# 8. [Artificial neural network](#section8)<br/>
# 9. [RVM or Relevance Vector Machine](#section9)<br/>
# 
# 

# <a id=section1></f> 
# ## Importing Machine Learning Packages

# In[61]:


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
# Data vizualization package
import pandas as pd 
import numpy as np
import random as rnd


# In[62]:


Titanic_data.head()


# In[63]:


Titanic_data['Age_band']=0
Titanic_data.loc[Titanic_data['Age']<=1,'Age_band']=1
Titanic_data.loc[(Titanic_data['Age']>1)&(Titanic_data['Age']<=12),'Age_band']=2
Titanic_data.loc[Titanic_data['Age']>12,'Age_band']=3
Titanic_data.head(2)


# ### Analyze by pivoting features¶
# 
# To confirm some of our observations and assumptions, we can quickly analyze our feature correlations by pivoting features against each other. We can only do so at this stage for features which do not have any empty values. It also makes sense doing so only for features which are categorical (Sex), ordinal (Pclass) or discrete (SibSp, Parch) type.
# 
# - **Pclass**: We observe significant correlation (>0.5) among Pclass=1 and Survived (classifying #3). We decide to include this feature in our model.
# - **Sex** :We confirm the observation during problem definition that Sex=female had very high survival rate at 74% (classifying #1).
# - **SibSp and Parch** : These features have zero correlation for certain values. It may be best to derive a feature or a set of features from these individual features (creating #1).

# In[64]:


Titanic_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[65]:


Titanic_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[66]:


Titanic_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# 
# #### Observations form EDA on Categorical Features
# 
# - Female passengers had much better survival rate than males.  **Classifying** .
# - Exception in Embarked=C where males had higher survival rate. This could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct correlation between Embarked and Survived.
# - Males had better survival rate in Pclass=3 when compared with Pclass=2 for C and Q ports.**Correlatring**
# - Ports of embarkation have varying survival rates for Pclass=3 and among male passengers. **Correlating**.
# 
# #### Decisions.
# - Add Sex feature to model training.
# - Complete and add Embarked feature to model training.

#  There are 60+ predictive modelling algorithms to choose from. We must understand the type of problem and solution requirement to narrow down to a select few models which we can evaluate. Here our problem is a **classification** and **regression** problem.
# 
# #### Lets identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port) and perform a category of machine learning which is called supervised learning 

# <a id=section1></f> 
# ## 1. Logistic Regression

# - Logistic regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome.
# - Logistic Regression is used when the dependent variable(target) is categorical.
# - Logistic regression measures the relationship between the categorical dependent variable (feature) and one or more independent variables (features) by estimating probabilities using a logistic function, which is the cumulative logistic distribution.

# In[67]:


Titanic_data.shape, Titanic_test.shape


# In[68]:


Titanic_test = Titanic_test.drop(['Ticket', 'Cabin','Name'], axis=1)


# In[69]:


X_titanic = Titanic_data.drop("Survived", axis=1)
Y_titanic = Titanic_data["Survived"]
X_test  = Titanic_test.drop("PassengerId", axis=1).copy()
X_titanic.shape, Y_titanic.shape, X_test.shape


# In[78]:


#Titanic_test  = Titanic_test.drop("PassengerId", axis=1)
Titanic_test.head()


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_titanic, Y_titanic)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_titanic, Y_titanic) * 100, 2)
acc_log


# - **We can use Logistic Regression to validate our assumptions and decisions for feature creating and completing goals. This can be done by calculating the coefficient of the features in the decision function**.

# In[80]:


coeff_df = pd.DataFrame(Titanic_data.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# 
# Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).
# 
# #### Insights 
# - Sex is highest positivie coefficient, implying as the Sex value increases (male: 0 to female: 1), the probability of Survived=1 increases the most.
# - Inversely as Pclass increases, probability of Survived=1 decreases the most.
# - This way Age*Class is a good artificial feature to model as it has second highest negative correlation with Survived.
# - So is Title as second highest positive correlation.

# ###  Support Vector Machines(SVM)
#  Support-vector machines also support-vector networks) are supervised learning models with associated learning algorithms that analyze data used for **classification** and **regression analysis**.

# In[ ]:



svc = SVC()
svc.fit(X_titanic, Y_titanic)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# ### k-Nearest Neighbors algorithm
# In pattern recognition, the k-Nearest Neighbors algorithm (or k-NN for short) is a non-parametric method used for classification and regression. A sample is classified by a majority vote of its neighbors, with the sample being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# ### Naive Bayes
# Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features. Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features) in a learning problem. 
# 
# The model generated confidence score is the lowest among the models evaluated so far.

# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# ### Perceptron
# The perceptron is an algorithm for supervised learning of binary classifiers (functions that can decide whether an input, represented by a vector of numbers, belongs to some specific class or not). It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector. The algorithm allows for online learning, in that it processes elements in the training set one at a time.

# In[ ]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[ ]:




