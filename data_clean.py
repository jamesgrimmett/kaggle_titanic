import random
import pandas as pd
import numpy as np

def clean(df, all_cat = True):
    ## Embarked ##
    df = df[df.Embarked.notna()]

    ## Ticket ##
    df['Ticket_unique'] = df.Ticket.apply(lambda x: True if len(df[df.Ticket == x]) == 1 else False)
    #df['Ticket_numeric'] = df.Ticket.apply(lambda x: True if x.isnumeric() else False)

    ## Name ##
    # Most common titles
    common = ['Master', 'Miss', 'Mr', 'Mrs', 'Rev', 'Dr']
    title_map = {'Don' : 'Mr', 'Mme' : 'Mrs', 'Ms' : 'Miss', 'Major' : 'Mr', 
                'Lady' : 'Mrs', 'Sir' : 'Mr', 'Mlle' : 'Mrs', 
                'Col' : 'Mr', 'Capt' : 'Mr', 'the Countess' : 'Mrs', 
                'Jonkheer' : 'Mr', 'Dona' : 'Mrs'}
    df['Title'] =  df.Name.apply(lambda x:
            x.split(',')[1].split('.')[0].strip())
           # if x.split(',')[1].split('.')[0].strip() in common
           # else "Other")
    df.loc[~df.Title.isin(common), 'Title'] = df[~df.Title.isin(common)].Title.map(title_map)

    ## Cabin ##
    # Number of Cabins assigned
    df['Num_cabins'] = df.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
    df['Cabin_letter'] = df.Cabin.apply(lambda x: x[0] if pd.notna(x) else 'U')
    df = df[df.Cabin_letter != 'T']

    ## Age ##
    # fill null Age values with random numbers drawn from existing distributions 
    groups = [(t,p) for t in df.Title.unique() for p in df.Pclass.unique()]
  
    rand_age = []
    for (t,p) in groups:
        n_null = len(df[df.Age.isna() & (df.Title == t) & (df.Pclass == p)])
        select_from = df[df.Age.notna() & (df.Title == t) & (df.Pclass == p)].Age.astype(int).to_list()
        if len(select_from) == 0:
            #select_from = np.ones(n_null) * df[df.Age.notna() & (df.Pclass == p)].Age.mean()
            select_from = df[df.Age.notna() & (df.Pclass == p)].Age.astype(int).to_list()
        vals = random.choices(select_from, k = n_null)
        rand_age += vals
    # fill null values for Age
    df.loc[pd.isna(df.Age), 'Age'] = rand_age
    df['Age'] = df.Age.astype(int)
    if all_cat == True:
        # bin ages
        q_age = 20
        age_bins = pd.qcut(df.Age, q = q_age, retbins = True)[1]
        age_labels = (age_bins + np.roll(age_bins,-1))[:-1] / 2
        df.loc[:,'Age'] = pd.qcut(df.Age, q = q_age, labels = age_labels)  

    ## Fare ##
    mean_fare_pclass = [df[df.Pclass == 1].Fare.mean(),
                        df[df.Pclass == 2].Fare.mean(),
                        df[df.Pclass == 3].Fare.mean()]
    nullfare_pclasses = df[df.Fare.isna()].Pclass.to_list()
    df.loc[df.Fare.isna(), 'Fare'] = [mean_fare_pclass[p - 1] for p in nullfare_pclasses]
    df['Fare'] = [np.log10(f + 1) for f in df.Fare]
    if all_cat == True:
        q_fare = 15
        fare_bins = pd.qcut(df.Fare, q = q_fare, retbins = True)[1]
        fare_labels = (fare_bins + np.roll(fare_bins,-1))[:-1] / 2
        df.loc[:,'Fare'] = pd.qcut(df.Fare, q = q_fare, labels = fare_labels)

    ## Parch / SibSp ##
    #df['Group_size'] = df['Parch'] * df['SibSp']
    #df.drop(columns = ['Parch','SibSp'], inplace = True)

    ## PClass
    df['Pclass'] = df.Pclass.astype(str)

    drop_cols = ['Cabin','Name','Ticket']
    df.drop(columns = drop_cols, inplace = True)

    return df