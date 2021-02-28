import random
import pandas as pd
import numpy as np

def clean(df, all_cat = True):
    ## Embarked ##
    df = df[df.Embarked.notna()]
    
    ## Ticket ##
    df['Ticket_unique'] = df.Ticket.apply(lambda x: True if len(df[df.Ticket == x]) == 1 else False)
    df['Ticket_numeric'] = df.Ticket.apply(lambda x: True if x.isnumeric() else False)

    ## Name ##
    # Most common titles
    titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs','Rev']
    df['Title'] = df.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip() if x.split(',')[1].split('.')[0].strip() in titles else "Other")

    ## Cabin ##
    # Number of Cabins assigned
    df['Num_cabins'] = df.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
    df['Cabin_letter'] = df.Cabin.apply(lambda x: x[0] if pd.notna(x) else 'U')
    df = df[df.Cabin_letter != 'T']

    ## Age ##
    # fill null Age values with random numbers drawn from existing distributions 
    groups = [(s,p) for s in df.Sex.unique() for p in df.Pclass.unique()]
    rand_age = []
    for s,p in groups:
        n_null = len(df[df.Age.isna() & (df.Sex == s) & (df.Pclass == p)])
        rand_age += random.choices(df[df.Age.notna() & (df.Sex == s) & (df.Pclass == p)].Age.astype(int).to_list(), k = n_null)
    # fill null values for Age
    df.loc[pd.isna(df.Age), 'Age'] = rand_age
    if all_cat == True:
        # bin ages
        #age_labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30','31-35','36-40','41-50','>50']
        #age_bins = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 100])
        #df.loc[:,'Age'] = df.Age.apply(lambda x : (x <= age_bins).argmax())
        df.loc[:,'Age'] = pd.qcut(df.Age, q = 4)    

    ## Fare ##
    df = df[df.Fare.notna()]
    #df['Fare'] = np.log10(df.Fare + 1)
    if all_cat == True:
        # Bin the fares into sensible ranges
        #fare_labels = ['0', '1-7.5','7.5-10', '10-15', '16-25','26-35','36-50','51-100','>100']
        #fare_bins = np.array([-2, 0, 7.5, 10, 15, 25, 35, 50, 100, 1000])
        #df.loc[:,'Fare'] = df.Fare.apply(lambda x : (x <= fare_bins).argmax())   
        df.loc[:,'Fare'] = pd.qcut(df.Fare, q = 4)     

    ## Parch / SibSp ##
    df['Group_size'] = df['Parch'] + df['SibSp']
    df.drop(columns = ['Parch','SibSp'], inplace = True)

    ## PClass
    df['Pclass'] = df.Pclass.astype(str)
    
    drop_cols = ['Cabin','Name','Ticket','PassengerId','Ticket_numeric'] #+ ['Cabin_letter','Num_cabins','Embarked','Group_size']#,'Fare','Ticket_unique']
    df.drop(columns = drop_cols, inplace = True)

    return df