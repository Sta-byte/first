
import bs4
import pandas as pd
import requests
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection
import pickle
website = 'https://backlinko.com/social-media-users'
website_url = requests.get(website).text
soup = BeautifulSoup(website_url, 'html.parser')
my_table = soup.find('tbody')
print(soup.prettify())
my_table = soup.find('table')
df = pd.DataFrame(my_table)
soup.title
table_data = []
for row in my_table.findAll('tr'):
    row_data = []
    for cell in row.findAll('td'):
        row_data.append(cell.text)
    if (len(row_data)>0):
        data_item = {"Social_network": row_data[0],
                     "Male_(% usage)": row_data[1],
                     "Female_(% usage)": row_data[2],

                     }
        table_data.append(data_item)
        df = pd.DataFrame(table_data)
        df.to_excel('data.xlsx', index=True)
        df.shape
        df.isnull().sum()
        df.info()
        df['Male_(% usage)'] = pd.to_numeric(df['Male_(% usage)'], errors='coerce').astype('Int64')
        df['Female_(% usage)'] = pd.to_numeric(df['Female_(% usage)'], errors='coerce').astype('Int64')
        df.info()
        df
        df['Social_Media_Users(Total%)'] = df['Male_(% usage)'] + df['Female_(% usage)']
        df
        # SOCIAL MEDIA USAGE
        df_quantity = df.groupby(["Social_Media_Users(Total%)"])["Social_network"].sum().reset_index()
        plt.figure(figsize=(15, 10))
        sns.barplot(x="Social_Media_Users(Total%)", y="Social_network", data=df_quantity)
        plt.title("Social Network % Usage")
        plt.xlabel("usage")
        plt.ylabel("Social_Network")

        # SOCIAL MEDIA USAGE
        df_quantity = df.groupby(["Social_Media_Users(Total%)"])["Social_network"].sum().reset_index()
        plt.figure(figsize=(15, 10))
        sns.relplot(x="Social_Media_Users(Total%)", y="Social_network", data=df_quantity)
        plt.title("Social Network % Usage")
        plt.xlabel("usage")
        plt.ylabel("Social_Network")

        df1 = df.drop(['Social_network'], axis=1)
        df1
        # Creating x (all the feature columns)
        x = df1.drop("Social_Media_Users(Total%)", axis=1)
        # Creating y (the target column)
        y = df1["Social_Media_Users(Total%)"]
        x.shape
        y.shape

        # Split the data into training and test sets

        from sklearn.model_selection import train_test_split


        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=8)

        # Random Forest
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestRegressor()
    from sklearn.model_selection import train_test_split


    # Saving
    pickle.dump(RandomForestRegressor(), open("RandomForestRegressor_model_1.pkl", "wb"))

    # Load a saved model
    loaded_pickle_model_r = pickle.load(open("RandomForestRegressor_model_1.pkl", "rb"))


