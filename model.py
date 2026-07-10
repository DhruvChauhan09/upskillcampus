import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#df_area = pd.read_csv("datafile.csv")
df_cost = pd.read_csv("datafile (1).csv")
df_production = pd.read_csv("datafile (2).csv")
df_cropinfo = pd.read_csv("datafile (3).csv")
#df_extra = pd.read_csv("produce.csv")



#print(df_area.info())
#print(df_area.isnull().sum()) #1

#print(df_cost.info())
#print(df_cost.isnull().sum()) #0

#print(df_production.info())
#print(df_production.isnull().sum()) #0

#print(df_cropinfo.info())
#print(df_cropinfo.isnull().sum()) #0,0,28,1,78

#print(df_extra.info())
#print(df_extra.isnull().sum())


df_cropinfo = df_cropinfo.drop(columns=["Unnamed: 4"])
df_cropinfo["Season/ duration in days"] = df_cropinfo["Season/ duration in days"].fillna("Unknown")
df_cropinfo["Recommended Zone"] = df_cropinfo["Recommended Zone"].fillna("Unknown")

#print(df_cropinfo.isnull().sum())

#print(df_production.columns.tolist())
#print(df_cost.columns.tolist())
#print(df_cropinfo.columns.tolist())


df_cost["Crop"] = df_cost["Crop"].str.strip().str.lower()

df_production["Crop"] = df_production["Crop"].str.strip().str.lower()

df_cropinfo["Crop"] = df_cropinfo["Crop"].str.strip().str.lower()




rows = []

years = [
    "2006-07",
    "2007-08",
    "2008-09",
    "2009-10",
    "2010-11"
]

for _, row in df_production.iterrows():

    crop = row["Crop"]

    for year in years:

        rows.append({
            "Crop": crop,
            "Year": year,
            "Production": row[f"Production {year}"],
            "Area": row[f"Area {year}"],
            "Yield_Yearly": row[f"Yield {year}"]
        })

df_long = pd.DataFrame(rows)
#print(df_long.head())
#print(df_long.shape)


#print(df_cost["Crop"].value_counts().head(20))

#print(df_cropinfo["Crop"].value_counts().head(20))


#print(df_cost.columns.tolist())


cost_summary = df_cost.groupby("Crop").agg({
    "Cost of Cultivation (`/Hectare) A2+FL":"mean",
    "Cost of Cultivation (`/Hectare) C2":"mean",
    "Cost of Production (`/Quintal) C2":"mean",
    "Yield (Quintal/ Hectare) ":"mean"
}).reset_index()

#print(cost_summary.head())
#print(cost_summary.shape)




cropinfo_summary = df_cropinfo.groupby("Crop").agg({
    "Variety":"first",
    "Season/ duration in days":"first",
    "Recommended Zone":"first"
}).reset_index()
#print(cropinfo_summary.head())



df = pd.merge(
    df_long,
    cost_summary,
    on="Crop",
    how="left"
)
#print(df.shape)

df = pd.merge(
    df,
    cropinfo_summary,
    on="Crop",
    how="left"
)
#print(df.shape)


#print(df.shape)
#print(df.columns.tolist())
#print(df.isnull().sum())




prod_crops = set(df_long["Crop"])
cost_crops = set(cost_summary["Crop"])

#print(prod_crops - cost_crops)

cropinfo_crops = set(cropinfo_summary["Crop"])

#print(prod_crops - cropinfo_crops)


# filling 
num_cols = [
    "Cost of Cultivation (`/Hectare) A2+FL",
    "Cost of Cultivation (`/Hectare) C2",
    "Cost of Production (`/Quintal) C2",
    "Yield (Quintal/ Hectare) "
]

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = [
    "Variety",
    "Season/ duration in days",
    "Recommended Zone"
]

for col in cat_cols:
    df[col] = df[col].fillna("Unknown")
#print(df.isnull().sum())    

#duplicates
df.drop_duplicates(inplace=True)

#year
df["Year"] = df["Year"].str[:4]

df["Year"] = df["Year"].astype(int)


#encode categ
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cat_cols = [
    "Crop",
    "Variety",
    "Season/ duration in days",
    "Recommended Zone"
]

for col in cat_cols:
    df[col] = le.fit_transform(
        df[col].astype(str)
    )


#feature sel
X = df.drop(
    ["Production","Yield_Yearly"],
    axis=1
)
y = df["Production"]



#train/test split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


#prediction
pred = model.predict(X_test)


#evaluation
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

mae = mean_absolute_error(
    y_test,
    pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred
    )
)

r2 = r2_score(
    y_test,
    pred
)

print("MAE =",mae)
print("RMSE =",rmse)
print("R2 =",r2)