from fastapi import FastAPI
import math
import pandas as pd
import numpy as np
import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# my imports
from models import Urls
from helpers import columns, arranged_cols_1, arranged_cols_2, data_columns, truncate

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO please sort env
load_dotenv()

api_key = os.getenv("PRIVATE_KEY").replace('\\n', '\n')
print(len(api_key))

service = dict(type="service_account",
               project_id=os.getenv("PROJECT_ID"),
               private_key_id=os.getenv("PRIVATE_KEY_ID"),
               private_key=api_key,
               client_email=os.getenv("CLIENT_EMAIL"),
               client_id=os.getenv("CLIENT_ID"),
               auth_uri=os.getenv("AUTH_URI"),
               token_uri=os.getenv("TOKEN_URI"),
               auth_provider_x509_cert_url=os.getenv("AUTH_PROVIDER"),
               client_x509_cert_url=os.getenv("CLIENT_CERT"),
               universe_domain=os.getenv("DOMAIN"),
               )

print(service)

try:
    cred = credentials.Certificate(service)
    firebase_admin.initialize_app(cred)
    print("Firebase ready")
except Exception as e:
    print(e)
    print("Could not connect")

db = firestore.client()

qb_max_td: float
qb_max_yards: float
qb_max_int: float
max_pts: float
rushers_max_td: float
rushers_max_yards: float
rushers_max_rec: float
wr_max_td: float
wr_max_yards: float
wr_max_rec: float
te_max_td: float
te_max_yards: float
te_max_rec: float


def assign_weights(row):
    # find pts
    tds = row["PTD"] + row["RTD"] + row["RETD"]
    yds = row["PYDS"] + row["RYDS"] + row["REYDS"]
    rec = row["REC"]
    interceptions = row["INT"]
    year_one = row["Last year"]
    # you started keeping track with CSV 7
    if math.isnan(year_one):
        year_one = 0

    if row["Pos"] == "QB":
        ratio_tds = tds / qb_max_td
        ratio_yds = yds / qb_max_yards
        ratio_int = interceptions / qb_max_int
        assigned_tds = ratio_tds * 10
        assigned_yds = ratio_yds * 10
        assigned_int = ratio_int * 2
        projections_sum_value = assigned_tds + assigned_yds - assigned_int

        total_pts = year_one
        ratio = total_pts / max_pts
        assigned_pts = ratio * 4
        sum_value = assigned_pts + projections_sum_value

        row["Value"] = sum_value
    elif row["Pos"] == "WR":

        ratio_tds = tds / wr_max_td
        ratio_yds = yds / wr_max_yards
        ratio_rec = rec / int(wr_max_rec)
        assigned_tds = ratio_tds * 11
        assigned_yds = ratio_yds * 7
        assigned_rec = ratio_rec * 2
        projections_sum_value = assigned_tds + assigned_yds + assigned_rec

        total_pts = year_one
        ratio = total_pts / max_pts
        assigned_pts = ratio * 5
        sum_value = assigned_pts + projections_sum_value

        row["Value"] = sum_value
    elif row["Pos"] == "TE":

        ratio_tds = tds / te_max_td
        ratio_yds = yds / te_max_yards
        ratio_rec = rec / int(te_max_rec)
        assigned_tds = ratio_tds * 12
        assigned_yds = ratio_yds * 7.85
        assigned_rec = ratio_rec * 2.155
        projections_sum_value = assigned_tds + assigned_yds + assigned_rec

        total_pts = year_one
        ratio = total_pts / max_pts
        assigned_pts = ratio * 3
        sum_value = assigned_pts + projections_sum_value - 1.5

        row["Value"] = sum_value
    else:
        ratio_tds = tds / rushers_max_td
        ratio_yds = yds / rushers_max_yards
        ratio_rec = rec / int(rushers_max_rec)
        assigned_tds = ratio_tds * 11
        assigned_yds = ratio_yds * 7.5
        assigned_rec = ratio_rec * 2
        projections_sum_value = assigned_tds + assigned_yds + assigned_rec

        total_pts = year_one
        ratio = total_pts / max_pts
        assigned_pts = ratio * 4.5
        sum_value = assigned_pts + projections_sum_value

        row["Value"] = sum_value

    return row


@app.get("/")
def read_root():
    return {"is": "working"}


@app.post("/create-csv")
def create_csv(urls: Urls):
    # 1. pull all the files and update the data
    passing_data = pd.read_csv(urls.passing_url)
    rushing_data = pd.read_csv(urls.rushing_url)
    receiving_data = pd.read_csv(urls.receiving_url)

    if len(passing_data.iloc[0, :]) != len(columns) or len(rushing_data.iloc[0, :]) != len(columns) or len(
            receiving_data.iloc[0, :]) != len(columns):
        message = "Please ensure that the CSVs have equal columns of size " + str(len(columns))
        return {"message": message}

    # 2. arrange the csv columns using columns
    passing_data.columns = columns
    passing_data = passing_data.iloc[1:, :]

    rushing_data.columns = columns
    rushing_data = rushing_data.iloc[1:, :]

    receiving_data.columns = columns
    receiving_data = receiving_data.iloc[1:, :]

    # 3.make sure data only has players with allowed position
    passing_data = passing_data[passing_data["Pos"] == "QB"]

    rushing_data = rushing_data[(rushing_data["Pos"] == "RB") | (rushing_data["Pos"] == "FB")]

    receiving_data = receiving_data[(receiving_data["Pos"] == "WR") | (receiving_data["Pos"] == "TE")]

    # 4.Add old data(old points)
    # -> add new column
    unknowns = np.full(shape=len(passing_data), fill_value=np.nan)
    passing_data["Two years ago"] = unknowns

    unknowns = np.full(shape=len(rushing_data), fill_value=np.nan)
    rushing_data["Two years ago"] = unknowns

    unknowns = np.full(shape=len(receiving_data), fill_value=np.nan)
    receiving_data["Two years ago"] = unknowns

    # -> old data csv file
    old_dataset = pd.read_csv("old_data_3.csv")
    old_data = {"name": old_dataset["Name"].values, "two_years_ago": old_dataset["Two years ago"].values,
                "last_year": old_dataset["Last year"].values}

    # ->file a way to loop through csv file and give data where necessary
    "passing - start"
    names = passing_data["Name"]

    two_year_pts = []
    last_year_pts = []

    for name in names:
        index_test = np.where(old_data["name"] == name)[0]

        if len(index_test) > 0:
            two_year_pts.append(old_data["two_years_ago"][index_test[0]])
            last_year_pts.append(old_data["last_year"][index_test[0]])
        else:
            two_year_pts.append(np.nan)
            last_year_pts.append(np.nan)

    passing_data["Two years ago"] = two_year_pts
    passing_data["Last year"] = last_year_pts
    "passing - end"

    "rushing - start"
    names = rushing_data["Name"]

    two_year_pts = []
    last_year_pts = []

    for name in names:
        index_test = np.where(old_data["name"] == name)[0]

        if len(index_test) > 0:
            two_year_pts.append(old_data["two_years_ago"][index_test[0]])
            last_year_pts.append(old_data["last_year"][index_test[0]])
        else:
            two_year_pts.append(np.nan)
            last_year_pts.append(np.nan)

    rushing_data["Two years ago"] = two_year_pts
    rushing_data["Last year"] = last_year_pts
    "rushing - end"

    "receiving - start"
    names = receiving_data["Name"]

    two_year_pts = []
    last_year_pts = []

    for name in names:
        index_test = np.where(old_data["name"] == name)[0]

        if len(index_test) > 0:
            two_year_pts.append(old_data["two_years_ago"][index_test[0]])
            last_year_pts.append(old_data["last_year"][index_test[0]])
        else:
            two_year_pts.append(np.nan)
            last_year_pts.append(np.nan)

    receiving_data["Two years ago"] = two_year_pts
    receiving_data["Last year"] = last_year_pts
    "receiving - end"

    # 5.Aragnge the columns
    passing_data = passing_data[arranged_cols_1]
    rushing_data = rushing_data[arranged_cols_1]
    receiving_data = receiving_data[arranged_cols_1]

    # 6.combine dataset and send it to firebase

    combined_csv = pd.concat([passing_data, receiving_data, rushing_data], axis=0)

    data = combined_csv

    for col in data.columns[3:]:
        data[col] = data[col].astype("float")
    # 2.Change necessary column values to float datatype

    # 3.Add last year and this and fill them with nan values

    # 4.convert the old passing data to dictionary

    # 5.add old data to their respective columns in dataframe

    # 6. Function that calculates max pts and everything
    global qb_max_td, qb_max_yards, qb_max_int, rushers_max_td, rushers_max_yards, rushers_max_rec, wr_max_td, \
        wr_max_yards, wr_max_rec, te_max_td, te_max_rec, te_max_yards, max_pts

    qb_max_td = data[data["Pos"] == "QB"]["PTD"].max() + data[data["Pos"] == "QB"]["RTD"].max()
    wr_max_td = data[data["Pos"] == "WR"]["PTD"].max() + data[data["Pos"] == "WR"]["RTD"].max() + data[data["Pos"] ==
                                                                                                       "WR"][
        "RETD"].max()
    te_max_td = data[data["Pos"] == "TE"]["PTD"].max() + data[data["Pos"] == "TE"]["RETD"].max() + \
                data[data["Pos"] == "TE"]["RTD"].max()
    rb_max_td = data[data["Pos"] == "RB"]["PTD"].max() + data[data["Pos"] == "RB"]["RETD"].max() + \
                data[data["Pos"] == "RB"]["RTD"].max()
    fb_max_td = data[data["Pos"] == "FB"]["PTD"].max() + data[data["Pos"] == "FB"]["RETD"].max() + \
                data[data["Pos"] == "FB"]["RTD"].max()

    te_max_td = (te_max_td + wr_max_td) / 2

    qb_max_yards = data[data["Pos"] == "QB"]["PYDS"].max() + data[data["Pos"] == "QB"]["RYDS"].max() + \
                   data[data["Pos"] == "QB"]["REYDS"].max()
    wr_max_yards = data[data["Pos"] == "WR"]["PYDS"].max() + data[data["Pos"] == "WR"]["RYDS"].max() + \
                   data[data["Pos"] == "WR"]["REYDS"].max()
    te_max_yards = data[data["Pos"] == "TE"]["PYDS"].max() + data[data["Pos"] == "TE"]["RYDS"].max() + \
                   data[data["Pos"] == "TE"]["REYDS"].max()
    rb_max_yards = data[data["Pos"] == "RB"]["PYDS"].max() + data[data["Pos"] == "RB"]["RYDS"].max() + \
                   data[data["Pos"] == "RB"]["REYDS"].max()
    fb_max_yards = data[data["Pos"] == "FB"]["PYDS"].max() + data[data["Pos"] == "FB"]["RYDS"].max() + \
                   data[data["Pos"] == "FB"]["REYDS"].max()

    wr_max_rec = data[data["Pos"] == "WR"]["REC"].max()
    te_max_rec = data[data["Pos"] == "TE"]["REC"].max()
    rb_max_rec = data[data["Pos"] == "RB"]["REC"].max()
    fb_max_rec = data[data["Pos"] == "FB"]["REC"].max()

    qb_max_int = data[data["Pos"] == "QB"]["INT"].max()

    max_pts = data['Last year'].max()

    if 'Unnamed: 0' in data.columns:
        data = data[data.columns.drop('Unnamed: 0')]

    if len(data.columns) != len(data_columns):
        return {
            "message": "there seems to be a problem with out column headers please, the CSV link may not have been "
                       "created successfully"}

    rushers_max_td = fb_max_td if (fb_max_td > rb_max_td) else rb_max_td
    rushers_max_yards = fb_max_yards if fb_max_yards > rb_max_yards else rb_max_yards
    rushers_max_rec = fb_max_rec if fb_max_rec > rb_max_rec else rb_max_rec

    # 7. Run weight assigner
    new_data = data.apply(assign_weights, axis=1)
    name_team_pos_data = {"name": new_data["Name"].values,
                          "team": new_data["Team"].values,
                          "pos": new_data["Pos"].values}
    new_data = new_data.groupby(['Name'], as_index=False).sum()
    unknowns = np.full(shape=len(new_data), fill_value=np.nan)
    new_data["Team"] = unknowns
    new_data["Pos"] = unknowns
    new_data = new_data[arranged_cols_2]
    names = new_data["Name"].values

    values_db = []
    pos_db = []
    names_db = []
    teams = []
    pos = []
    values = []

    # Put teams and postions back into the dataset
    for name in names:
        index_test = np.where(name_team_pos_data["name"] == name)[0]

        if len(index_test) > 1:
            teams.append(name_team_pos_data["team"][index_test[1]])
            pos.append(name_team_pos_data["pos"][index_test[1]])
        elif len(index_test) > 0:
            teams.append(name_team_pos_data["team"][index_test[0]])
            pos.append(name_team_pos_data["pos"][index_test[0]])
        else:
            teams.append(np.nan)
            pos.append(np.nan)

    new_data["Team"] = teams
    new_data["Pos"] = pos

    # Why are we doing this?
    # this is to get a reference of the players previous value
    docs = db.collection("nfl_players").stream()

    for doc in docs:
        pos_db.append(doc.to_dict()["position"])
        names_db.append(doc.to_dict()["name"])
        values_db.append(doc.to_dict()["value"])

    for i in range(len(names)):

        found_values = np.where((np.array(names_db) == names[i]) & (np.array(pos_db) == pos[i]))[0]
        if len(found_values) > 0:
            values.append(values_db[found_values[0]])
        elif len(found_values) == 0:
            values.append(np.nan)

    new_data["Old_Value"] = values
    results_csv = new_data

    results_csv.to_csv("results.csv")

    # 8.Return as JSON

    return json.dumps(json.loads(results_csv[['Name', 'Team', 'Pos', 'Old_Value', 'Value']].to_json(orient="records")))


@app.post("/save-to-db")
def save_to_db():
    results_csv = pd.read_csv("results.csv")
    if not results_csv.empty:

        docs_sleeper = db.collection("sleeper_players").stream()

        ids_db = []
        pos_db = []
        names_db = []

        for doc in docs_sleeper:
            ids_db.append(doc.id)
            pos_db.append(doc.to_dict()["position"])
            names_db.append(doc.to_dict()["first_name"] + " " + doc.to_dict()["last_name"])

        ids_db = np.array(ids_db)
        pos_db = np.array(pos_db)
        names_db = np.array(names_db)

        if len(ids_db) > 0:
            collection = db.collection("sleeper_players")
            names = results_csv["Name"].values
            pos = results_csv["Pos"].values
            values = results_csv["Value"].values

            for i in range(len(names)):

                ids = ids_db[np.where((names_db == names[i]) & (pos_db == pos[i]))[0]]

                if len(ids) > 0:
                    doc_ref = collection.document(ids[0])
                    doc_ref.update({"value": truncate(values[i], 2)})

        # Update NFL players now
        docs = db.collection("nfl_players").stream()

        ids_db = []
        pos_db = []
        names_db = []

        for doc in docs:
            ids_db.append(doc.id)
            pos_db.append(doc.to_dict()["position"])
            names_db.append(doc.to_dict()["name"])

        ids_db = np.array(ids_db)
        pos_db = np.array(pos_db)
        names_db = np.array(names_db)

        if len(ids_db) > 0:
            collection = db.collection("nfl_players")
            names = results_csv["Name"].values
            pos = results_csv["Pos"].values
            values = results_csv["Value"].values

            for i in range(len(names)):

                ids = ids_db[np.where((names_db == names[i]) & (pos_db == pos[i]))[0]]

                if len(ids) > 0:
                    doc_ref = collection.document(ids[0])
                    doc_ref.update({"value": truncate(values[i], 2)})

        return {"message": "Your data has been updated"}

    return {"message": "something went wrong you are probably sending over an empty dataframe"}
