import pandas as pd

# Tamil
df1 = pd.read_csv("predictions_goog_mur_res_Tamil.csv")
df2 = pd.read_csv("predictions_indic_Tamil.csv")
df3 = pd.read_csv("predictions_ind_res_Tamil.csv")


df1 = df1[["image_id", "predicted_labels"]]
df2 = df2[["image_id", "predicted_labels"]]
df3 = df3[["image_id", "predicted_labels"]]

# Merge the dataframes on image_id
df_merged = (
    df1.merge(df2, on="image_id", suffixes=("_m1", "_m2"))
       .merge(df3, on="image_id", suffixes=("", "_m3"))
)


def parse_label(label_str):
    # Strip brackets and convert to integer
    return int(label_str.strip("[]"))


df_merged["label_m1"] = df_merged["predicted_labels_m1"].apply(parse_label)
df_merged["label_m2"] = df_merged["predicted_labels_m2"].apply(parse_label)
df_merged["label_m3"] = df_merged["predicted_labels"].apply(parse_label)

def majority_vote(row):
    s = row["label_m1"] + row["label_m2"] + row["label_m3"]
    # If sum of labels >= 2, final label is 1, else 0
    return 1 if s >= 2 else 0

df_merged["final_label"] = df_merged.apply(majority_vote, axis=1)


df_final = df_merged[["image_id", "final_label"]]
df_final.to_csv("output_res_Tamil.csv", index=False)




# Malayalam
df1 = pd.read_csv("predictions_goog_mur_res_Malayalam.csv")
df2 = pd.read_csv("predictions_indic_Malayalam.csv")
df3 = pd.read_csv("predictions_ind_res_Malayalam.csv")


df1 = df1[["image_id", "predicted_labels"]]
df2 = df2[["image_id", "predicted_labels"]]
df3 = df3[["image_id", "predicted_labels"]]

# Merge the dataframes on image_id
df_merged = (
    df1.merge(df2, on="image_id", suffixes=("_m1", "_m2"))
       .merge(df3, on="image_id", suffixes=("", "_m3"))
)


def parse_label(label_str):
    # Strip brackets and convert to integer
    return int(label_str.strip("[]"))


df_merged["label_m1"] = df_merged["predicted_labels_m1"].apply(parse_label)
df_merged["label_m2"] = df_merged["predicted_labels_m2"].apply(parse_label)
df_merged["label_m3"] = df_merged["predicted_labels"].apply(parse_label)

def majority_vote(row):
    s = row["label_m1"] + row["label_m2"] + row["label_m3"]
    # If sum of labels >= 2, final label is 1, else 0
    return 1 if s >= 2 else 0

df_merged["final_label"] = df_merged.apply(majority_vote, axis=1)


df_final = df_merged[["image_id", "final_label"]]
df_final.to_csv("output_res_Malayalam.csv", index=False)


