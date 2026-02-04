from pathlib import Path
import openml
from pandas import CategoricalDtype

from statsformer.data.dataset import DEFAULT_TRAIN_RATIOS, Dataset
from statsformer.models.base import ModelTask


def build_nomao_dataset(
    data_output_dir: str | Path,
    num_splits: int=10,
    train_ratios: list[float]=DEFAULT_TRAIN_RATIOS,
    seed: int=42,
    max_dataset_size: int | None=1000,
):
    dataset = openml.datasets.get_dataset(1486)

    name_mapping = {}
    for line in NAMES_RAW.split("\n"):
        if line.strip():
            idx, rest = line.split(" ", 1)
            name, _ = rest.split(":", 1)
            name_mapping[f"V{int(idx)-1}"] = f"{name.strip()}"
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    X.rename(columns=name_mapping, inplace=True)

    for col in X.columns:
        if isinstance(X[col].dtype, CategoricalDtype) :
            X[col] = X[col].astype(float)
    
    y = y.astype(int).map({1: "not_merged", 2: "merged"})
    return Dataset.from_Xy(
        X=X,
        y=y,
        save_dir=data_output_dir,
        display_name="NOMAO",
        problem_type=ModelTask.BINARY_CLASSIFICATION,
        num_splits=num_splits,
        train_ratios=train_ratios,
        seed=seed,
        crop_dataset_to_size=max_dataset_size
    )


NAMES_RAW = """2 clean_name_intersect_min: continuous.
3 clean_name_intersect_max: continuous.
4 clean_name_levenshtein_sim: continuous.
5 clean_name_trigram_sim: continuous.
6 clean_name_levenshtein_term: continuous.
7 clean_name_trigram_term: continuous.
8 clean_name_including: n,s,m.
9 clean_name_equality: n,s,m.
10 city_intersect_min: continuous.
11 city_intersect_max: continuous.
12 city_levenshtein_sim: continuous.
13 city_trigram_sim: continuous.
14 city_levenshtein_term: continuous.
15 city_trigram_term: continuous.
16 city_including: n,s,m.
17 city_equality: n,s,m.
18 zip_intersect_min: continuous.
19 zip_intersect_max: continuous.
20 zip_levenshtein_sim: continuous.
21 zip_trigram_sim: continuous.
22 zip_levenshtein_term: continuous.
23 zip_trigram_term: continuous.
24 zip_including: n,s,m.
25 zip_equality: n,s,m.
26 street_intersect_min: continuous.
27 street_intersect_max: continuous.
28 street_levenshtein_sim: continuous.
29 street_trigram_sim: continuous.
30 street_levenshtein_term: continuous.
31 street_trigram_term: continuous.
32 street_including: n,s,m.
33 street_equality: n,s,m.
34 website_intersect_min: continuous.
35 website_intersect_max: continuous.
36 website_levenshtein_sim: continuous.
37 website_trigram_sim: continuous.
38 website_levenshtein_term: continuous.
39 website_trigram_term: continuous.
40 website_including: n,s,m.
41 website_equality: n,s,m.
42 countryname_intersect_min: continuous.
43 countryname_intersect_max: continuous.
44 countryname_levenshtein_sim: continuous.
45 countryname_trigram_sim: continuous.
46 countryname_levenshtein_term: continuous.
47 countryname_trigram_term: continuous.
48 countryname_including: n,s,m.
49 countryname_equality: n,s,m.
50 geocoderlocalityname_intersect_min: continuous.
51 geocoderlocalityname_intersect_max: continuous.
52 geocoderlocalityname_levenshtein_sim: continuous.
53 geocoderlocalityname_trigram_sim: continuous.
54 geocoderlocalityname_levenshtein_term: continuous.
55 geocoderlocalityname_trigram_term: continuous.
56 geocoderlocalityname_including: n,s,m.
57 geocoderlocalityname_equality: n,s,m.
58 geocoderinputaddress_intersect_min: continuous.
59 geocoderinputaddress_intersect_max: continuous.
60 geocoderinputaddress_levenshtein_sim: continuous.
61 geocoderinputaddress_trigram_sim: continuous.
62 geocoderinputaddress_levenshtein_term: continuous.
63 geocoderinputaddress_trigram_term: continuous.
64 geocoderinputaddress_including: n,s,m.
65 geocoderinputaddress_equality: n,s,m.
66 geocoderoutputaddress_intersect_min: continuous.
67 geocoderoutputaddress_intersect_max: continuous.
68 geocoderoutputaddress_levenshtein_sim: continuous.
69 geocoderoutputaddress_trigram_sim: continuous.
70 geocoderoutputaddress_levenshtein_term: continuous.
71 geocoderoutputaddress_trigram_term: continuous.
72 geocoderoutputaddress_including: n,s,m.
73 geocoderoutputaddress_equality: n,s,m.
74 geocoderpostalcodenumber_intersect_min: continuous.
75 geocoderpostalcodenumber_intersect_max: continuous.
76 geocoderpostalcodenumber_levenshtein_sim: continuous.
77 geocoderpostalcodenumber_trigram_sim: continuous.
78 geocoderpostalcodenumber_levenshtein_term: continuous.
79 geocoderpostalcodenumber_trigram_term: continuous.
80 geocoderpostalcodenumber_including: n,s,m.
81 geocoderpostalcodenumber_equality: n,s,m.
82 geocodercountrynamecode_intersect_min: continuous.
83 geocodercountrynamecode_intersect_max: continuous.
84 geocodercountrynamecode_levenshtein_sim: continuous.
85 geocodercountrynamecode_trigram_sim: continuous.
86 geocodercountrynamecode_levenshtein_term: continuous.
87 geocodercountrynamecode_trigram_term: continuous.
88 geocodercountrynamecode_including: n,s,m.
89 geocodercountrynamecode_equality: n,s,m.
90 phone_diff: continuous.
91 phone_levenshtein: continuous.
92 phone_trigram: continuous.
93 phone_equality: n,s,m.
94 fax_diff: continuous.
95 fax_levenshtein: continuous.
96 fax_trigram: continuous.
97 fax_equality: n,s,m.
98 street_number_diff: continuous.
99 street_number_levenshtein: continuous.
100 street_number_trigram: continuous.
101 street_number_equality: n,s,m.
102 geocode_coordinates_long_diff: continuous.
103 geocode_coordinates_long_levenshtein: continuous.
104 geocode_coordinates_long_trigram: continuous.
105 geocode_coordinates_long_equality: n,s,m.
106 geocode_coordinates_lat_diff: continuous.
107 geocode_coordinates_lat_levenshtein: continuous.
108 geocode_coordinates_lat_trigram: continuous.
109 geocode_coordinates_lat_equality: n,s,m.
110 coordinates_long_diff: continuous.
111 coordinates_long_levenshtein: continuous.
112 coordinates_long_trigram: continuous.
113 coordinates_long_equality: n,s,m.
114 coordinates_lat_diff: continuous.
115 coordinates_lat_levenshtein: continuous.
116 coordinates_lat_trigram: continuous.
117 coordinates_lat_equality: n,s,m.
118 geocode_coordinates_diff: continuous.
119 coordinates_diff: continuous.
120 label: +1,-1.
"""