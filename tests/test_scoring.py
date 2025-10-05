import pandas as pd
from pathlib import Path
import pytest

from scoring import score_survey

# Path to the CSV fixture (adjust path if necessary)
FIXTURE = (
    Path(__file__)            # e.g. .../tests/test_scoring.py
    .parent                    # .../tests
    .parent                    # .../<root>
    / "data"                   # .../<root>/data
    / "PORTRAIT_test_updated.csv"
)
# Expected numeric results for each user
EXPECTED = {
    "Miltos2": {
        "phq_total": 18,
        "oci_total": 54,
        "bai_total": 63,
        "stai_total": 33,
        "BFI_Extraversion": 3.25,
        "BFI_Agreeableness": 3.11,
        "BFI_Conscientiousness": 3.11,
        "BFI_Neuroticism": 3.25,
        "BFI_Openness": 3.6,
        # ASSIST scores
        "Tabaco (cigarrillos, tabaco de mascar, puros, etc.)": 12,
        "Bebidas alcohólicas (cerveza, vinos, licores, etc.)": 0,
        "Cannabis (marihuana, mota, hierba, hachís, etc.)": 0,
        "Estimulantes tipo anfetamina (speed, anfetaminas, éxtasis, etc.)": 0,
        "Inhalantes (óxido nitroso, pegamento, gasolina, solvente para pintura, etc.)": 19,
        "Sedantes o pastillas para dormir (diazepam, alprazolam, flunitrazepam, midazolam, etc.)": 0,
        "Alucinógenos (LSD, ácidos, hongos, ketamina, etc.)": 0,
        "Opiáceos (heroína, morfina, metadona, buprenorfina, codeína, etc.)": 0,
        "Otro (especifique)": 20,
    },
    "MILT1": {
        "phq_total": 14,
        "oci_total": 38,
        "bai_total": 51,
        "stai_total": 26,
        "BFI_Extraversion": 2.25,
        "BFI_Agreeableness": 3.00,
        "BFI_Conscientiousness": 3.56,
        "BFI_Neuroticism": 3.88,
        "BFI_Openness": 3.20,
        # ASSIST scores
        "Tabaco (cigarrillos, tabaco de mascar, puros, etc.)": 0,
        "Bebidas alcohólicas (cerveza, vinos, licores, etc.)": 30,
        "Cannabis (marihuana, mota, hierba, hachís, etc.)": 0,
        "Estimulantes tipo anfetamina (speed, anfetaminas, éxtasis, etc.)": 0,
        "Inhalantes (óxido nitroso, pegamento, gasolina, solvente para pintura, etc.)": 0,
        "Sedantes o pastillas para dormir (diazepam, alprazolam, flunitrazepam, midazolam, etc.)": 18,
        "Alucinógenos (LSD, ácidos, hongos, ketamina, etc.)": 28,
        "Opiáceos (heroína, morfina, metadona, buprenorfina, codeína, etc.)": 0,
        "Otro (especifique)": 0,
    },
    "DIOG1": {
        "phq_total": 7,
        "oci_total": 39,
        "bai_total": 48,
        "stai_total": 27,
        "BFI_Extraversion": 2.75,
        "BFI_Agreeableness": 2.78,
        "BFI_Conscientiousness": 3.44,
        "BFI_Neuroticism": 3.12,
        "BFI_Openness": 2.80,
        # ASSIST scores
        "Tabaco (cigarrillos, tabaco de mascar, puros, etc.)": 0,
        "Bebidas alcohólicas (cerveza, vinos, licores, etc.)": 23,
        "Cannabis (marihuana, mota, hierba, hachís, etc.)": 0,
        "Estimulantes tipo anfetamina (speed, anfetaminas, éxtasis, etc.)": 0,
        "Inhalantes (óxido nitroso, pegamento, gasolina, solvente para pintura, etc.)": 12,
        "Sedantes o pastillas para dormir (diazepam, alprazolam, flunitrazepam, midazolam, etc.)": 0,
        "Alucinógenos (LSD, ácidos, hongos, ketamina, etc.)": 0,
        "Opiáceos (heroína, morfina, metadona, buprenorfina, codeína, etc.)": 0,
        "Otro (especifique)": 0,
    },
    "Ele01": {
        "phq_total": 0,
        "oci_total": 0,
        "bai_total": 21,
        "stai_total": 21,
        "BFI_Extraversion": 2.62,
        "BFI_Agreeableness": 2.78,
        "BFI_Conscientiousness": 2.78,
        "BFI_Neuroticism": 2.50,
        "BFI_Openness": 1.80,
        # ASSIST scores
        "Tabaco (cigarrillos, tabaco de mascar, puros, etc.)": 0,
        "Bebidas alcohólicas (cerveza, vinos, licores, etc.)": 0,
        "Cannabis (marihuana, mota, hierba, hachís, etc.)": 0,
        "Estimulantes tipo anfetamina (speed, anfetaminas, éxtasis, etc.)": 0,
        "Inhalantes (óxido nitroso, pegamento, gasolina, solvente para pintura, etc.)": 0,
        "Sedantes o pastillas para dormir (diazepam, alprazolam, flunitrazepam, midazolam, etc.)": 0,
        "Alucinógenos (LSD, ácidos, hongos, ketamina, etc.)": 0,
        "Opiáceos (heroína, morfina, metadona, buprenorfina, codeína, etc.)": 0,
        "Otro (especifique)": 0,
    },
    "alepiloto": {
        "phq_total": 0,
        "oci_total": 0,
        "bai_total": 21,
        "stai_total": 22,
        "BFI_Extraversion": 2.88,
        "BFI_Agreeableness": 3.22,
        "BFI_Conscientiousness": 2.89,
        "BFI_Neuroticism": 2.88,
        "BFI_Openness": 2.80,
        # ASSIST scores
        "Tabaco (cigarrillos, tabaco de mascar, puros, etc.)": 0,
        "Bebidas alcohólicas (cerveza, vinos, licores, etc.)": 14,
        "Cannabis (marihuana, mota, hierba, hachís, etc.)": 0,
        "Estimulantes tipo anfetamina (speed, anfetaminas, éxtasis, etc.)": 0,
        "Inhalantes (óxido nitroso, pegamento, gasolina, solvente para pintura, etc.)": 0,
        "Sedantes o pastillas para dormir (diazepam, alprazolam, flunitrazepam, midazolam, etc.)": 0,
        "Alucinógenos (LSD, ácidos, hongos, ketamina, etc.)": 0,
        "Opiáceos (heroína, morfina, metadona, buprenorfina, codeína, etc.)": 0,
        "Otro (especifique)": 0,
    },
    "OCDM": {
        "phq_total":   0,
        "oci_total":   13,
        "bai_total":   21,
        "stai_total":  31,
        "BFI_Extraversion": 2.75,
        "BFI_Agreeableness": 2.89,
        "BFI_Conscientiousness": 2.89,
        "BFI_Neuroticism": 2.75,
        "BFI_Openness": 2.40,
        "Tabaco (cigarrillos, tabaco de mascar, puros, etc.)": 0,
        "Bebidas alcohólicas (cerveza, vinos, licores, etc.)": 0,
        "Cannabis (marihuana, mota, hierba, hachís, etc.)": 0,
        "Estimulantes tipo anfetamina (speed, anfetaminas, éxtasis, etc.)": 0,
        "Inhalantes (óxido nitroso, pegamento, gasolina, solvente para pintura, etc.)": 0,
        "Sedantes o pastillas para dormir (diazepam, alprazolam, flunitrazepam, midazolam, etc.)": 0,
        "Alucinógenos (LSD, ácidos, hongos, ketamina, etc.)": 0,
        "Opiáceos (heroína, morfina, metadona, buprenorfina, codeína, etc.)": 0,
        "Otro (especifique)": 0,
    },
    "OCDF1": {
        "phq_total":   11,
        "oci_total":   13,
        "bai_total":   46,
        "stai_total":  23,
        "BFI_Extraversion": 2.88,
        "BFI_Agreeableness": 2.89,
        "BFI_Conscientiousness": 3.11,
        "BFI_Neuroticism": 2.75,
        "BFI_Openness": 3.20,
        "Tabaco (cigarrillos, tabaco de mascar, puros, etc.)": 27,
        "Bebidas alcohólicas (cerveza, vinos, licores, etc.)": 0,
        "Cannabis (marihuana, mota, hierba, hachís, etc.)": 0,
        "Estimulantes tipo anfetamina (speed, anfetaminas, éxtasis, etc.)": 0,
        "Inhalantes (óxido nitroso, pegamento, gasolina, solvente para pintura, etc.)": 0,
        "Sedantes o pastillas para dormir (diazepam, alprazolam, flunitrazepam, midazolam, etc.)": 0,
        "Alucinógenos (LSD, ácidos, hongos, ketamina, etc.)": 29,
        "Opiáceos (heroína, morfina, metadona, buprenorfina, codeína, etc.)": 0,
        "Otro (especifique)": 0,
    },
}

@pytest.fixture
def raw_df():
    return pd.read_csv(FIXTURE, header=None)


def test_compute_scores(raw_df):
    scores_df = score_survey(raw_df)
    errors = []
    for username, expected in EXPECTED.items():
        user_rows = scores_df[scores_df["username"] == username]
        if user_rows.empty:
            errors.append(f"Missing scores for {username}")
            continue
        row = user_rows.iloc[0]
        for col, exp_val in expected.items():
            if col not in row:
                errors.append(f"{username}: missing column {col}")
                continue
            actual = row[col]
            if isinstance(exp_val, float):
                if not pytest.approx(exp_val, rel=1e-3) == actual:
                    errors.append(f"{username}: {col} expected {exp_val}, got {actual}")
            else:
                if exp_val != actual:
                    errors.append(f"{username}: {col} expected {exp_val}, got {actual}")
    if errors:
        pytest.fail("Scoring mismatches:\n" + "\n".join(errors))