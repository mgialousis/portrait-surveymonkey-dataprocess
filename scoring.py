#!/usr/bin/env python
# coding: utf-8
"""
scoring.py

Refactored for use with test_scoring.py. Provides the `score_survey` function that
loads an input CSV, processes all instruments, and returns a result DataFrame.
Optionally writes to CSV if `output_file` is provided.
"""

import pandas as pd
import numpy as np


def find_column_index(hrow, target_value):
    """
    Finds the index of target_value in the header_row.
    Raises a ValueError if the target is not found.
    """
    try:
        return hrow.tolist().index(target_value)
    except ValueError:
        raise ValueError(f"Header value '{target_value}' was not found in the DataFrame header.")


def score_survey(df, output_csv=None):
    """
    Load the survey CSV (with no header row), process all instruments (PHQ, BAI,
    OCI, STAI, BFI, ASSIST), merge the ASSIST scores, compute totals and
    classifications, and return a DataFrame with selected result columns.
    If output_csv is provided, save the result to that path.
    """
    # 1. Read input
    header_row      = df.iloc[0]
    subheader_row   = df.iloc[1]
    subsubheader_row= df.iloc[2]

    # 2. Locate key indices
    user_idx = find_column_index(subheader_row, "Código de usuario")
    sex_idx  = find_column_index(subheader_row, "¿Con qué género se identifica más usted? (Selecciona la opción que más te identifique)")

    # 3. Find start/end for each instrument (reuse original labels)
    # PHQ
    PHQ_HEADER = (
        "Durante las últimas dos semanas, ¿con qué frecuencia ha tenido molestias "
        "debido a los siguientes problemas?"
    )
    PHQ_LAST = (
        "Si ha marcado cualquiera de los problemas, ¿Qué tanta dificultad le han "
        "dado estos problemas para hacer su trabajo, encargarse de las tareas "
        "del hogar, o llevarse bien con otras personas?"
    )
    phq_start = find_column_index(subheader_row, PHQ_HEADER)
    phq_end   = find_column_index(subheader_row, PHQ_LAST)

    # BAI
    BAI_HEADER = (
        "En el cuestionario hay una lista de síntomas comunes de la ansiedad. Lea "
        "cada uno de los ítems atentamente, e indique cuanto le ha afectado en la "
        "última semana incluyendo hoy:"
    )
    BAI_LAST_SUB = "Con sudores, fríos o calientes."
    bai_start = find_column_index(subheader_row, BAI_HEADER)
    bai_end   = find_column_index(subsubheader_row, BAI_LAST_SUB)

    # OCI
    OCI_HEADER = (
        "Escoge la opción que mejor describe CUÁNTO malestar o molestia te ha "
        "producido esta experiencia durante el último mes."
    )
    OCI_LAST   = (
        "Tener con frecuencia pensamientos repugnantes y que le cueste librarse de ellos."
    )
    oci_start = find_column_index(subheader_row, OCI_HEADER)
    oci_end   = find_column_index(subsubheader_row, OCI_LAST)

    # STAI
    STAI_HEADER = (
        "Lea cada frase y señale la opción que indique mejor cómo se siente en "
        "general, en la mayoría de las ocasiones. No hay respuestas buenas ni malas. "
        "No emplee demasiado tiempo en cada frase y conteste señalando la respuesta "
        "que mejor describa cómo se siente usted generalmente."
    )
    STAI_LAST_SUB = (
        "Cuando pienso sobre asuntos y preocupaciones actuales me pongo tenso y agitado."
    )
    stai_start = find_column_index(subheader_row, STAI_HEADER)
    stai_end   = find_column_index(subsubheader_row, STAI_LAST_SUB)

    stai_reverse_items = [0, 5, 6, 9, 12, 15, 18]

    # BFI
    BFI_HEADER = (
        "Por favor, valore cada afirmación del cuestionario en una escala del 1 "
        "al 5, donde 1 significa \"Muy en desacuerdo\" y 5 \"Muy de acuerdo\"."
    )
    BFI_LAST = "Es sofisticado en arte, música o literatura."
    bfi_start = find_column_index(subheader_row, BFI_HEADER)
    bfi_end   = find_column_index(subsubheader_row, BFI_LAST)

    # define BFI subscales and reverse
    bfi_subscales = {
        "Extraversion":      [bfi_start + q-1 for q in [1,6,11,16,21,26,31,36]],
        "Agreeableness":     [bfi_start + q-1 for q in [2,7,12,17,22,27,32,37,42]],
        "Conscientiousness": [bfi_start + q-1 for q in [3,8,13,18,23,28,33,38,43]],
        "Neuroticism":       [bfi_start + q-1 for q in [4,9,14,19,24,29,34,39]],
        "Openness":          [bfi_start + q-1 for q in [5,10,15,20,25,30,35,40,41,44]],
    }
    bfi_reverse = {
        "Extraversion":      [6,21,31],
        "Agreeableness":     [2,12,27,37],
        "Conscientiousness": [8,18,23,43],
        "Neuroticism":       [9,24,34],
        "Openness":          [35,41],
    }

    # ASSIST
    ASSIST_HDR  = (
        "A lo largo de la vida, ¿cuál de las siguientes sustancias ha consumido alguna vez? "
        "(solo que consumió sin receta médica)"
    )
    ASSIST_LAST = "¿Alguna vez ha consumido alguna droga por vía inyectada? (solo las que consumió sin receta médica)"
    ASSIST_FIRST= "Ninguna de las anteriores"
    ASSIST_RESP = "Response"

    assist_start = find_column_index(subheader_row, ASSIST_HDR)
    assist_end   = find_column_index(subheader_row, ASSIST_LAST)
    first_subidx = assist_start + 1
    base_cols    = list(range(first_subidx, first_subidx+9))
    sub_names    = df.iloc[2, base_cols].tolist()
    assist_allowed = {
        2: {0,2,3,4,6}, 3: {0,3,4,5,6}, 4: {0,4,5,6,7},
        5: {0,5,6,7,8}, 6: {0,6,3}, 7: {0,6,3}
    }

    # build ASSIST scores
    results = []
    for r in range(3, df.shape[0]):
        username = df.iat[r, user_idx]
        rec = {"username": username}
        for c, name in zip(base_cols, sub_names):
            offsets = [9*i for i in range(1,7)]
            if name.startswith("Tabaco"): offsets.remove(36)
            total = sum(
                int(df.iat[r, c+off]) for off in offsets
                if pd.notna(df.iat[r, c+off])
            )
            rec[name] = total
        results.append(rec)
    assist_scores_df = pd.DataFrame(results)
    # assign risk
    low_max = {"Tabaco (cigarrillos, tabaco de mascar, puros, etc.)":3, "Alcohol":10}
    mod_max = 26
    for col in assist_scores_df.columns:
        if col=="username": continue
        lo = low_max.get(col,3)
        sc = assist_scores_df[col]
        masks = [sc<=lo, sc.between(lo+1, mod_max), sc>mod_max]
        lbls = ["No requiere intervención", "Recibir intervención breve", "Tratamiento más intensivo"]
        assist_scores_df[f"{col}_risk"] = np.select(masks, lbls, default="Unknown")

    # merge
    df = df.merge(assist_scores_df, left_on=df.columns[user_idx], right_on="username", how="left")

    # compute totals & classifications
    results_list = []
    for r in range(3, df.shape[0]):
        row = df.iloc[r]
        uname = row[user_idx]
        # PHQ total
        phq_vals = row[phq_start:phq_end].astype(float)
        phq_sum = int(phq_vals.sum())
        if phq_sum<5: phq_cls="1-minimal depression"
        elif phq_sum<10: phq_cls="2-mild depression"
        elif phq_sum<15: phq_cls="3-moderate depression"
        elif phq_sum<20: phq_cls="4-moderately severe depression"
        else: phq_cls="5-severe depression"
        # BAI total
        bai_vals = row[bai_start:bai_end+1].astype(float)
        bai_sum = int(bai_vals.sum())
        bai_cls = "low anxiety" if bai_sum<=21 else ("moderate anxiety" if bai_sum<=35 else "potentially concerning levels of anxiety")
        # OCI total
        oci_vals = row[oci_start:oci_end+1].astype(float)
        oci_sum = int(oci_vals.sum())
        oci_cls = "has OCD" if oci_sum>=21 else "does not have OCD"
        # STAI
        stai_vals = row[stai_start:stai_end+1].astype(float)
        for item in stai_reverse_items:
            if not pd.isna(stai_vals.iloc[item]):
                stai_vals.iloc[item] = 3 - stai_vals.iloc[item]
        stai_sum = int(stai_vals.sum())
        sex = row[sex_idx]
        if sex=="2" and stai_sum>=29: stai_cls="High"
        elif sex=="1" and stai_sum>=37: stai_cls="High"
        elif sex=="3": stai_cls="Binary"
        else: stai_cls="Normal"
        # BFI
        bfi_scores = {}
        for scale, cols in bfi_subscales.items():
            vals = []
            for c in cols:
                v = float(row[c])
                if (c - bfi_start + 1) in bfi_reverse.get(scale,[]):
                    v = 6 - v
                vals.append(v)
            bfi_scores[scale] = round(sum(vals)/len(vals),2)
        # assemble record
        rec = {"username": uname,
               "gender": sex,
               "phq_total": phq_sum, "phq_classification": phq_cls,
               "bai_total": bai_sum, "bai_classification": bai_cls,
               "oci_total": oci_sum, "oci_classification": oci_cls,
               "stai_total": stai_sum, "stai_classification": stai_cls,
        }
        # add BFI
        for sc, val in bfi_scores.items(): rec[f"BFI_{sc}"] = val
        # add ASSIST
        for sub in sub_names:
            rec[sub] = row[sub]
            rec[f"{sub}_risk"] = row[f"{sub}_risk"]
        results_list.append(rec)

    final_df = pd.DataFrame(results_list)

    # write if requested
    if output_csv:
        final_df.to_csv(output_csv, index=False)
    return final_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Score survey and output results CSV.")
    #parser.add_argument('input_csv', help='Path to input CSV with no headers')
    parser.add_argument('-o','--output', help='Path to write results CSV')
    args = parser.parse_args()
    filename_updated = "data/PORTRAIT_v3_updated"
    df_in = pd.read_csv(filename_updated + ".csv", header=None)
    df_out = score_survey(df_in, args.output)
    print(f"Processed {len(df_out)} respondents.")