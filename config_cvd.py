#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared configuration for CVD event definitions and feature schemas.
"""
from typing import Dict

# Event variables provided by clinicians
# Dict values: "cat" for categorical/binary, "num" for continuous.
# We OR these into a single binary target; includes baseline and since-last-visit events.
# 如果你后面要把 classification 和 risk prediction 明确区分，建议在 label 构建函数里拆成两套 label（baseline-only vs future-only）。
CVD_EVENT_COLS: Dict[str, str] = {
    "CoronaryArtery": "cat",       # Coronary artery disease diagnosis
    "coronaryartery_slv": "cat",   # Coronary artery disease since last visit
    "CABG": "cat",                 # Coronary artery bypass surgery
    "cabg_slv": "cat",             # CABG since last visit
    "Angioplasty": "cat",          # Angioplasty or cardiac stents placed
    "angioplasty_slv": "cat",      # Angioplasty or cardiac stents placed since last visit
    "PeriphVascular": "cat",       # Peripheral vascular disease
    "periphvascular_slv": "cat",   # Peripheral vascular disease since last visit
    "HeartAttack": "cat",          # Myocardial infarction (heart attack)
    "heartattack_slv": "cat",      # Heart attack since last visit
    "Stroke": "cat",               # Stroke
    "stroke_slv": "cat",           # Stroke since last visit
    "TIA": "cat",                  # Transient ischemic attack
    "tia_slv": "cat",              # TIA since last visit
    "CongestHeartFail": "cat",     # Congestive heart failure
    "congestheartfail_slv": "cat", # Congestive heart failure since last visit
}


# 以下定义都没用了

# Fixed feature set (clinical only; adjust as needed)
# Core demographic/vitals/metabolic covariates for CVD risk.
CLINICAL_FIXED_FEATURES: Dict[str, str] = {
    "age_visit": "num",       # Age at visit (years)
    "gender": "cat",          # Sex (coded)
    "race": "cat",            # Race (coded)
    "BMI": "num",             # Body mass index (kg/m^2)
    "sysBP": "num",           # Systolic blood pressure
    "diasBP": "num",          # Diastolic blood pressure
    "HR": "num",              # Resting heart rate
    "smoking_status": "cat",  # Smoking status (coded)
    "Diabetes": "cat",        # Diabetes diagnosis flag
}

# Additional clinical candidates (present in merged table)
CLINICAL_CANDIDATES: Dict[str, str] = {
    "ATS_PackYears": "num",         # Pack-years (ATS)
    "EverSmokedCig": "cat",         # Ever smoked cigarettes
    "CigsPerDay_Fagerstrom": "num", # Cigarettes/day (Fagerstrom)
    "SmokCigNow": "cat",            # Currently smoke cigarettes
    "CigPerDaySmokNow": "num",      # Cig/day currently
    "CigPerDaySmokLast5Years": "num", # Cig/day in last 5 years
    "currmedhighcholesterol": "cat",  # On medication for high cholesterol
    "HighCholest": "cat",             # High cholesterol diagnosis
    "currmedhighbp": "cat",           # On medication for high blood pressure
    "Waist_CM": "num",                # Waist circumference (cm)
}

# Combined clinical feature set (fixed + candidates)
CLINICAL_FEATURES: Dict[str, str] = {**CLINICAL_FIXED_FEATURES, **CLINICAL_CANDIDATES}

# CAC features (if available in the dataset)
CAC_FEATURES: Dict[str, str] = {
    "LM_Lesions": "num",      # Number of calcified lesions in the left main coronary artery
    "LM_Score": "num",        # Agatston calcium score for the left main coronary artery
    "LM_Volumes": "num",      # Calcium volume (mm^3) in the left main coronary artery
    "LAD_Lesions": "num",     # Number of calcified lesions in the left anterior descending artery
    "LAD_Score": "num",       # Agatston calcium score for the left anterior descending artery
    "LAD_Volumes": "num",     # Calcium volume (mm^3) in the left anterior descending artery
    "LCx_Lesions": "num",     # Number of calcified lesions in the left circumflex artery
    "LCx_Score": "num",       # Agatston calcium score for the left circumflex artery
    "LCx_Volumes": "num",     # Calcium volume (mm^3) in the left circumflex artery
    "RCA_Lesion": "num",      # Number of calcified lesions in the right coronary artery
    "RCA_Score": "num",       # Agatston calcium score for the right coronary artery
    "RCA_Volumes": "num",     # Calcium volume (mm^3) in the right coronary artery
    "Total_Score": "num",     # Total Agatston calcium score across all coronary arteries
    "Total_Lesions": "num",   # Total number of calcified lesions across all coronary arteries
    "Total_Volume": "num",    # Total calcium volume (mm^3) across all coronary arteries
}

# Combined clinical + CAC feature pool
CLINICAL_CAC_FEATURES: Dict[str, str] = {**CLINICAL_FEATURES, **CAC_FEATURES}
