"""Feature metadata + rule-based clinical flags.

RULE: a threshold appears below ONLY if its number was read on a WHO / government source page
(see SOURCES). Every flag carries that source so the app can show it. Values we could not
verify were removed (see REMOVED at the bottom). Everything still needs clinician sign-off."""

SOURCES = {
    "WHO_PE": ("WHO pre-eclampsia fact sheet", "https://www.who.int/news-room/fact-sheets/detail/pre-eclampsia"),
    "NICE": ("NICE NG133 (UK) hypertension in pregnancy", "https://nice.org.uk/guidance/ng133/resources/intrapartum-care-and-place-of-care-pdf-8720711395"),
    "FL_MEWS": ("Florida Dept of Health maternal early warning table", "https://www.floridahealth.gov/statistics-and-data/PAMR/_documents/maternal-early-warning-system.pdf"),
    "WHO_HB24": ("WHO 2024 haemoglobin cutoffs guideline", "https://www.who.int/publications/i/item/9789240088542"),
    "WHO_GLU13": ("WHO 2013 hyperglycaemia in pregnancy guideline", "https://www.paho.org/hq/dmdocuments/2015/guide-who-hyperglycaemia-pregnancy.pdf"),
    "NIDDK": ("NIDDK (NIH) A1C test", "https://www.niddk.nih.gov/health-information/diagnostic-tests/a1c-test"),
    "MEDLINE": ("MedlinePlus vital signs (NIH)", "https://medlineplus.gov/ency/article/002341.htm"),
}

# mod = a value that can be changed in a what-if test; ref = what-if target; ref_note = where the target comes from.
# min/max are only plausibility limits for input validation, not clinical claims.
FEATURES = [
    dict(name="Age", label="Age", unit="years", min=14, max=55, default=26, step=1, group="History", mod=False, hint="Mother's age"),
    dict(name="G", label="Gravida (G)", unit="", min=1, max=12, default=2, step=1, group="History", mod=False, hint="Total pregnancies incl. current"),
    dict(name="P", label="Para (P)", unit="", min=0, max=11, default=1, step=1, group="History", mod=False, hint="Previous births"),
    dict(name="L", label="Living children (L)", unit="", min=0, max=11, default=1, step=1, group="History", mod=False, hint="Children currently alive"),
    dict(name="A", label="Abortions / miscarriages (A)", unit="", min=0, max=8, default=0, step=1, group="History", mod=False, hint="Previous pregnancy losses"),
    dict(name="D", label="Child deaths (D)", unit="", min=0, max=5, default=0, step=1, group="History", mod=False, hint="Previous child deaths (assumed meaning of D)"),
    dict(name="SystolicBP", label="Systolic BP", unit="mmHg", min=70, max=220, default=112, step=1, group="Vitals", mod=True, ref=112, ref_note="112 is inside the MedlinePlus normal range 90/60 to 120/80"),
    dict(name="DiastolicBP", label="Diastolic BP", unit="mmHg", min=40, max=140, default=72, step=1, group="Vitals", mod=True, ref=72, ref_note="72 is inside the MedlinePlus normal range 90/60 to 120/80"),
    dict(name="HeartRate", label="Heart rate", unit="bpm", min=40, max=160, default=80, step=1, group="Vitals", mod=True, ref=80, ref_note="80 is inside the MedlinePlus adult resting range 60 to 100"),
    dict(name="RR", label="Respiratory rate", unit="/min", min=8, max=40, default=18, step=1, group="Vitals", mod=True, ref=18, ref_note="18 is the top of the MedlinePlus adult range 12 to 18"),
    dict(name="BodyTemp", label="Body temperature", unit="°F", min=95, max=106, default=98.6, step=0.1, group="Vitals", mod=True, ref=98.6, ref_note="98.6 is the MedlinePlus average"),
    dict(name="RBS", label="Random blood sugar", unit="mg/dL", min=50, max=450, default=105, step=1, group="Labs", mod=True, ref=110, ref_note="110 is a project choice: no verified normal value exists for random glucose"),
    dict(name="HB", label="Haemoglobin", unit="g/dL", min=4, max=17, default=11.5, step=0.1, group="Labs", mod=True, ref=12.0, ref_note="12.0 is above the WHO 2024 pregnancy anaemia cutoffs (11.0 and 10.5)"),
    dict(name="HBA1C", label="HbA1c", unit="%", min=3.5, max=14, default=5.2, step=0.1, group="Labs", mod=True, ref=5.2, ref_note="5.2 is below the NIDDK normal limit of 5.7"),
]
BY_NAME = {f["name"]: f for f in FEATURES}

DRIVER_GROUPS = {
    "Blood pressure": ["SystolicBP", "DiastolicBP"],
    "Blood sugar": ["RBS", "HBA1C"],
    "Haemoglobin": ["HB"],
    "Temperature": ["BodyTemp"],
    "Heart rate": ["HeartRate"],
    "Breathing rate": ["RR"],
    "Maternal age": ["Age"],
    "Obstetric history": ["G", "P", "L", "A", "D"],
}
FEATURE_TO_GROUP = {f: g for g, fs in DRIVER_GROUPS.items() for f in fs}

GROUP_QUERY = {
    "Blood pressure": "high blood pressure hypertension preeclampsia pregnancy monitoring warning signs",
    "Blood sugar": "high blood sugar gestational diabetes diet exercise glucose monitoring HbA1c",
    "Haemoglobin": "anaemia low haemoglobin iron folic acid diet pregnancy",
    "Temperature": "fever infection pregnancy temperature",
    "Heart rate": "fast heart rate tachycardia pregnancy",
    "Breathing rate": "fast breathing respiratory rate breathlessness pregnancy",
    "Maternal age": "maternal age teenage adolescent advanced maternal age antenatal care",
    "Obstetric history": "previous miscarriage child death many pregnancies birth spacing delivery planning",
}

LEVEL_RANK = {"info": 0, "borderline": 1, "high": 2, "critical": 3}


def _f(group, feature, level, text, target, src, note=""):
    name, url = SOURCES[src]
    return dict(group=group, feature=feature, level=level, text=text, target=target,
                source=src, source_name=name, source_url=url, note=note)


def flags_for(v, trimester=None):
    """Rule-based flags from raw inputs, independent of the ML model.
    trimester (1, 2, 3 or None) is optional and used ONLY for the anaemia rule."""
    out = []
    sb, db, rbs, a1c, hb = v["SystolicBP"], v["DiastolicBP"], v["RBS"], v["HBA1C"], v["HB"]
    hr, rr = v["HeartRate"], v["RR"]

    # Blood pressure
    if sb >= 160 or db >= 110:
        out.append(_f("Blood pressure", "SystolicBP" if sb >= 160 else "DiastolicBP", "critical",
                      f"Severe-range blood pressure ({sb:.0f}/{db:.0f} mmHg)", "below 140/90", "NICE",
                      "NICE: 160/110 or more is severe hypertension."))
    elif sb >= 140 or db >= 90:
        out.append(_f("Blood pressure", "SystolicBP" if sb >= 140 else "DiastolicBP", "high",
                      f"Raised blood pressure ({sb:.0f}/{db:.0f} mmHg)", "below 140/90", "WHO_PE",
                      "WHO: 140/90 or more; pre-eclampsia also needs protein in the urine after 20 weeks."))
    elif sb <= 80:
        out.append(_f("Blood pressure", "SystolicBP", "high", f"Very low systolic pressure ({sb:.0f} mmHg)", "90 or above", "FL_MEWS",
                      "Top-urgency low trigger in the early warning table."))
    elif sb < 90:
        out.append(_f("Blood pressure", "SystolicBP", "borderline", f"Low systolic pressure ({sb:.0f} mmHg)", "90 or above", "FL_MEWS",
                      "Intermediate low trigger in the early warning table."))

    # Blood sugar
    if rbs >= 200:
        out.append(_f("Blood sugar", "RBS", "high", f"Random blood sugar in the diabetic range ({rbs:.0f} mg/dL)",
                      "confirm with a fasting or glucose tolerance test", "WHO_GLU13",
                      "WHO applies 200 mg/dL (11.1 mmol/L) with diabetes symptoms; this app has no symptom input."))
    if a1c >= 6.5:
        out.append(_f("Blood sugar", "HBA1C", "high", f"HbA1c in the diabetic range ({a1c:.1f}%)", "below 5.7", "NIDDK",
                      "Confirm with a repeat test. Not a test for gestational diabetes; can read falsely high in iron-deficiency anaemia."))
    elif a1c >= 5.7:
        out.append(_f("Blood sugar", "HBA1C", "borderline", f"HbA1c in the pre-diabetic range ({a1c:.1f}%)", "below 5.7", "NIDDK",
                      "Not a test for gestational diabetes; can read falsely high in iron-deficiency anaemia."))

    # Haemoglobin (WHO 2024; first/third trimester values are used when trimester is unknown)
    if trimester == 2:
        cut, mild_lo, mod_lo, basis = 10.5, 9.5, 7.0, "second trimester"
    else:
        cut, mild_lo, mod_lo, basis = 11.0, 10.0, 7.0, "first/third trimester" if trimester in (1, 3) else "trimester not given: first/third values used"
    if hb < 7.0:
        out.append(_f("Haemoglobin", "HB", "critical", f"Severe anaemia (Hb {hb:.1f} g/dL)", f"{cut:.1f} or above", "WHO_HB24", f"WHO 2024, {basis}."))
    elif hb < mild_lo:  # moderate: 7.0 up to the mild band (WHO 2024)
        out.append(_f("Haemoglobin", "HB", "high", f"Moderate anaemia (Hb {hb:.1f} g/dL)", f"{cut:.1f} or above", "WHO_HB24", f"WHO 2024, {basis}."))
    elif hb < cut:
        out.append(_f("Haemoglobin", "HB", "borderline", f"Mild anaemia (Hb {hb:.1f} g/dL)", f"{cut:.1f} or above", "WHO_HB24", f"WHO 2024, {basis}."))

    # Heart rate and breathing rate (early warning table; "or more" boundaries as printed)
    if hr >= 120:
        out.append(_f("Heart rate", "HeartRate", "critical", f"Very fast heart rate ({hr:.0f} bpm)", "60 to 100 (adult resting range)", "FL_MEWS", "Top-urgency trigger is 120 or more."))
    elif hr >= 111:
        out.append(_f("Heart rate", "HeartRate", "high", f"Fast heart rate ({hr:.0f} bpm)", "60 to 100 (adult resting range)", "FL_MEWS", "Intermediate trigger is 111 to 119."))
    if rr >= 30:
        out.append(_f("Breathing rate", "RR", "critical", f"Very fast breathing ({rr:.0f}/min)", "12 to 18 (adult resting range)", "FL_MEWS", "Top-urgency trigger is 30 or more."))
    elif rr >= 22:
        out.append(_f("Breathing rate", "RR", "high", f"Fast breathing ({rr:.0f}/min)", "12 to 18 (adult resting range)", "FL_MEWS", "Intermediate trigger is 22 to 29."))
    elif rr <= 10:
        out.append(_f("Breathing rate", "RR", "high", f"Slow breathing ({rr:.0f}/min)", "12 to 18 (adult resting range)", "FL_MEWS", "Trigger is 10 or fewer."))
    return out


# REMOVED because no WHO/government source was opened for them. Re-add only with a source:
#   random sugar 140-199 "raised" and below 70 "low"; fever 100.4 F and 102.2 F; heart rate above 100;
#   age under 18 and 35 or over; five or more pregnancies/births; three or more losses; any child death.
