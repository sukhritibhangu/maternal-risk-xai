"""Feature metadata and source-linked maternal screening flags.

The numeric min/max values below are application input bounds only. They are not
clinical reference intervals. Rule flags use WHO or Government of India sources
and retain the source URL so the interface can show the basis and limitations.
"""

SOURCES = {
    "GOI_PMSMA": (
        "MoHFW/PMSMA high-risk pregnancy checklist",
        "https://pmsma.mohfw.gov.in/wp-content/uploads/2016/10/High-Risk-Conditions-in-preg-modified-Final.pdf",
    ),
    "WHO_PE": (
        "WHO pre-eclampsia fact sheet",
        "https://www.who.int/news-room/fact-sheets/detail/pre-eclampsia",
    ),
    "GOI_SAFE_CHILDBIRTH": (
        "NHM Safe Childbirth Checklist",
        "https://www.nhm.gov.in/images/pdf/Dakshata_Implementation/GoI_Recommended_Casesheets_LR_Register_Reporting_Tool/Case-sheet-L3.pdf",
    ),
    "GOI_DAKSHATA": (
        "NHM DAKSHATA mentoring and support visit 1",
        "https://www.nhm.gov.in/images/pdf/Dakshata_Implementation/Mentoring_Obstetric_Drills_Package/MSV_1.pdf",
    ),
    "GOI_MATERNAL_NEAR_MISS": (
        "NHM Maternal Near Miss Operational Guidelines",
        "https://www.nhm.gov.in/images/pdf/programmes/maternal-health/guidelines/Maternal_Near_Miss_Operational_Guidelines.pdf",
    ),
    "WHO_DIABETES": (
        "WHO diabetes diagnostic criteria",
        "https://www.who.int/news-room/questions-and-answers/item/diabetes",
    ),
    "GOI_GDM": (
        "NHM gestational diabetes technical and operational guidelines",
        "https://nhm.gov.in/New_Updates_2018/NHM_Components/RMNCH_MH_Guidelines/Gestational-Diabetes-Mellitus.pdf",
    ),
    "WHO_HB24": (
        "WHO 2024 haemoglobin cutoffs guideline",
        "https://www.who.int/publications/i/item/9789240088542",
    ),
    "WHO_GDM25": (
        "WHO recommendations on care for women with diabetes during pregnancy (2025)",
        "https://www.who.int/publications/i/item/9789240117044",
    ),
}

# ``min``/``max`` are form validation bounds. ``ref`` is only a model
# sensitivity-analysis scenario value, not a clinical target.
SCENARIO_NOTE = "Project sensitivity-analysis value only; not a clinical target."

FEATURES = [
    dict(name="Age", label="Age", unit="years", min=14, max=55, default=26, step=1, group="History", mod=False, hint="Mother's age"),
    dict(name="G", label="Gravida (G)", unit="", min=1, max=12, default=2, step=1, group="History", mod=False, hint="Total pregnancies including the current pregnancy"),
    dict(name="P", label="Para (P)", unit="", min=0, max=11, default=1, step=1, group="History", mod=False, hint="Previous pregnancies ending in delivery, counted as pregnancy events"),
    dict(name="L", label="Living children (L)", unit="", min=0, max=11, default=1, step=1, group="History", mod=False, hint="Children currently alive"),
    dict(name="A", label="Pregnancy losses (A)", unit="", min=0, max=8, default=0, step=1, group="History", mod=False, hint="Previous pregnancy losses per the project's confirmed definition"),
    dict(name="D", label="Child deaths (D)", unit="", min=0, max=5, default=0, step=1, group="History", mod=False, hint="Children who died, per the project's confirmed definition"),
    dict(name="SystolicBP", label="Systolic BP", unit="mmHg", min=70, max=220, default=112, step=1, group="Vitals", mod=True, ref=112, ref_note=SCENARIO_NOTE),
    dict(name="DiastolicBP", label="Diastolic BP", unit="mmHg", min=40, max=140, default=72, step=1, group="Vitals", mod=True, ref=72, ref_note=SCENARIO_NOTE),
    dict(name="HeartRate", label="Heart rate", unit="bpm", min=40, max=160, default=80, step=1, group="Vitals", mod=True, ref=80, ref_note=SCENARIO_NOTE),
    dict(name="RR", label="Respiratory rate", unit="/min", min=8, max=40, default=18, step=1, group="Vitals", mod=True, ref=18, ref_note=SCENARIO_NOTE),
    dict(name="BodyTemp", label="Body temperature", unit="°F", min=95, max=106, default=98.6, step=0.1, group="Vitals", mod=True, ref=98.6, ref_note=SCENARIO_NOTE),
    dict(name="RBS", label="Random blood sugar", unit="mg/dL", min=50, max=450, default=105, step=1, group="Labs", mod=True, ref=110, ref_note=SCENARIO_NOTE),
    dict(name="HB", label="Haemoglobin", unit="g/dL", min=4, max=17, default=11.5, step=0.1, group="Labs", mod=True, ref=12.0, ref_note=SCENARIO_NOTE),
    dict(name="HBA1C", label="HbA1c", unit="%", min=3.5, max=14, default=5.2, step=0.1, group="Labs", mod=True, ref=5.2, ref_note=SCENARIO_NOTE),
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
    "Blood sugar": "high blood sugar gestational diabetes glucose testing HbA1c",
    "Haemoglobin": "anaemia low haemoglobin pregnancy",
    "Temperature": "fever infection pregnancy temperature labour childbirth",
    "Heart rate": "fast heart rate tachycardia pregnancy maternal near miss",
    "Breathing rate": "fast breathing respiratory rate pregnancy maternal near miss",
    "Maternal age": "maternal age young primigravida elderly gravida antenatal care",
    "Obstetric history": "gravida parity living children pregnancy losses child deaths",
}

LEVEL_RANK = {"info": 0, "borderline": 1, "high": 2, "critical": 3}


def _f(group, feature, level, text, target, src, note=""):
    name, url = SOURCES[src]
    return dict(group=group, feature=feature, level=level, text=text, target=target,
                source=src, source_name=name, source_url=url, note=note)


def flags_for(v, trimester=None, intrapartum=False):
    """Build source-linked screening flags from inputs.

    The optional ``intrapartum`` context gates the Government of India fever
    criterion. ``trimester`` is used only for the WHO haemoglobin criteria.
    These flags are not diagnoses and do not replace repeat measurements or
    clinical assessment.
    """
    out = []
    sb, db = v["SystolicBP"], v["DiastolicBP"]
    rbs, hb = v["RBS"], v["HB"]
    hr, rr, temp = v["HeartRate"], v["RR"], v["BodyTemp"]
    age, gravida = v["Age"], v["G"]

    # MoHFW/PMSMA age criteria: young primigravida <20; elderly gravida >35.
    if age < 20 and gravida == 1:
        out.append(_f("Maternal age", "Age", "info",
                      f"Age-related high-risk screening criterion: young primigravida under 20 ({age:.0f} years).",
                      "Antenatal clinician review", "GOI_PMSMA",
                      "The under-20 criterion applies only when this is the first pregnancy."))
    if age > 35:
        out.append(_f("Maternal age", "Age", "info",
                      f"Age-related high-risk screening criterion: gravida over 35 ({age:.0f} years).",
                      "Antenatal clinician review", "GOI_PMSMA",
                      "The source wording is more than 35 years; exactly 35 does not meet it."))

    # Pregnancy blood pressure screening. The application has no repeat BP,
    # proteinuria, symptom or gestational-week input, so it never diagnoses PE.
    if sb >= 160 or db >= 110:
        out.append(_f("Blood pressure", "SystolicBP" if sb >= 160 else "DiastolicBP", "critical",
                      f"Severe-range blood pressure reading ({sb:.0f}/{db:.0f} mmHg).",
                      "Prompt in-person clinical assessment", "GOI_SAFE_CHILDBIRTH",
                      "A Government of India checklist uses these values in a referral context. This app does not collect repeat readings, proteinuria, symptoms or gestational age and cannot diagnose pre-eclampsia."))
    elif sb >= 140 or db >= 90:
        out.append(_f("Blood pressure", "SystolicBP" if sb >= 140 else "DiastolicBP", "high",
                      f"Blood pressure reading at or above the pregnancy hypertension threshold ({sb:.0f}/{db:.0f} mmHg).",
                      "Repeat measurement and antenatal assessment", "GOI_PMSMA",
                      "The Government of India checklist defines hypertension using two consecutive readings. WHO pre-eclampsia diagnosis also requires clinical context, including proteinuria after 20 weeks."))
    elif sb < 90:
        out.append(_f("Blood pressure", "SystolicBP", "high",
                      f"Low systolic blood pressure reading ({sb:.0f} mmHg).",
                      "Clinical assessment, especially if unwell", "GOI_DAKSHATA",
                      "A Government of India training document lists systolic BP below 90 mmHg as a hypotension/shock screening sign; this reading alone does not diagnose shock."))

    # RBS and OGTT are different measurements. This general WHO diabetes
    # criterion needs symptoms or a hyperglycaemic crisis; the app has no
    # symptom field, so it is an information flag only and not a GDM diagnosis.
    if rbs >= 200:
        out.append(_f("Blood sugar", "RBS", "info",
                      f"Random plasma glucose is at or above 200 mg/dL ({rbs:.0f} mg/dL).",
                      "Clinical confirmation and symptom review", "WHO_DIABETES",
                      "WHO's random-glucose diabetes criterion requires classic symptoms or hyperglycaemic crisis. This app has no symptom field. India's GDM criterion is a timed 2-hour result after a 75 g oral glucose load, not this random reading."))

    # WHO 2024 trimester-specific anaemia thresholds. If trimester is absent,
    # use the shared first/third trimester cutoff and disclose that assumption.
    if trimester == 2:
        cut, mild_lo, basis = 10.5, 9.5, "second trimester"
    else:
        cut, mild_lo = 11.0, 10.0
        basis = "first/third trimester" if trimester in (1, 3) else "trimester not given; first/third cutoff used"
    if hb < 7.0:
        out.append(_f("Haemoglobin", "HB", "critical", f"Severe anaemia threshold met (Hb {hb:.1f} g/dL).",
                      f"Hb {cut:.1f} g/dL or higher", "WHO_HB24", f"WHO 2024, {basis}."))
    elif hb < mild_lo:
        out.append(_f("Haemoglobin", "HB", "high", f"Moderate anaemia range (Hb {hb:.1f} g/dL).",
                      f"Hb {cut:.1f} g/dL or higher", "WHO_HB24", f"WHO 2024, {basis}."))
    elif hb < cut:
        out.append(_f("Haemoglobin", "HB", "borderline", f"Mild anaemia range (Hb {hb:.1f} g/dL).",
                      f"Hb {cut:.1f} g/dL or higher", "WHO_HB24", f"WHO 2024, {basis}."))

    # The GoI source includes these values in specified maternal morbidity
    # contexts, not as universal normal ranges or stand-alone diagnosis bands.
    if hr > 120:
        out.append(_f("Heart rate", "HeartRate", "info",
                      f"Heart rate above 120 beats/min ({hr:.0f} bpm).",
                      "Clinical review in context", "GOI_MATERNAL_NEAR_MISS",
                      "The GoI value appears as a tachycardia sign in specified morbidity contexts; no universal pregnancy range is implied."))
    if rr > 20:
        out.append(_f("Breathing rate", "RR", "info",
                      f"Respiratory rate above 20 breaths/min ({rr:.0f}/min).",
                      "Clinical review in context", "GOI_MATERNAL_NEAR_MISS",
                      "The GoI value appears as a tachypnea sign in specified morbidity contexts; no universal pregnancy range is implied."))

    # This Government of India fever trigger is specific to labour/childbirth.
    if intrapartum and temp >= 100.4:
        out.append(_f("Temperature", "BodyTemp", "high",
                      f"Maternal temperature at or above 100.4 °F ({temp:.1f} °F).",
                      "Intrapartum clinical assessment", "GOI_SAFE_CHILDBIRTH",
                      "The >=38 °C fever criterion is from a labour/childbirth checklist and is shown only when that context is selected."))

    # WHO 2025 advises against routine HbA1c measurement for women with GDM.
    # No pregnancy-specific HbA1c alert is generated here.
    return out
