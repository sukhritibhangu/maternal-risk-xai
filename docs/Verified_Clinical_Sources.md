# Clinical source register and verified criteria

This register maps the existing source-linked screening code to primary references reviewed on 4 November 2026. It documents a research prototype rather than a complete diagnostic protocol. Source review does not automatically verify documents in the running knowledge base.

| Area | Existing condition | Source and qualification |
| --- | --- | --- |
| Age | Under 20 with first pregnancy or over 35 | PMSMA [1]. Exactly 35 does not meet the latter condition. |
| Blood pressure | Systolic at least 140 or diastolic at least 90 mmHg | PMSMA [1] requires consecutive readings. The app has one reading. |
| Severe blood pressure | Systolic at least 160 or diastolic at least 110 mmHg | Childbirth checklist [2] uses a wider referral algorithm with proteinuria or symptoms. The app cannot reproduce that diagnostic algorithm. |
| Low systolic pressure | Below 90 mmHg | DAKSHATA MSV 1 triage table, printed page 9 [3]. A reading alone does not establish shock. |
| Heart and breathing rates | Heart rate over 120 beats per minute or respiratory rate over 20 per minute | Maternal Near Miss Guidelines, printed page 38 [4]. Contextual morbidity signs rather than universal normal ranges. |
| Random glucose | At least 200 mg/dL | WHO pregnancy assessment [5] requires diabetes symptoms too. The app lacks symptoms; this is a confirmation flag rather than a GDM diagnosis. |
| Fever | At least 100.4 degrees Fahrenheit when labour context is selected | Checklist [2] gives 38 degrees Celsius and prints 100.5 degrees Fahrenheit. The code uses the exact conversion of 100.4. This rounding difference is disclosed and unchanged. |
| HbA1c | No pregnancy-specific rule alert | WHO 2025 recommendation 9 [7] advises against routine HbA1c measurement in GDM. Pre-existing type 1 and type 2 diabetes have different guidance. |

WHO pre-eclampsia information [8] includes gestational timing and proteinuria. The app lacks the complete clinical inputs needed to establish that diagnosis.

## Trimester-specific haemoglobin

WHO 2024 tables 2 and 3 [6] supply these thresholds, converted from g/L to g/dL.

| Trimester | Mild | Moderate | Severe |
| --- | --- | --- | --- |
| First and third | At least 10 and below 11 | At least 7 and below 10 | Below 7 |
| Second | At least 9.5 and below 10.5 | At least 7 and below 9.5 | Below 7 |

Missing trimester uses first and third trimester thresholds with disclosure. The app does not implement every WHO measurement adjustment, including elevation and smoking. These bands are not treatment targets.

## Source review qualifications

Random glucose is not interchangeable with India's timed two-hour test after a 75 g glucose load [9]. The diabetes FAQ currently recorded in features.py no longer visibly provides the numerical random-glucose threshold; [5] provides additional primary support for the number and symptom qualifier. This document does not alter the code's source URL.

Official source locations were reviewed through publisher pages, full text or indexed official passages. Preserve the actual edition and passage during administrator review. Verification does not establish comprehensive clinical correctness.

## Primary references

1. [PMSMA high-risk pregnancy synopsis](https://pmsma.mohfw.gov.in/wp-content/uploads/2016/10/High-Risk-Conditions-in-preg-modified-Final.pdf).
2. [NHM Safe Childbirth Checklist in L3 case sheet](https://www.nhm.gov.in/images/pdf/Dakshata_Implementation/GoI_Recommended_Casesheets_LR_Register_Reporting_Tool/Case-sheet-L3.pdf).
3. [NHM DAKSHATA MSV 1](https://nhm.gov.in/images/pdf/Dakshata_Implementation/Mentoring_Obstetric_Drills_Package/MSV_1.pdf).
4. [NHM Maternal Near Miss Operational Guidelines December 2014](https://nhm.gov.in/images/pdf/programmes/maternal-health/guidelines/Maternal_Near_Miss_Operational_Guidelines.pdf).
5. [WHO Assessment for nutrition-related disorders in pregnancy](https://www.who.int/tools/elena/interventions/assessment-methods-pregnancy).
6. [WHO Haemoglobin cutoffs guideline 2024](https://www.who.int/publications/i/item/9789240088542), [full text](https://www.ncbi.nlm.nih.gov/books/n/who376196/pdf/).
7. [WHO Diabetes during pregnancy 2025](https://www.who.int/publications/i/item/9789240117044), [recommendations 8 and 9](https://www.ncbi.nlm.nih.gov/books/NBK619432/pdf/Bookshelf_NBK619432.pdf).
8. [WHO Pre-eclampsia fact sheet](https://www.who.int/news-room/fact-sheets/detail/pre-eclampsia).
9. [NHM Gestational diabetes technical and operational guidelines](https://nhm.gov.in/New_Updates_2018/NHM_Components/RMNCH_MH_Guidelines/Gestational-Diabetes-Mellitus.pdf).
