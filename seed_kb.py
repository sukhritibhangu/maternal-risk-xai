"""Starter knowledge base. Paraphrased general antenatal guidance (WHO / ACOG / NICE-style themes).
Every document is loaded as UNVERIFIED. An admin must review and mark it verified, or replace it with the book."""

SOURCE = "Starter content (paraphrased public guidance) - needs clinical review"

SEED = [
 dict(title="Raised blood pressure in pregnancy", tags=["SystolicBP", "DiastolicBP"], text="""
In pregnancy, blood pressure of 140 systolic or 90 diastolic or higher on repeated readings is considered hypertension and needs medical follow-up. Raised pressure can be an early sign of pre-eclampsia, which can harm both mother and baby if it is not monitored.

Attend every antenatal visit so blood pressure and urine protein are checked each time. Take any prescribed blood-pressure medicine exactly as advised and do not stop it without speaking to the doctor. Rest, avoid heavy exertion, and avoid very salty processed food, but do not cut food or fluids drastically.

Ask your clinician whether low-dose aspirin or calcium supplements are appropriate for you; these are used for some women at higher risk and must only be started on medical advice.
"""),
 dict(title="Severe blood pressure - urgent care", tags=["SystolicBP", "DiastolicBP"], text="""
A reading of 160 systolic or 110 diastolic or higher in pregnancy is an emergency. The woman should be assessed at a hospital the same day, without waiting for the next routine visit.

Go immediately if high blood pressure comes with a severe headache, blurred vision or flashing lights, pain in the upper abdomen, sudden swelling of the face or hands, vomiting, breathlessness, or reduced baby movements. Do not drive yourself; ask someone to take you.
"""),
 dict(title="Low blood pressure in pregnancy", tags=["SystolicBP"], text="""
Mild drops in blood pressure are common in pregnancy, especially in the middle months. Stand up slowly, drink enough fluids, eat regular meals, and avoid standing still for long periods or lying flat on your back.

If readings stay low or you feel faint, have palpitations, or have bleeding or severe abdominal pain, contact your clinician promptly.
"""),
 dict(title="High blood sugar and gestational diabetes", tags=["RBS", "HBA1C"], text="""
A random blood sugar of 200 mg/dL or more, especially with thirst, frequent urination or tiredness, suggests diabetes and needs confirmation with a fasting test or glucose tolerance test arranged by the doctor. A single random reading is not a diagnosis, and any raised reading should be followed up.

Diet is the first step in managing high sugar in pregnancy. Eat small regular meals, choose whole grains, pulses, vegetables and protein, and limit sugary drinks, sweets, white rice and refined flour portions. Regular walking is helpful unless your clinician has advised against it.

If you are asked to check blood sugar at home, record the readings and bring them to every visit. Some women need medicine or insulin; this is decided by the doctor, not by the woman, and is safe when supervised.
"""),
 dict(title="HbA1c interpretation and diabetes before pregnancy", tags=["HBA1C", "RBS"], text="""
An HbA1c of 6.5 percent or more is in the diabetic range and 5.7 to 6.4 percent is pre-diabetic. A high result should be confirmed with a repeat test. HbA1c is not used to diagnose gestational diabetes, and it can read falsely high when the body is low in iron, as in iron-deficiency anaemia, so the doctor will usually combine it with glucose tests.

Women with diabetes before pregnancy have better outcomes when blood sugar is brought under control early. The individual HbA1c target is set by the clinician. Folic acid and early specialist review are usually recommended.
"""),
 dict(title="Low blood sugar", tags=["RBS"], text="""
Low blood sugar can cause shakiness, sweating, confusion and fainting. If the woman is awake and able to swallow, give a fast source of sugar such as glucose tablets or half a glass of juice, then seek medical advice. If she is drowsy or unconscious, get emergency help.
"""),
 dict(title="Anaemia in pregnancy", tags=["HB"], text="""
Haemoglobin below 11 g/dL in the first and third trimesters, or below 10.5 g/dL in the second trimester, is anaemia in pregnancy. It is graded as mild, moderate (7 to 9.9 g/dL) or severe (below 7 g/dL). Anaemia raises the risk of tiredness, preterm birth, low birth weight and heavy bleeding after delivery.

Daily iron and folic acid tablets are the standard treatment and prevention; take them as prescribed, ideally with a source of vitamin C, and not together with tea, coffee or milk. Eat iron-rich foods such as green leafy vegetables, lentils and pulses, eggs, fish and meat where eaten.

Severe anaemia, or anaemia with breathlessness, chest pain, dizziness or a racing heart, needs urgent medical assessment because injectable iron or a transfusion may be needed. Follow-up haemoglobin tests show whether treatment is working.
"""),
 dict(title="Fever in pregnancy", tags=["BodyTemp"], text="""
A fever during pregnancy should be assessed by a clinician promptly, because infections such as urinary infection, influenza, malaria, dengue or infection of the womb can affect the baby.

Drink plenty of fluids and rest. Do not take ibuprofen or other anti-inflammatory painkillers unless prescribed; ask the doctor or pharmacist which fever medicine is suitable and at what dose. Go to hospital urgently if fever comes with stiff neck, severe headache, rash, burning urination, fluid leaking from the vagina, or reduced baby movements.
"""),
 dict(title="Fast heart rate in pregnancy", tags=["HeartRate"], text="""
Resting heart rate rises in pregnancy, so a fast rate needs to be read with the other signs. Maternal early warning tables flag a rate of 111 to 119 beats per minute for prompt review and 120 or more as urgent. Possible causes include anaemia, infection, dehydration, an overactive thyroid, anxiety or a heart condition.

Rest, drink fluids, limit caffeine, and have haemoglobin, temperature and thyroid checked. Seek urgent care if it comes with chest pain, fainting, or breathlessness at rest.
"""),
 dict(title="Fast breathing in pregnancy", tags=["RR"], text="""
A normal adult resting breathing rate is about 12 to 18 breaths per minute. Maternal early warning tables flag 22 to 29 breaths per minute for prompt review and 30 or more as urgent, especially with fever, cough, chest pain, or blue lips.

Causes include chest infection, asthma, anaemia, blood clots in the lung, heart conditions and pre-eclampsia. Sit upright while waiting for care.
"""),
 dict(title="Age-related risk: adolescent and older mothers", tags=["Age"], text="""
Pregnancy at a very young age, and pregnancy at an older age, can both carry higher risks of anaemia, high blood pressure, diabetes and complications. Good nutrition, iron and folic acid, early and regular antenatal visits, and delivery at a facility that can handle complications are important, along with emotional and social support.

More frequent visits, early screening for blood pressure and diabetes, and a planned place of delivery are usually advised. Do not delay booking antenatal care.
"""),
 dict(title="Many pregnancies, previous loss, previous child death", tags=["G", "P", "L", "A", "D"], text="""
Women who have had many pregnancies have a higher chance of heavy bleeding after delivery, abnormal labour and anaemia. They should plan delivery in a facility with blood available and skilled attendants rather than at home.

Allowing enough time between pregnancies lowers risks for mother and baby; ask about family planning options at the postnatal visit.

After a miscarriage or abortion, early booking and an early ultrasound in the next pregnancy are advised. Repeated losses should prompt specialist investigation. After a previous stillbirth or child death, discuss the cause with the doctor, share the earlier records, and plan delivery where newborn care is available.
"""),
 dict(title="General antenatal care and danger signs", tags=["general"], text="""
Attend at least eight antenatal contacts during pregnancy, or more if you are at higher risk. Eat a varied diet with extra protein, fruit and vegetables, take prescribed iron, folic acid and calcium, avoid tobacco, alcohol and unprescribed medicines, and sleep on your side in late pregnancy.

Go to a hospital urgently for vaginal bleeding, fluid leaking, severe abdominal pain, severe headache or blurred vision, convulsions, fever, difficulty breathing, or if the baby moves much less than usual. Keep your records and emergency contact and transport plan ready.
"""),
]
