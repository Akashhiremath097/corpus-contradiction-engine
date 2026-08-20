"""
Generates the seeded demo corpus for the Corpus Contradiction Engine.

Each document is a plain-text file with a small metadata header (TITLE, SOURCE, DATE)
followed by the body. ingestion.py will parse this header to attach source + date
to every chunk, which is what lets the engine show "Source A (2019) vs Source B (2024)"
in the flagged output instead of just raw text.

Run once: python create_corpus.py
Writes files into data/corpus/ and an answer_key.json alongside it.
"""

import json
import os

OUTPUT_DIR = os.path.join("data", "corpus")
ANSWER_KEY_PATH = os.path.join("data", "answer_key.json")

# Each doc: (filename, title, source, date, body)
DOCS = [
    # ---------- HR: conflict pair 1 — approval requirement (categorical) ----------
    (
        "hr_handbook_2019.txt",
        "Employee Handbook - Remote Work",
        "Employee Handbook 2019",
        "2019-03-01",
        "Remote work requires prior approval from your reporting manager for each "
        "instance. Employees must submit a request at least 24 hours in advance.",
    ),
    (
        "hr_hybrid_policy_2024.txt",
        "Hybrid Work Policy",
        "Hybrid Work Policy 2024",
        "2024-01-15",
        "Employees may work remotely up to three days per week without prior "
        "approval. Managers should be notified but do not need to approve each instance.",
    ),
    # ---------- HR: conflict pair 2 — sick days (numeric) ----------
    (
        "hr_leave_policy_2020.txt",
        "Leave Policy",
        "Leave Policy 2020",
        "2020-06-01",
        "Full-time employees are entitled to 12 paid sick days per calendar year. "
        "Unused sick days do not carry over to the next year.",
    ),
    (
        "hr_leave_policy_2023.txt",
        "Leave Policy - Revised",
        "Leave Policy 2023 Revision",
        "2023-04-10",
        "Full-time employees are entitled to 15 paid sick days per calendar year, "
        "effective this revision. Unused sick days do not carry over.",
    ),
    # ---------- HR: conflict pair 3 — expense claim window (numeric) ----------
    (
        "hr_expense_policy_2021.txt",
        "Expense Reimbursement Policy",
        "Expense Policy 2021",
        "2021-02-01",
        "All expense claims must be submitted within 30 days of the purchase date. "
        "Claims submitted after this window will not be reimbursed.",
    ),
    (
        "hr_expense_policy_2024.txt",
        "Expense Reimbursement Policy - Update",
        "Expense Policy 2024 Update",
        "2024-05-01",
        "All expense claims must be submitted within 60 days of the purchase date. "
        "This extends the previous window to reduce missed reimbursements.",
    ),
    # ---------- HR: conflict pair 4 — probation length (numeric) ----------
    (
        "hr_probation_policy_2019.txt",
        "Probation Policy",
        "Probation Policy 2019",
        "2019-08-01",
        "All new hires undergo a probation period of 6 months from their start "
        "date, during which performance is formally reviewed monthly.",
    ),
    (
        "hr_probation_policy_2023.txt",
        "Probation Policy - Revised",
        "Probation Policy 2023 Revision",
        "2023-09-01",
        "All new hires undergo a probation period of 3 months from their start "
        "date, shortened from the previous policy following management review.",
    ),
    # ---------- HR: distractors (no conflict) ----------
    (
        "hr_dress_code_2018.txt",
        "Dress Code Policy",
        "Dress Code Policy 2018",
        "2018-01-01",
        "Business formal attire is required Monday through Thursday. Casual "
        "dress is permitted on Fridays.",
    ),
    (
        "hr_onboarding_guide_2022.txt",
        "New Hire Onboarding Guide",
        "Onboarding Guide 2022",
        "2022-03-15",
        "New hire orientation runs for two full days and covers company policy, "
        "benefits enrollment, and workstation setup.",
    ),
    # ---------- Healthcare: conflict pair 1 — dosage (numeric, fictional drug) ----------
    (
        "hc_dosage_guideline_2020.txt",
        "Medazol Dosage Guideline",
        "Dosage Guideline 2020",
        "2020-02-01",
        "The standard adult dose of Medazol is 500mg administered twice daily "
        "for the standard treatment course.",
    ),
    (
        "hc_dosage_protocol_2023.txt",
        "Medazol Dosage Protocol - Updated",
        "Dosage Protocol 2023",
        "2023-07-01",
        "The standard adult dose of Medazol has been revised to 250mg administered "
        "twice daily, following updated safety data on side effects at higher doses.",
    ),
    # ---------- Healthcare: conflict pair 2 — observation window (numeric) ----------
    (
        "hc_discharge_protocol_2019.txt",
        "Post-Procedure Discharge Protocol",
        "Discharge Protocol 2019",
        "2019-05-01",
        "Patients undergoing Procedure Q must be observed for 24 hours before "
        "discharge is approved.",
    ),
    (
        "hc_discharge_protocol_2022.txt",
        "Post-Procedure Discharge Protocol - Revised",
        "Discharge Protocol 2022 Revision",
        "2022-11-01",
        "Patients undergoing Procedure Q must be observed for 12 hours before "
        "discharge, shortened based on updated outcomes data.",
    ),
    # ---------- Healthcare: conflict pair 3 — visiting hours (range) ----------
    (
        "hc_visiting_hours_2021.txt",
        "Visiting Hours Policy",
        "Visiting Hours Policy 2021",
        "2021-01-01",
        "Visiting hours are 9:00 AM to 5:00 PM daily. Visitors outside these "
        "hours require prior authorization from the ward nurse.",
    ),
    (
        "hc_visiting_hours_2024.txt",
        "Visiting Hours Policy - Extended",
        "Visiting Hours Policy 2024",
        "2024-02-01",
        "Visiting hours have been extended to 9:00 AM to 8:00 PM daily to "
        "accommodate working family members.",
    ),
    # ---------- Healthcare: conflict pair 4 — fasting window (numeric) ----------
    (
        "hc_fasting_guideline_2020.txt",
        "Pre-Procedure Fasting Guideline",
        "Fasting Guideline 2020",
        "2020-09-01",
        "Patients must fast for 8 hours prior to Procedure R, including no "
        "clear liquids in the final 8-hour window.",
    ),
    (
        "hc_fasting_guideline_2023.txt",
        "Pre-Procedure Fasting Guideline - Updated",
        "Fasting Guideline 2023",
        "2023-03-01",
        "The fasting requirement before Procedure R has been reduced to 6 hours, "
        "per updated anesthesia society guidance.",
    ),
    # ---------- Healthcare: distractors (no conflict) ----------
    (
        "hc_infection_control_2020.txt",
        "Infection Control Guideline",
        "Infection Control Guideline 2020",
        "2020-04-01",
        "Personal protective equipment is required for all direct patient "
        "contact in isolation wards.",
    ),
    (
        "hc_staff_training_2022.txt",
        "Clinical Staff Training Manual",
        "Staff Training Manual 2022",
        "2022-06-01",
        "All clinical staff must complete annual certification renewal covering "
        "emergency response and equipment handling.",
    ),
]

# (doc_a_filename, doc_b_filename, label) — label is "contradiction" or "no_conflict"
ANSWER_KEY = [
    ("hr_handbook_2019.txt", "hr_hybrid_policy_2024.txt", "contradiction"),
    ("hr_leave_policy_2020.txt", "hr_leave_policy_2023.txt", "contradiction"),
    ("hr_expense_policy_2021.txt", "hr_expense_policy_2024.txt", "contradiction"),
    ("hr_probation_policy_2019.txt", "hr_probation_policy_2023.txt", "contradiction"),
    ("hc_dosage_guideline_2020.txt", "hc_dosage_protocol_2023.txt", "contradiction"),
    ("hc_discharge_protocol_2019.txt", "hc_discharge_protocol_2022.txt", "contradiction"),
    ("hc_visiting_hours_2021.txt", "hc_visiting_hours_2024.txt", "contradiction"),
    ("hc_fasting_guideline_2020.txt", "hc_fasting_guideline_2023.txt", "contradiction"),
    # a couple of explicit no-conflict pairs, for false-positive spot checks
    ("hr_dress_code_2018.txt", "hr_onboarding_guide_2022.txt", "no_conflict"),
    ("hc_infection_control_2020.txt", "hc_staff_training_2022.txt", "no_conflict"),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename, title, source, date, body in DOCS:
        path = os.path.join(OUTPUT_DIR, filename)
        content = f"TITLE: {title}\nSOURCE: {source}\nDATE: {date}\n\n{body}\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Wrote {path}")

    with open(ANSWER_KEY_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [{"doc_a": a, "doc_b": b, "label": label} for a, b, label in ANSWER_KEY],
            f,
            indent=2,
        )
    print(f"\nWrote {ANSWER_KEY_PATH}")

    contradiction_count = sum(1 for _, _, label in ANSWER_KEY if label == "contradiction")
    print(f"\nTotal documents: {len(DOCS)}")
    print(f"Seeded conflict pairs: {contradiction_count}")


if __name__ == "__main__":
    main()