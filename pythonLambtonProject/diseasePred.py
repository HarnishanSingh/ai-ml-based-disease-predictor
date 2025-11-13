# diseasePred_pretty.py
# Same logic as your original script; only GUI appearance/styling changed.
from tkinter import *
from tkinter import ttk, messagebox
import sqlite3
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import tree, ensemble, naive_bayes
from sklearn.metrics import accuracy_score

# ---------------------------
# Symptom list (unchanged)
# ---------------------------
l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
      'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
      'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
      'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
      'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
      'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
      'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
      'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
      'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
      'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
      'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
      'family_history', 'mucoid_sputum',
      'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
      'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
      'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
      'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
      'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
      'yellow_crust_ooze']

# ---------------------------
# Disease labels (consistent mapping)
# ---------------------------
disease_labels = {
    'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
    'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8,
    'Bronchial Asthma': 9, 'Hypertension ': 10, 'Migraine': 11, 'Cervical spondylosis': 12,
    'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
    'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19, 'Hepatitis B': 20, 'Hepatitis C': 21,
    'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
    'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
    'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32, 'Hypoglycemia': 33,
    'Osteoarthristis': 34, 'Arthritis': 35, '(vertigo) Paroymsal  Positional Vertigo': 36,
    'Acne': 37, 'Urinary tract infection': 38, 'Psoriasis': 39, 'Impetigo': 40
}

# Reverse mapping for predictions
label_to_disease = {v: k for k, v in disease_labels.items()}

# Prepare a normalized mapping for labels (strip and lowercase keys)
_norm_label_map = {k.strip().lower(): v for k, v in disease_labels.items()}

# ---------------------------
# SQLite setup (lightweight DB to store predictions)
# ---------------------------
DB_FILE = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    # Create table with a safe superset of columns (older DBs will be upgraded by ensure_db_schema)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            name TEXT,
            symptoms TEXT,
            model TEXT,
            top_prediction TEXT,
            top_prob REAL,
            top_3_json TEXT,
            consensus_top TEXT,
            consensus_prob REAL
        )
    """)
    conn.commit()
    conn.close()
    # Ensure any missing columns are added (migrates older DB)
    ensure_db_schema()

def ensure_db_schema():
    required_cols = {
        "id": "INTEGER",
        "timestamp": "TEXT",
        "name": "TEXT",
        "symptoms": "TEXT",
        "model": "TEXT",
        "top_prediction": "TEXT",
        "top_prob": "REAL",
        "top_3_json": "TEXT",
        "consensus_top": "TEXT",
        "consensus_prob": "REAL"
    }
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(predictions)")
        existing = {row[1]: row for row in cur.fetchall()}
        for col, coltype in required_cols.items():
            if col not in existing:
                if col == "id":
                    continue
                sql = f"ALTER TABLE predictions ADD COLUMN {col} {coltype}"
                cur.execute(sql)
        conn.commit()
    except Exception as e:
        try:
            messagebox.showwarning("DB Migration Warning", f"Could not fully migrate DB schema: {e}")
        except Exception:
            print("DB Migration Warning:", e)
    finally:
        try:
            conn.close()
        except Exception:
            pass

def save_prediction_to_db(name, symptoms, model_name, top_pred_name, top_prob, top3, consensus_top=None, consensus_prob=None):
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO predictions (timestamp, name, symptoms, model, top_prediction, top_prob, top_3_json, consensus_top, consensus_prob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), name, ",".join(symptoms), model_name, top_pred_name, float(top_prob), json.dumps(top3), consensus_top, consensus_prob))
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass

# Robust fetch_all_records (will re-raise exceptions to be handled by caller)
def fetch_all_records(limit=200):
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("SELECT id, timestamp, name, symptoms, model, top_prediction, top_prob, top_3_json, consensus_top, consensus_prob FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        raise

# Initialize DB and ensure schema
init_db()

# ---------------------------
# Data loading (robust to label formatting)
# ---------------------------
def _try_read_csv(names):
    for name in names:
        if os.path.exists(name):
            return pd.read_csv(name)
    for name in names:
        try:
            return pd.read_csv(name)
        except Exception:
            pass
    raise FileNotFoundError(f"None of the dataset files found: {names}")

def _map_prognosis_value(val):
    if pd.isna(val):
        raise ValueError("Empty prognosis value encountered.")
    if isinstance(val, (int, np.integer)):
        return int(val)
    s = str(val).strip()
    if s.isdigit():
        return int(s)
    key = s.lower()
    if key in _norm_label_map:
        return int(_norm_label_map[key])
    import re
    key2 = re.sub(r'[^a-z0-9]', '', key)
    for k_norm, v in _norm_label_map.items():
        k2 = re.sub(r'[^a-z0-9]', '', k_norm)
        if k2 == key2:
            return int(v)
    raise ValueError(f"Could not map prognosis label '{val}' to a numeric code. Try normalizing label or updating disease_labels dict.")

def load_data(file_path_candidates):
    df = _try_read_csv(file_path_candidates)
    if 'prognosis' not in df.columns:
        raise KeyError("Dataset must contain a 'prognosis' column.")
    df = df.copy()
    mapped = []
    for i, val in enumerate(df['prognosis'].values):
        try:
            mapped_val = _map_prognosis_value(val)
        except Exception as e:
            raise ValueError(f"Unable to parse prognosis at row {i}: {e}")
        mapped.append(mapped_val)
    df['prognosis'] = mapped
    X_data = df[l1]
    y_data = df['prognosis']
    return X_data, y_data, df

# Try to load training/testing (many environments have lowercase filenames)
try:
    X_train, y_train, train_df_full = load_data(["Training.csv", "training.csv"])
    X_test, y_test, test_df_full = load_data(["Testing.csv", "testing.csv"])
except Exception as e:
    message = f"Dataset loading error: {e}"
    try:
        messagebox.showerror("Dataset loading error", message)
    except Exception:
        print(message)
    raise

# ---------------------------
# Check for overlap between train and test
# ---------------------------
def check_overlap(train_df, test_df):
    common = pd.merge(train_df.drop(columns=['prognosis']), test_df.drop(columns=['prognosis']), how='inner')
    return len(common)

overlap_count = check_overlap(train_df_full, test_df_full)
if overlap_count > 0:
    messagebox.showwarning("Data Warning",
                           f"There are {overlap_count} identical rows between training and testing sets.\n"
                           "This may inflate test accuracy. Consider creating a proper train/test split.")

# ---------------------------
# Model training utilities
# ---------------------------
def train_decision_tree(X_train, y_train):
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, np.ravel(y_train))
    return clf

def train_random_forest(X_train, y_train):
    clf = ensemble.RandomForestClassifier()
    clf.fit(X_train, np.ravel(y_train))
    return clf

def train_naive_bayes(X_train, y_train):
    clf = naive_bayes.GaussianNB()
    clf.fit(X_train, np.ravel(y_train))
    return clf

# Train models (once)
decision_tree_model = train_decision_tree(X_train, y_train)
random_forest_model = train_random_forest(X_train, y_train)
naive_bayes_model = train_naive_bayes(X_train, y_train)

# Compute and store accuracies on test set
def compute_accuracy(model, X_test, y_test):
    preds = model.predict(X_test)
    return accuracy_score(np.ravel(y_test), preds)

dt_acc = compute_accuracy(decision_tree_model, X_test, y_test)
rf_acc = compute_accuracy(random_forest_model, X_test, y_test)
nb_acc = compute_accuracy(naive_bayes_model, X_test, y_test)

# ---------------------------
# Prediction helpers
# ---------------------------
def build_input_vector(symptoms):
    vec = [0] * len(l1)
    normalized = [s for s in symptoms if s and s != "Select Here" and s != "None"]
    for s in normalized:
        try:
            idx = l1.index(s)
            vec[idx] = 1
        except ValueError:
            pass
    return vec

def model_proba_vector(model, input_vector):
    max_label = max(label_to_disease.keys())
    proba_vec = np.zeros(max_label + 1, dtype=float)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([input_vector])[0]
        classes = getattr(model, "classes_", np.arange(len(probs)))
        for i, cls in enumerate(classes):
            cls_int = int(cls)
            if 0 <= cls_int <= max_label:
                proba_vec[cls_int] = float(probs[i])
    else:
        pred = int(model.predict([input_vector])[0])
        if 0 <= pred <= max_label:
            proba_vec[pred] = 1.0
    return proba_vec

def get_top_k_from_vector(proba_vec, k=3):
    idx_sorted = np.argsort(proba_vec)[::-1]
    pairs = []
    for idx in idx_sorted[:k]:
        name = label_to_disease.get(int(idx), str(idx))
        prob = float(proba_vec[idx])
        pairs.append((name, prob))
    return pairs

def get_top_k_predictions(model, input_vector, k=3):
    proba_vec = model_proba_vector(model, input_vector)
    return get_top_k_from_vector(proba_vec, k=k)

def get_consensus(input_vector, k=3):
    weights = np.array([max(dt_acc, 1e-6), max(rf_acc, 1e-6), max(nb_acc, 1e-6)], dtype=float)
    weights = weights / weights.sum()
    dt_vec = model_proba_vector(decision_tree_model, input_vector)
    rf_vec = model_proba_vector(random_forest_model, input_vector)
    nb_vec = model_proba_vector(naive_bayes_model, input_vector)
    combined = weights[0] * dt_vec + weights[1] * rf_vec + weights[2] * nb_vec
    topk = get_top_k_from_vector(combined, k=k)
    return topk, combined

# ---------------------------
# GUI setup (prettified)
# ---------------------------
root = Tk()
root.title('Smart Disease Predictor System by Harnishan Singh and Kunal Sharma')
root.geometry("")  # keep sizing default (resizable set below)
root.configure(background='#F5F7FA')
root.resizable(0, 0)

# Setup ttk style for nicer-looking widgets (keeps grid positions the same)
style = ttk.Style(root)
# Try to use a clean theme if available
try:
    style.theme_use('clam')
except Exception:
    pass

# Configure custom styles
style.configure('Title.TLabel', font=('Helvetica', 20, 'bold'), foreground='#2b2b2b', background='#F5F7FA')
style.configure('LabelBold.TLabel', font=('Helvetica', 11, 'bold'), background='#F5F7FA')
style.configure('Stat.TLabel', font=('Helvetica', 10), background='#F5F7FA')
style.configure('Action.TButton', font=('Helvetica', 11, 'bold'), padding=6)
style.configure('Neutral.TButton', font=('Helvetica', 10), padding=4)
style.map('Action.TButton',
          background=[('active', '#1f8f4a'), ('!active', '#2bb673')],
          foreground=[('!disabled', '#000')])

# Fonts
HEADER_FONT = ("Helvetica", 18, "bold")
LABEL_FONT = ("Helvetica", 11, "bold")
TEXT_FONT = ("Helvetica", 11)
SMALL_FONT = ("Helvetica", 10)

# Top title (kept in same grid row/col to preserve alignment)
w2 = ttk.Label(root, text="Disease Predictor using Machine Learning", style='Title.TLabel')
w2.grid(row=0, column=0, columnspan=4, padx=80, pady=(12, 6), sticky=W)

# Name & input labels (same grid positions)
Name = StringVar()
Symptom1 = StringVar(); Symptom1.set("Select Here")
Symptom2 = StringVar(); Symptom2.set("Select Here")
Symptom3 = StringVar(); Symptom3.set("Select Here")
Symptom4 = StringVar(); Symptom4.set("Select Here")
Symptom5 = StringVar(); Symptom5.set("Select Here")

NameLb = ttk.Label(root, text="Name of the Patient", style='LabelBold.TLabel')
NameLb.grid(row=1, column=0, pady=6, sticky=W)

S1Lb = ttk.Label(root, text="Symptom 1", style='LabelBold.TLabel')
S1Lb.grid(row=2, column=0, pady=4, sticky=W)

S2Lb = ttk.Label(root, text="Symptom 2", style='LabelBold.TLabel')
S2Lb.grid(row=3, column=0, pady=4, sticky=W)

S3Lb = ttk.Label(root, text="Symptom 3", style='LabelBold.TLabel')
S3Lb.grid(row=4, column=0, pady=4, sticky=W)

S4Lb = ttk.Label(root, text="Symptom 4", style='LabelBold.TLabel')
S4Lb.grid(row=5, column=0, pady=4, sticky=W)

S5Lb = ttk.Label(root, text="Symptom 5", style='LabelBold.TLabel')
S5Lb.grid(row=6, column=0, pady=4, sticky=W)

# Accuracy labels (kept positions)
lrLb = ttk.Label(root, text="Decision Tree", style='LabelBold.TLabel')
lrLb.grid(row=8, column=0, pady=6, sticky=W)
dt_acc_label = ttk.Label(root, text=f"Accuracy: {dt_acc:.2%}", style='Stat.TLabel')
dt_acc_label.grid(row=8, column=2, sticky=W)

destreeLb = ttk.Label(root, text="Random Forest", style='LabelBold.TLabel')
destreeLb.grid(row=9, column=0, pady=6, sticky=W)
rf_acc_label = ttk.Label(root, text=f"Accuracy: {rf_acc:.2%}", style='Stat.TLabel')
rf_acc_label.grid(row=9, column=2, sticky=W)

ranfLb = ttk.Label(root, text="Naive Bayes", style='LabelBold.TLabel')
ranfLb.grid(row=10, column=0, pady=6, sticky=W)
nb_acc_label = ttk.Label(root, text=f"Accuracy: {nb_acc:.2%}", style='Stat.TLabel')
nb_acc_label.grid(row=10, column=2, sticky=W)

consLb = ttk.Label(root, text="Consensus", style='LabelBold.TLabel')
consLb.grid(row=11, column=0, pady=6, sticky=W)

# Make the options list sorted and slightly shorter display width
OPTIONS = sorted(l1)

# Input widgets (keep same grid coordinates)
NameEn = ttk.Entry(root, textvariable=Name, width=30, font=TEXT_FONT)
NameEn.grid(row=1, column=1, pady=6, sticky=W)

# Keep OptionMenu (to avoid layout shift), but style it via config and add padding
def make_optionmenu(var, row):
    om = OptionMenu(root, var, *OPTIONS)
    om.config(width=28, relief='groove', bd=1, highlightthickness=0, anchor='w')
    om.grid(row=row, column=1, sticky=W, padx=(0, 4))
    return om

S1 = make_optionmenu(Symptom1, 2)
S2 = make_optionmenu(Symptom2, 3)
S3 = make_optionmenu(Symptom3, 4)
S4 = make_optionmenu(Symptom4, 5)
S5 = make_optionmenu(Symptom5, 6)

# Outputs: Text widgets with nicer border, slightly taller and same grid positions
t1 = Text(root, font=TEXT_FONT, height=4, bg="#FFFFFF", width=60, fg="#222", relief="solid", bd=1, padx=6, pady=6)
t1.grid(row=8, column=1, columnspan=2, padx=10, pady=4)

t2 = Text(root, font=TEXT_FONT, height=4, bg="#FFFFFF", width=60, fg="#222", relief="solid", bd=1, padx=6, pady=6)
t2.grid(row=9, column=1, columnspan=2, padx=10, pady=4)

t3 = Text(root, font=TEXT_FONT, height=4, bg="#FFFFFF", width=60, fg="#222", relief="solid", bd=1, padx=6, pady=6)
t3.grid(row=10, column=1, columnspan=2, padx=10, pady=4)

t_cons = Text(root, font=("Helvetica", 11, "bold"), height=2, bg="#FFFFFF", width=60, fg="#111", relief="solid", bd=1, padx=6, pady=6)
t_cons.grid(row=11, column=1, columnspan=2, padx=10, pady=4)

# Small helper functions (unchanged)
def validate_input(name, symptoms):
    if not name or name.strip() == "":
        return "Please enter the name."
    if all((s is None) or (s == "") or (s == "Select Here") or (s == "None") for s in symptoms):
        return "Please select at least one symptom."
    return None

def format_top3_text(pairs):
    lines = []
    for i, (dname, prob) in enumerate(pairs, start=1):
        lines.append(f"{i}. {dname} — {prob:.2%}")
    return "\n".join(lines)

def predict_and_display(symptoms, model, output_text_widget, model_name, return_top3=False):
    name = Name.get()
    v = validate_input(name, symptoms)
    if v:
        messagebox.showerror("Error", v)
        return None
    input_vector = build_input_vector(symptoms)
    top3 = get_top_k_predictions(model, input_vector, k=3)
    output_text_widget.delete("1.0", END)
    output_text_widget.insert(END, format_top3_text(top3))
    return top3

def make_all_predictions_and_consensus():
    symptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    name = Name.get()
    v = validate_input(name, symptoms)
    if v:
        messagebox.showerror("Error", v)
        return
    input_vector = build_input_vector(symptoms)
    top_dt = predict_and_display(symptoms, decision_tree_model, t1, "DecisionTree", return_top3=True)
    top_rf = predict_and_display(symptoms, random_forest_model, t2, "RandomForest", return_top3=True)
    top_nb = predict_and_display(symptoms, naive_bayes_model, t3, "NaiveBayes", return_top3=True)
    consensus_top3, combined_vec = get_consensus(input_vector, k=3)
    t_cons.delete("1.0", END)
    t_cons.insert(END, format_top3_text(consensus_top3))
    top1s = []
    for top in (top_dt, top_rf, top_nb):
        if top and len(top) > 0:
            top1s.append(top[0][0])
    disagreement = len(set(top1s)) > 1
    if disagreement:
        try:
            warning_label.config(text="Models disagree on top prediction", foreground="red")
        except Exception:
            pass
    else:
        try:
            warning_label.config(text="Models agree on top prediction", foreground="green")
        except Exception:
            pass
    try:
        if top_dt:
            save_prediction_to_db(name, [s for s in symptoms if s and s != "Select Here"], "DecisionTree",
                                  top_dt[0][0], top_dt[0][1], top_dt)
        if top_rf:
            save_prediction_to_db(name, [s for s in symptoms if s and s != "Select Here"], "RandomForest",
                                  top_rf[0][0], top_rf[0][1], top_rf)
        if top_nb:
            save_prediction_to_db(name, [s for s in symptoms if s and s != "Select Here"], "NaiveBayes",
                                  top_nb[0][0], top_nb[0][1], top_nb)
        if consensus_top3:
            save_prediction_to_db(name, [s for s in symptoms if s and s != "Select Here"], "Consensus",
                                  consensus_top3[0][0], consensus_top3[0][1], consensus_top3, consensus_top3[0][0], consensus_top3[0][1])
    except Exception as e:
        messagebox.showwarning("DB Warning", f"Could not save some records to DB: {e}")

def Reset():
    Symptom1.set("Select Here")
    Symptom2.set("Select Here")
    Symptom3.set("Select Here")
    Symptom4.set("Select Here")
    Symptom5.set("Select Here")
    NameEn.delete(0, END)
    reset_output_fields()

def reset_output_fields():
    t1.delete("1.0", END)
    t2.delete("1.0", END)
    t3.delete("1.0", END)
    t_cons.delete("1.0", END)
    try:
        warning_label.config(text="")
    except Exception:
        pass

def Exit():
    qExit = messagebox.askyesno("System", "Do you want to exit the system?")
    if qExit:
        root.destroy()
        exit()

# show_past_records_window (unchanged except some cosmetic scrollbar packing kept same)
def show_past_records_window():
    try:
        rows = fetch_all_records(limit=500)
    except Exception as exc:
        messagebox.showerror("DB Error", f"Could not fetch records from database:\n{exc}")
        return

    if not rows:
        messagebox.showinfo("No Records", "No past prediction records found in the database.")
        return

    # Create window
    win = Toplevel(root)
    win.title("Past Predictions")
    win.geometry("1000x450")
    win.configure(background="#F5F7FA")

    cols = ("id", "timestamp", "name", "symptoms", "model", "top_prediction", "top_prob", "top_3_json", "consensus_top", "consensus_prob")
    tree = ttk.Treeview(win, columns=cols, show='headings', height=18)
    for c in cols:
        tree.heading(c, text=c)
        width = 120
        if c == "top_3_json":
            width = 350
        elif c in ("timestamp", "symptoms"):
            width = 180
        tree.column(c, width=width, anchor=W)

    for r in rows:
        safe_row = []
        for val in r:
            try:
                safe_row.append(str(val))
            except Exception:
                safe_row.append(repr(val))
        tree.insert("", END, values=safe_row)

    vsb = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    vsb.pack(side=RIGHT, fill=Y)
    tree.pack(side=LEFT, fill=BOTH, expand=1)

    def export_to_csv():
        try:
            import csv
            fname = "predictions_export.csv"
            with open(fname, "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                for r in rows:
                    writer.writerow(r)
            messagebox.showinfo("Exported", f"Records exported to {os.path.abspath(fname)}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not export records:\n{e}")

    def clear_db():
        if not messagebox.askyesno("Confirm", "This will DELETE all saved prediction records. Continue?"):
            return
        try:
            conn = sqlite3.connect(DB_FILE)
            cur = conn.cursor()
            cur.execute("DELETE FROM predictions")
            conn.commit()
            conn.close()
            messagebox.showinfo("Cleared", "All records deleted.")
            win.destroy()
        except Exception as e:
            messagebox.showerror("DB Error", f"Could not clear database:\n{e}")

    btn_frame = Frame(win, bg="#F5F7FA")
    btn_frame.pack(fill=X, pady=6)
    export_btn = ttk.Button(btn_frame, text="Export CSV", style='Neutral.TButton', command=export_to_csv)
    export_btn.pack(side=LEFT, padx=6)
    close_btn = ttk.Button(btn_frame, text="Close", style='Neutral.TButton', command=win.destroy)
    close_btn.pack(side=LEFT, padx=6)
    clear_btn = ttk.Button(btn_frame, text="Clear All Records", style='Neutral.TButton', command=clear_db)
    clear_btn.pack(side=LEFT, padx=6)

# Buttons (kept same grid positions; styled)
dst = ttk.Button(root, text="Prediction 1", style='Action.TButton',
                 command=lambda: predict_and_display([Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()], decision_tree_model, t1, "DecisionTree"))
dst.config()
dst.grid(row=1, column=3, padx=10, sticky=W)

rnf = ttk.Button(root, text="Prediction 2", style='Neutral.TButton',
                 command=lambda: predict_and_display([Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()], random_forest_model, t2, "RandomForest"))
rnf.grid(row=2, column=3, padx=10, sticky=W)

lr = ttk.Button(root, text="Prediction 3", style='Neutral.TButton',
                command=lambda: predict_and_display([Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()], naive_bayes_model, t3, "NaiveBayes"))
lr.grid(row=3, column=3, padx=10, sticky=W)

all_btn = ttk.Button(root, text="All + Consensus", style='Action.TButton', command=make_all_predictions_and_consensus, width=16)
all_btn.grid(row=4, column=3, padx=10, pady=4, sticky=W)

rs = ttk.Button(root, text="Reset Inputs", style='Neutral.TButton', command=Reset, width=16)
rs.grid(row=5, column=3, padx=10, sticky=W)

rs_output = ttk.Button(root, text="Reset Output", style='Neutral.TButton', command=reset_output_fields, width=16)
rs_output.grid(row=6, column=3, padx=10, sticky=W)

show_db_btn = ttk.Button(root, text="Show Past Records", style='Neutral.TButton', command=show_past_records_window, width=16)
show_db_btn.grid(row=7, column=3, padx=10, sticky=W)

ex = ttk.Button(root, text="Exit System", style='Neutral.TButton', command=Exit, width=16)
ex.grid(row=8, column=3, padx=10, sticky=W)

warning_label = ttk.Label(root, text="", font=SMALL_FONT)
warning_label.grid(row=12, column=1, columnspan=2, pady=6)

# subtle separator line to visually divide inputs and outputs (keeps alignment)
sep = ttk.Separator(root, orient='horizontal')
sep.grid(row=7, column=0, columnspan=3, sticky="ew", padx=10, pady=6)

root.mainloop()
