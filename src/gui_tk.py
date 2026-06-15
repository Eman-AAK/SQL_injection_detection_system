# gui_tk.py
# -------------------------------------------------------------------
# Graphical Interface for SQL Injection Detection System
# - Predict injection attempts using RandomForest model
# - Log encrypted predictions via secure_log.py
# - Demonstrate safe vs unsafe SQL handling techniques
# -------------------------------------------------------------------

import os
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timezone
import numpy as np
import joblib
from scipy.sparse import csr_matrix, hstack

# ---- Project paths ----
BASE_DIR     = r"C:\Users\HP\Downloads\DB_Sql_injection_proj"
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH   = os.path.join(ARTIFACT_DIR, "rf_model.pkl")
VECT_PATH    = os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl")

# ---- Import core detection helpers ----
from sql_injection_pipeline import clean_query, extract_handcrafted_features

# ---- Secure logging ----
try:
    from secure_log import encrypt_and_append, read_and_decrypt
    LOGGING_AVAILABLE = True
except Exception:
    LOGGING_AVAILABLE = False

# ---- Prevention technique demos ----
try:
    from prevention_techniques import unsafe_query, safe_query
    PREVENTION_AVAILABLE = True
except Exception:
    PREVENTION_AVAILABLE = False


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SQL Injection Detector (Random Forest)")
        self.geometry("800x600")
        self.minsize(760, 540)

        # Load trained model + TF-IDF vectorizer
        if not (os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH)):
            messagebox.showerror("Error", f"Artifacts not found in:\n{ARTIFACT_DIR}")
            self.destroy()
            return

        self.vectorizer = joblib.load(VECT_PATH)
        self.model = joblib.load(MODEL_PATH)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 8}

        # --- Header ---
        header = ttk.Label(self, text="SQL Injection Detection System", font=("Segoe UI", 16, "bold"))
        header.pack(anchor="w", **pad)
        sub = ttk.Label(self, text="Enter a SQL query below and click Predict or Demo options.")
        sub.pack(anchor="w", **pad)

        # --- Input Frame ---
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(frm, height=8, wrap="word", font=("Consolas", 10))
        self.txt.pack(fill="both", expand=True)

        # --- Control Buttons ---
        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", **pad)

        self.log_var = tk.BooleanVar(value=True if LOGGING_AVAILABLE else False)
        self.chk_log = ttk.Checkbutton(ctrl, text="Encrypt-log predictions", variable=self.log_var)
        self.chk_log.grid(row=0, column=0, sticky="w")

        btn_predict = ttk.Button(ctrl, text="Predict", command=self.on_predict)
        btn_predict.grid(row=0, column=1, padx=10)

        btn_clear = ttk.Button(ctrl, text="Clear", command=lambda: self.txt.delete("1.0", "end"))
        btn_clear.grid(row=0, column=2)

        btn_view_logs = ttk.Button(
            ctrl,
            text="View Decrypted Logs",
            command=self.on_view_logs,
            state=("normal" if LOGGING_AVAILABLE else "disabled"),
        )
        btn_view_logs.grid(row=0, column=3, padx=10)

        # --- Prevention technique demo buttons ---
        btn_demo_unsafe = ttk.Button(
            ctrl, text="Run Unsafe Demo", command=self.on_demo_unsafe,
            state=("normal" if PREVENTION_AVAILABLE else "disabled")
        )
        btn_demo_unsafe.grid(row=1, column=0, pady=5, sticky="w")

        btn_demo_safe = ttk.Button(
            ctrl, text="Run Safe Demo", command=self.on_demo_safe,
            state=("normal" if PREVENTION_AVAILABLE else "disabled")
        )
        btn_demo_safe.grid(row=1, column=1, pady=5, sticky="w")

        # --- Output Labels ---
        sep = ttk.Separator(self, orient="horizontal")
        sep.pack(fill="x", padx=10, pady=6)

        self.lbl_res = ttk.Label(self, text="Prediction: —", font=("Segoe UI", 12, "bold"))
        self.lbl_res.pack(anchor="w", padx=10, pady=4)

        self.lbl_prob = ttk.Label(self, text="Probability: —", font=("Segoe UI", 11))
        self.lbl_prob.pack(anchor="w", padx=10, pady=2)

        note_txt = (
            f"Logging enabled (encrypted to {os.path.join('artifacts','detections.log.enc')})"
            if self.log_var.get() else "Logging disabled"
        )
        self.lbl_note = ttk.Label(self, text=note_txt, font=("Segoe UI", 9, "italic"))
        self.lbl_note.pack(anchor="w", padx=10, pady=2)

    # --------------------------------------------------------------------
    # CORE DETECTION
    # --------------------------------------------------------------------
    def predict_text(self, raw_query: str):
        raw = raw_query.strip()
        if not raw:
            raise ValueError("Empty query.")
        cleaned = clean_query(raw)
        length, special, digits, sql_kw, digit_ratio = extract_handcrafted_features(cleaned)
        X_text = self.vectorizer.transform([cleaned])
        X_extra = csr_matrix(np.array([[length, special, digits, sql_kw, digit_ratio]], dtype=float))
        X = hstack([X_text, X_extra], format="csr")
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0, 1]
        return int(pred), float(proba), raw

    def on_predict(self):
        try:
            content = self.txt.get("1.0", "end")
            pred, proba, raw = self.predict_text(content)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        label = "ATTACK (1)" if pred == 1 else "NORMAL (0)"
        self.lbl_res.config(text=f"Prediction: {label}")
        self.lbl_prob.config(text=f"Probability: {proba:.4f}")

        if self.log_var.get() and LOGGING_AVAILABLE:
            try:
                encrypt_and_append({
                    "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "query": raw,
                    "pred": pred,
                    "proba": proba
                })
                self.lbl_note.config(text=f"Logged (encrypted) → {os.path.join('artifacts','detections.log.enc')}")
            except Exception as e:
                messagebox.showwarning("Log Warning", f"Failed to write encrypted log:\n{e}")
        elif not LOGGING_AVAILABLE:
            self.lbl_note.config(text="Logging module not found (secure_log.py).")

    # --------------------------------------------------------------------
    # LOG VIEWER
    # --------------------------------------------------------------------
    def on_view_logs(self):
        if not LOGGING_AVAILABLE:
            messagebox.showinfo("Logs", "Logging module not available.")
            return
        try:
            entries = read_and_decrypt()
            if not entries:
                messagebox.showinfo("Logs", "No entries yet.")
                return
            # show last ~10 entries
            last = entries[-10:]
            lines = []
            for r in last:
                try:
                    prob_str = f"{float(r.get('proba', 0.0)):.4f}"
                except Exception:
                    prob_str = str(r.get('proba'))
                lines.append(f"[{r.get('ts','?')}] pred={r.get('pred')} proba={prob_str}\n{r.get('query')}\n")
            self._show_scroller("Decrypted Logs (last 10)", "\n".join(lines))
        except Exception as e:
            messagebox.showerror("Error", f"Could not read logs:\n{e}")

    # --------------------------------------------------------------------
    # SAFE vs UNSAFE DEMOS
    # --------------------------------------------------------------------
    def on_demo_unsafe(self):
        content = self.txt.get("1.0", "end").strip()
        if not content:
            messagebox.showinfo("Demo", "Please enter a query to test.")
            return
        try:
            result = unsafe_query(content)
            messagebox.showwarning("Unsafe Query Demo", f"This is vulnerable!\n\n{result}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_demo_safe(self):
        content = self.txt.get("1.0", "end").strip()
        if not content:
            messagebox.showinfo("Demo", "Please enter a query to test.")
            return
        try:
            safe_query(content)
            messagebox.showinfo("Safe Query Demo", "Safe parameterized query executed successfully.\n(Check console output.)")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # --------------------------------------------------------------------
    # Utility window for displaying logs
    # --------------------------------------------------------------------
    def _show_scroller(self, title, text):
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("740x420")
        box = tk.Text(win, wrap="word", font=("Consolas", 10))
        box.pack(fill="both", expand=True)
        box.insert("1.0", text)
        box.config(state="disabled")


# --------------------------------------------------------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()
