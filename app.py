from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib
import sqlite3

app = Flask(__name__)
app.secret_key = "smart_lender_secret"

# =====================================
# Load Model & Encoders
# =====================================
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")


# =====================================
# Create Database
# =====================================
def create_database():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gender TEXT,
        married TEXT,
        education TEXT,
        applicant_income REAL,
        loan_amount REAL,
        prediction TEXT
    )
    """)

    conn.commit()
    conn.close()


create_database()


# =====================================
# HOME
# =====================================
@app.route("/")
def home():
    return render_template("index.html")


# =====================================
# ABOUT
# =====================================
@app.route("/about")
def about():
    return render_template("about.html")


# =====================================
# CONTACT
# =====================================
@app.route("/contact")
def contact():
    return render_template("contact.html")


# =====================================
# DASHBOARD
# =====================================
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# =====================================
# PERFORMANCE
# =====================================
@app.route("/performance")
def performance():
    return render_template("performance.html")


# =====================================
# LOGIN
# =====================================
@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login_post():

    username = request.form["username"]
    password = request.form["password"]

    if username == "admin" and password == "admin123":
        session["user"] = username
        return redirect(url_for("dashboard"))

    return render_template(
        "login.html",
        error="Invalid Username or Password"
    )


# =====================================
# LOGOUT
# =====================================
@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")


# =====================================
# LOAN PREDICTION
# =====================================
@app.route("/predict", methods=["POST"])
def predict():

    try:

        gender = request.form["gender"]
        married = request.form["married"]
        dependents = request.form["dependents"]
        education = request.form["education"]
        self_employed = request.form["self_employed"]

        applicant_income = float(request.form["applicant_income"])
        coapplicant_income = float(request.form["coapplicant_income"])
        loan_amount = float(request.form["loan_amount"])
        loan_term = float(request.form["loan_term"])
        credit_history = float(request.form["credit_history"])

        property_area = request.form["property_area"]

        # Encode values

        gender_value = encoders["Gender"].transform([gender])[0]
        married_value = encoders["Married"].transform([married])[0]
        dependents_value = encoders["Dependents"].transform([dependents])[0]
        education_value = encoders["Education"].transform([education])[0]
        self_emp_value = encoders["Self_Employed"].transform([self_employed])[0]
        property_value = encoders["Property_Area"].transform([property_area])[0]

        input_df = pd.DataFrame([[
            gender_value,
            married_value,
            dependents_value,
            education_value,
            self_emp_value,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history,
            property_value
        ]], columns=[
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area"
        ])

        prediction = model.predict(input_df)[0]

        if prediction == 1:
            result = "Loan Approved ✅"
            alert = "success"
        else:
            result = "Loan Rejected ❌"
            alert = "danger"

        # Save Prediction

        conn = sqlite3.connect("predictions.db")
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO prediction_history
        (
        gender,
        married,
        education,
        applicant_income,
        loan_amount,
        prediction
        )

        VALUES

        (?, ?, ?, ?, ?, ?)
        """,

        (
        gender,
        married,
        education,
        applicant_income,
        loan_amount,
        result
        ))

        conn.commit()
        conn.close()

        return render_template(
            "index.html",
            prediction=result,
            alert=alert
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction=str(e),
            alert="warning"
        )


# =====================================
# PREDICTION HISTORY
# =====================================
@app.route("/history")
def history():

    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM prediction_history")

    rows = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        rows=rows
    )


# =====================================
# DELETE HISTORY
# =====================================
@app.route("/delete_history")
def delete_history():

    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM prediction_history")

    conn.commit()

    conn.close()

    return redirect("/history")


# =====================================
# RUN
# =====================================
if __name__ == "__main__":
    app.run(debug=True)