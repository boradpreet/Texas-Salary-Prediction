import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import gdown

# Load the trained model
@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?export=download&id=1rofxWTn1jhlyYcn0_iu-IsPAtlW5R01-"
    output = "Taxes.pkl"
    gdown.download(url, output, quiet=False)
    
    with open(output, "rb") as f:
        model = pickle.load(f)
    return model

# Load the model
model = load_model()

agency_names = [
    'COMPTROLLER OF PUBLIC ACCOUNTS, JUDICIARY SECTION ',
    'OFFICE OF COURT ADMINISTRATION                    ',
    'TEXAS DEPARTMENT OF CRIMINAL JUSTICE              ',
    'OFFICE OF THE ATTORNEY GENERAL                    ',
    'TEXAS DEPARTMENT OF TRANSPORTATION                ',
    'TEXAS BEHAVIORAL HEALTH EXECUTIVE COUNCIL         ',
    'BOARD OF EXAMINERS OF PSYCHOLOGISTS               ',
    'DEPARTMENT OF STATE HEALTH SERVICES               ',
    'DEPARTMENT OF FAMILY AND PROTECTIVE SERVICES      ',
    'HEALTH AND HUMAN SERVICES COMMISSION              ',
    'DEPARTMENT OF AGRICULTURE                         ',
    'SCHOOL FOR THE BLIND AND VISUALLY IMPAIRED        ',
    'SCHOOL FOR THE DEAF                               ',
    'SENATE                                            ',
    'LEGISLATIVE BUDGET BOARD                          ',
    'HOUSE OF REPRESENTATIVES                          ',
    'TEXAS FACILITIES COMMISSION                       ',
    'STATE PRESERVATION BOARD                          ',
    'TEXAS LEGISLATIVE COUNCIL                         ',
    'LEGISLATIVE REFERENCE LIBRARY                     ',
    'SUPREME COURT OF TEXAS                            ',
    'CT CRIM APPEALS                                   ',
    'OFFICE OF STATE PROSECUTING ATTORNEY              ',
    'OFFICE OF CAPITAL AND FORENSIC WRITS              ',
    'FIRST COURT OF APPEALS DISTRICT                   ',
    'SECOND COURT OF APPEALS DISTRICT                  ',
    'THIRD COURT OF APPEALS DISTRICT                   ',
    'FOURTH COURT OF APPEALS DISTRICT                  ',
    'FIFTH COURT OF APPEALS DISTRICT                   ',
    'SIXTH COURT OF APPEALS DISTRICT                   ',
    'SEVENTH COURT OF APPEALS DISTRICT                 ',
    'EIGHTH COURT OF APPEALS DISTRICT                  ',
    'NINTH COURT OF APPEALS DISTRICT                   ',
    'TENTH COURT OF APPEALS DISTRICT                   ',
    'ELEVENTH COURT OF APPEALS DISTRICT                ',
    'TWELFTH COURT OF APPEALS DISTRICT                 ',
    'THIRTEENTH COURT OF APPEALS DISTRICT              ',
    'FOURTEENTH COURT OF APPEALS DISTRICT              ',
    'STATE COMMISSION ON JUDICIAL CONDUCT              ',
    'STATE LAW LIBRARY                                 ',
    "GOVERNOR'S OFFICE, TRUSTEE PROGRAMS               ",
    'OFFICE OF THE GOVERNOR                            ',
    'COMPTROLLER OF PUBLIC ACCOUNTS                    ',
    'LIBRARY AND ARCHIVES COMMISSION                   ',
    'SECRETARY OF STATE                                ',
    'DEPARTMENT OF INFORMATION RESOURCES               ',
    'TEXAS WORKFORCE COMMISSION                        ',
    'TEACHER RETIREMENT SYSTEM                         ',
    'TEXAS EMERGENCY SERVICES RETIREMENT SYSTEM        ',
    'REAL ESTATE COMMISSION                            ',
    'DEPARTMENT OF HOUSING AND COMMUNITY AFFAIRS       ',
    'STATE PENSION REVIEW BOARD                        ',
    'TEXAS BOND REVIEW BOARD                           ',
    'STATE OFFICE OF ADMINISTRATIVE HEARINGS           ',
    'TEXAS MILITARY DEPARTMENT                         ',
    'TEXAS VETERANS COMMISSION                         ',
    'DEPARTMENT OF PUBLIC SAFETY                       ',
    'COMMISSION ON LAW ENFORCEMENT OFFICER STANDARDS AN',
    'OFFICE OF INJURED EMPLOYEE COUNSEL                ',
    'DEPARTMENT OF LICENSING AND REGULATION            ',
    'TEXAS DEPARTMENT OF INSURANCE                     ',
    'RAILROAD COMMISSION                               ',
    'BOARD OF PUBLIC ACCOUNTANCY                       ',
    'ALCOHOLIC BEVERAGE COMMISSION                     ',
]

class_titles = [
    'JUDGE, RETIRED                                    ',
    'GENERAL COUNSEL IV                                ',
    'CORREC  OFFICER IV                                ',
    'CURATOR III                                       ',
    'CURATOR IV                                        ',
    'EQUIPMENT MAINT TECH I                            '
]



# Streamlit UI
st.set_page_config(page_title="Texas Salary Predictor", layout="centered")
st.markdown(
    """
    <h2 style='text-align: center; color: green;'>💰 Texas Annual Salary Prediction 💰</h2>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h5 style='text-align: center;'>Enter details below to predict the annual salary.</h5>
    """,
    unsafe_allow_html=True
)


# Input fields
agency_name = st.selectbox("Agency Name", ["Please select the agency name"] + agency_names)
class_title = st.selectbox("Class Title", ["Please select the class title"] + class_titles)
ethnicity = st.selectbox("Ethnicity", ["Please select the Ethnicity"] + ["AM INDIAN", "ASIAN", "BLACK", "HISPANIC", "OTHER", "WHITE"])
gender = st.radio("Gender", ["Female", "Male"])
status = st.selectbox("Status", ["Please select the Stat"] + ["Full-time", "Part-time"])
hrly_rate = st.number_input("Hourly Rate ($)", min_value=0.0, format="%.2f")
hours_per_week = st.number_input("Hours per Week", min_value=0, max_value=168, value=40)
monthly = st.number_input("Monthly Salary ($)", min_value=0.0, format="%.2f")

# Date inputs
day = st.number_input("Day", min_value=1, max_value=31, step=1)
month = st.number_input("Month", min_value=1, max_value=12, step=1)
year = st.number_input("Year", min_value=1900, max_value=2100, step=1)

# Encoding categorical features
encoder = OneHotEncoder(handle_unknown='ignore')
categorical_data = pd.DataFrame([[agency_name, class_title, ethnicity, gender, status]],
                                columns=["AGENCY NAME", "CLASS TITLE", "ETHNICITY", "GENDER", "STATUS"])
categorical_encoded = encoder.fit_transform(categorical_data).toarray()

# Prepare input data
numerical_data = np.array([hrly_rate, hours_per_week, monthly, day, month, year]).reshape(1, -1)
input_data = np.concatenate((categorical_encoded, numerical_data), axis=1)

# Predict button
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button("Predict Salary"):
    try:
        # Make prediction
        predicted_salary = model.predict(input_data)[0]

        # Center the output text
        st.markdown(
            f"""
            <h5 style='text-align: center; color: green;'>Predicted Annual Salary: ${predicted_salary:,.2f}</h5>
            """, 
            unsafe_allow_html=True
        )
    except ValueError as e:
        st.markdown(
            f"<h4 style='text-align: center; color: red;'>Error: {e}</h4>",
            unsafe_allow_html=True
        )
st.markdown("</div>", unsafe_allow_html=True)
