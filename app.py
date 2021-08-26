import streamlit as st
import joblib
import numpy as np

# Loading the saved Model
model = joblib.load('randomforest_model.pkl')


def predict_default(features):
    features = np.array(features).astype(np.float64).reshape(1, -1)

    prediction = model.predict(features)
    probability = model.predict_proba(features)

    return prediction, probability

def main():

    html_temp = """
            <div style = "background-color: tomato; padding: 10px;">
                <center><h1>Default Loan Prediction</h1></center>
            </div><br>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    loan_amnt = st.text_input("Loan Amount(USD Dollar)")

    term  = ["36", "60"] 

    Term = term.index(st.selectbox(
        "Select term",
        tuple(term)
    ))+1

    int_rate = st.text_input("Interest Rate")

    employment_length = st.text_input("How many Years you are employed.")

    annual_income = st.text_input("Annual Income(dollars)")

    purpose_detail = ["Debt consolidation", "Small business", "Home improvement", "Shopping", "Credit card", "House",
               "Vacation", "Car", "Wedding", "Education", "Medical", "Moving"]

    purpose = purpose_detail.index(st.selectbox(
        "Select Purpose",
        tuple(purpose_detail)
    ))+1

    dti = st.text_input("Debt to income ratio(should be less than 43)")

    delinq2yr = st.text_input("How many times you were behind on loan payment(in months)")

    credit_score_low = st.text_input("FICO Score lowest(Not less than 540)")

    open_acc = st.text_input("Open credit lines")

    pub_rec = st.text_input("Number of public derogatory")

    revol_bal = st.text_input("total credit revolving balance(dollars)")

    revol_util = st.text_input("revolving utilization rate(between 1-100)")

    total_acc = st.text_input("Total Credit lines")
        
    tot_cur_bal = st.text_input("Total Current Balance(Dollars)")

    mortage_acc = st.text_input("How many mortage account You have")

    if st.button("Predict"):

        features = [loan_amnt, Term, int_rate, employment_length, annual_income, purpose, dti, delinq2yr,
                    credit_score_low, open_acc, pub_rec, revol_bal,
                    revol_util, total_acc, tot_cur_bal, mortage_acc]
        prediction, probability = predict_default(features)
        # print(prediction)
        # print(probability[:,1][0])
        if prediction[0] == 0:
            # counselling_html = """
            #     <div style = "background-color: #f8d7da; font-weight:bold;padding:10px;border-radius:7px;">
            #         <p style = 'color: #721c24;'>This account will be defaulted with a probability of {round(np.max(probability)*100, 2))}%.</p>
            #     </div>
            # """
            # st.markdown(counselling_html, unsafe_allow_html=True)

            st.success(
                "This loan will be defaulted with a probability of {}%.".format(round(np.max(probability) * 100, 2)))

        else:
            st.success("This loan will not be defaulted with a probability of {}%.".format(
                round(np.max(probability) * 100, 2)))

if __name__ == '__main__':
        main()


    


