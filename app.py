import streamlit as st
import numpy as np
import pandas as pd
import pickle as pk

trained_model = pk.load(open('XGBreg_pipeline.pkl', 'rb'))


def MBS_Predict(input_data):
    # Assuming you have the correct column names, replace them accordingly
    columns = ['FirstPayment_Month', 'IsFirstTimeHomebuyer', 'Maturity_Month', 'MIP', 'Occupancy',
               'DTI', 'OrigUPB', 'OrigInterestRate', 'Channel', 'PPM',
               'PropertyState', 'PropertyType', 'LoanPurpose',
               'OrigLoanTerm', 'NumBorrowers', 'MonthsDelinquent',
               'Credit_range', 'LTV_range', 'Repay_range']
    input_df = pd.DataFrame([input_data], columns=columns)
    pred = trained_model.predict(input_df)
    print("Shape of input_arr:", input_df.shape)
    return pred


def main():
    st.title('Prepay Precision: Predicting MBS Loan Risks')
    st.markdown('Forecasting Mortgage-Backed Securities Loan Prepayment Risk: A Predictive Modeling Web App')

    month_mapping = {
        '': 0,
        'January (01)': 1,
        'February (02)': 2,
        'March (03)': 3,
        'April (04)': 4,
        'May (05)': 5,
        'June (06)': 6,
        'July (07)': 7,
        'August (08)': 8,
        'September (09)': 9,
        'October (10)': 10,
        'November (11)': 11,
        'December (12)': 12
    }

    yn_mapping = {
        '': -1,
        'No': 0,
        'Yes': 1
    }

    o_mapping = {
        '': -1,
        'OTHER': 0,
        'Investment Property': 1,
        'Second Home': 2
    }

    c_mapping = {
        '': -1,
        'Third Party': 0,
        'Retail': 1,
        'Correspondent': 2,
        'Broker': 3
    }

    ps_mapping = {
        '': -1,
        'Midwest': 0,
        'Others': 1,
        'West Coast': 2,
        'Northeast': 3,
        'South': 4
    }

    pt_mapping = {
        '': -1,
        'Single Family': 0,
        'Planned Unit': 1,
        'Condominium': 2,
        'Manufactured Home': 3,
        'Cooperative': 4,
        'Leasehold': 5
    }

    lp_mapping = {
        '': -1,
        'Purchase': 0,
        'Refinance': 1,
        'Cash-out Refinance': 2
    }

    nb_mapping = {
        '': -1,
        'Single Individual': 1,
        'Joint Application': 2
    }

    cr_mapping = {
        '': -1,
        'Poor': 1,
        'Fair': 2,
        'Good': 3,
        'Excellent': 4
    }

    ltv_mapping = {
        '': -1,
        'Low': 0,
        'Medium': 1,
        'High': 2
    }

    rr_mapping = {
        '': -1,
        '0-4': 1,
        '4-8': 2,
        '8-12': 3,
        '12-16': 4,
        '16-20': 5
    }

    fpm_ptr = st.selectbox('First Payment Month',
                           ['', 'January (01)', 'February (02)', 'March (03)', 'April (04)', 'May (05)', 'June (06)',
                            'July (07)', 'August (08)', 'September (09)', 'October (10)', 'November (11)',
                            'December (12)'])
    fpm = month_mapping[fpm_ptr]

    ifthb_ptr = st.selectbox('First Time Home Buyer', ['', 'Yes', 'No'])
    fthb = yn_mapping[ifthb_ptr]

    mm_ptr = st.selectbox('Maturity Month',
                          ['', 'January (01)', 'February (02)', 'March (03)', 'April (04)', 'May (05)', 'June (06)',
                           'July (07)', 'August (08)', 'September (09)', 'October (10)', 'November (11)',
                           'December (12)'])
    mm = month_mapping[mm_ptr]

    mip = st.text_input('MIP')

    o_ptr = st.selectbox('Occupancy',
                         ['', 'OTHER', 'Second Home', 'Investment Property'])
    o = o_mapping[o_ptr]

    dti = st.text_input('DTI')

    origupb = st.text_input('Original Unpaid Balance')

    OIR = st.text_input('Original Interest Rate')

    channel = st.selectbox('Channel', ['', 'Third Party', 'Retail', 'Correspondent', 'Broker'])
    c = c_mapping[channel]

    ppm_ptr = st.selectbox('PPM', ['', 'Yes', 'No'])
    ppm = yn_mapping[ppm_ptr]

    ps_ptr = st.selectbox('Property Ownership State', ['', 'Midwest', 'Others', 'West Coast', 'Northeast', 'South'])
    ps = ps_mapping[ps_ptr]

    pt_ptr = st.selectbox('Property Type',
                          ['', 'Single Family', 'Planned Unit', 'Condominium', 'Manufactured Home', 'Cooperative',
                           'Leasehold'])
    pt = pt_mapping[pt_ptr]

    lp_ptr = st.selectbox('Loan Purpose : ', ['', 'Purchase', 'Refinance', 'Cash-out Refinance'])
    lp = lp_mapping[lp_ptr]

    origlterm = st.text_input('Original Loan Term')

    nb_ptr = st.selectbox('Total Borrowers yet to Repay', ['', 'Single Individual', 'Joint Application'])
    nb = nb_mapping[nb_ptr]

    md = st.text_input('Missed Payment Duration (Months)')

    cr_ptr = st.selectbox('Credit Status', ['', 'Poor', 'Fair', 'Good', 'Excellent'])
    cr = cr_mapping[cr_ptr]

    ltv_ptr = st.selectbox('Loan-to-Value Range : ', ['', 'Low', 'Medium', 'High'])
    ltv = ltv_mapping[ltv_ptr]

    rr_ptr = st.selectbox('Repayment Category Bins', ['', '0-4', '4-8', '8-12', '12-16', '16-20'])
    rr = rr_mapping[rr_ptr]

    if st.button("Predict"):
        predict = MBS_Predict([fpm, fthb, mm, mip, o, dti, origupb, OIR, c, ppm, ps, pt, lp,
                               origlterm, nb, md, cr, ltv, rr])
        st.success(predict)


if __name__ == '__main__':
    main()