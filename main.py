
import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

# loading the models
diabetes_model = pickle.load(open('./models/diabetes_trained_model.sav','rb'))

heart_disease_model = pickle.load(open('./models/trained_model.sav','rb'))

parkinsons_disease_model = pickle.load(open('./models/parkinsons_model.sav','rb'))

# side bar 

def main_app():
    with st.sidebar:
        selected = option_menu('Multiple Disease Predction',
                            ['Diabetes Prediction',
                                'Heart Disease Prediction',
                                'Parkinsons Disease Prediction'],
                                icons = ['activity','heart','person'],
                            default_index = 0)

    #diabetes prediction page

    # DIABETES PREDICTION

    if (selected == 'Diabetes Prediction'):
        def diabetes_function(input_data):
            if not input_data or '' in input_data:
                return "Please provide valid input for all fields."
        
            try:
                input_numpy = np.asarray(input_data, dtype=float)  # Convert input data to floats
                input_data_reshape = input_numpy.reshape(1, -1)
                prediction = diabetes_model.predict(input_data_reshape)
                if prediction[0] == 0:
                    return 'The person is not diabetic!'
                else:
                    return 'The person is diabetic!'
            except ValueError:
                return "Invalid input. Please provide numerical values for all fields."
            

        # function end

        st.title('Diabetes Prediction using ML')

        #geting input data from user

        col1,col2,col3 = st.columns(3)


        with col1: 
            Pregnancies = st.text_input('Number of pregnencies')
        with col1:    
            Glucose = st.text_input('Glucose Level')
        with col1:
            BloodPressure = st.text_input('Blood pressure Value')
        with col2:    
            SkinThickness = st.text_input('Skin Thickness')
        with col2:
            Insulin = st.text_input('Insulin Level')
        with col2:    
            BMI = st.text_input('Body-Mass Index Value')
        with col3:
            DiabetesPedigreeFunction = st.text_input('DPF')
        with col3:
            Age = st.text_input('Age Of the person')


        # input_array = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]


        # prediction
        diab_diagnosis = ''

        #button
        if st.button('Generate Result'):
            diab_diagnosis = diabetes_function([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])        

        st.success(diab_diagnosis)


    # HEART DISEASE PREDICTION


    if (selected == 'Heart Disease Prediction'):

        def heart_prediction(input_data):

            if not input_data or '' in input_data:
                return "Please provide valid input for all fields."

            try:
                input_data_numeric = [float(value) for value in input_data]
                input_numpy = np.asarray(input_data_numeric)
            
                # Reshape the input data for prediction
                input_data_reshape = input_numpy.reshape(1, -1)
            
                # Make the prediction
                prediction = heart_disease_model.predict(input_data_reshape)
            
                # Return prediction result
                if prediction[0] == 0:
                    return 'The person is not diabetic!'
                else:
                    return 'The person is diabetic!'
            except ValueError:
                return "Invalid input.Please provide numerical/floating values for all fields."

        st.title('Heart Disease Prediction Using ML')
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            age = (st.text_input('Enter the age of person'))
        with col1:    
            sex = st.text_input('Enter the gender')
        with col1:    
            cp = st.text_input('Cp value')
        with col2:    
            trestbps = st.text_input('trestbps value')
        with col2:    
            chol = st.text_input('Cholestrol Value')
        with col2:    
            fbs = st.text_input('FBS value')
        restecg = st.text_input('RestEcg Value')
        with col3:    
            thalach = st.text_input('Thalach Value')
        with col3:    
            exang = st.text_input('Exang value')
        with col3:    
            oldpeak = st.text_input('Oldpeak value')
        with col4:    
            slope = st.text_input('Slope Value')
        with col4:    
            ca = st.text_input('Ca Value')
        with col4:    
            thal = st.text_input('Thal value')
        
        #code for prediction 
        heart_diagnosis = ""
        
        #creating a button
        
        if st.button('GENERATE RESULT'):
            heart_diagnosis = heart_prediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]) 
        
        st.success(heart_diagnosis)

    if (selected == 'Parkinsons Disease Prediction'):

        def parkinson_prediction(input_data):

            if not input_data or '' in input_data:
                return "Please provide valid input for all fields."
            

            try:
                input_array = np.asarray(input_data,dtype=float)

                input_data_reshape=input_array.reshape(1,-1)
                x = parkinsons_disease_model.predict(input_data_reshape)
                if x[0] == 1:
                    return "You have parkinsons"
                else:
                    return "You don't have parkisons"
            except ValueError:
                return "Invalid input. Please provide numerical values for all fields."

        st.title('Parkinsons Disease Prediction')

        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            Fo = st.text_input('MDVP : Fo(Hz)')
        with col2:
            Fhi = st.text_input('MDVP : Fhi(Hz)')
        with col3:
            Flo = st.text_input('MDVP : Flo(Hz)')
        with col4:
            Jitter_percent = st.text_input('MDVP : Jitter(%)')
        with col5:    
            Jitter_abs = st.text_input('MDVP : Jitter(Abs)')
        with col1:    
            RAP = st.text_input('MDVP : RAP')
        with col2:
            PPQ = st.text_input('MDVP : PPQ')
        with col3:
            DDP = st.text_input('Jitter : DDP')
        with col4:
            shimmer = st.text_input('MDVP : Shimmer')
        with col5:
            shimmer_dB = st.text_input('MDVP : Shimmer(dB)')
        with col1:
            APQ3 = st.text_input('Shimmer : APQ3')
        with col2:
            APQ5 = st.text_input('Shimmer : APQ5')
        with col3:
            APQ = st.text_input('MDVP : APQ')
        with col4:
            DDA = st.text_input('Shimmer : DDA')
        with col5:
            NHR = st.text_input('NHR')
        with col1:
            HNR = st.text_input('HNR')
        with col2:
            RPDE = st.text_input('RPDE')
        with col3:
            DFA = st.text_input('DFA')
        with col4:
            spread1 = st.text_input('spread1')
        with col5:
            spread2 = st.text_input('spread2')
        with col1:
            D2 = st.text_input('D2')
        with col2:    
            PPE = st.text_input('PPE')

        parkinson_diagnosis = ""
        
        #creating a button
        
        if st.button('GENERATE RESULT'):
            parkinson_diagnosis = parkinson_prediction([Fo,Fhi,Flo,Jitter_percent,Jitter_abs,RAP,PPQ,DDP,shimmer,shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]) 
        
        st.success(parkinson_diagnosis)



if __name__=="__main__":
    main_app()