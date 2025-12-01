import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    model_info = joblib.load('health_risk_model.pkl')
    return model_info

def prepare_input(inputs, model_info):
    """Helper function to prepare input dataframe for prediction"""
    input_df = pd.DataFrame([inputs])
    
    # CREATE THE HAS_FEVER FEATURE
    fever_info = model_info['feature_engineer_info']
    input_df['Has_Fever'] = (
        (input_df['Temperature'] > fever_info['has_fever_temp_threshold']) &
        (input_df['Heart_Rate'] > fever_info['has_fever_hr_threshold'])
    ).astype(int)
    
    # Encode categorical variables
    input_df = pd.get_dummies(input_df, columns=['Consciousness', 'On_Oxygen'], drop_first=False, dtype=int)
    
    # Rename On_Oxygen columns to match what the model expects
    if 'On_Oxygen_No' in input_df.columns:
        input_df = input_df.rename(columns={'On_Oxygen_No': 'On_Oxygen_0', 'On_Oxygen_Yes': 'On_Oxygen_1'})
    
    # Ensure all expected Consciousness columns exist
    for col in ['Consciousness_A', 'Consciousness_C', 'Consciousness_P', 'Consciousness_U', 'Consciousness_V']:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Ensure both On_Oxygen columns exist
    for col in ['On_Oxygen_0', 'On_Oxygen_1']:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Get expected features
    original_features = model_info['feature_names']
    
    # Reorder columns to match model expectations
    input_df = input_df[original_features]
    
    # Convert all columns to float64
    input_df = input_df.astype('float64')
    
    return input_df

model_info = load_model()
model = model_info['model']

st.title(':violet[Real World Patient Simulation]')

# Initialize session state for prediction results
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

inputs = {
    'Respiratory_Rate': st.slider('Respiratory Rate', 0.0, 30.0, 12.0),
    'Oxygen_Saturation': st.slider('Oxygen Saturation', 60.0, 100.0, 74.0),
    'O2_Scale': st.slider('O2 Scale', 0.0, 2.0, 1.0, step=1.0),
    'Systolic_BP': st.slider('Systolic Blood Pressure', 40.0, 160.0, 100.0),
    'Heart_Rate': st.slider('Heart Rate', 50.0, 170.0, 95.0),
    'Temperature': st.slider('Temperature °F', 90.0, 120.0, 98.0),
    'Consciousness': st.selectbox('Consciousness Level', ['A', 'C', 'P', 'U', 'V']),
    'On_Oxygen': st.selectbox('Enter whether they are on oxygen', ["No", 'Yes'])
}

input_df = prepare_input(inputs, model_info)

# Color mapping for risk levels
risk_colors = {
    "High": "red",
    "Medium": "orange",
    "Low": "blue",
    "Normal": "green"
}

if st.button("Predict Patient Risk Level"):
    prediction_encoded = model.predict(input_df)
    
    # Decode if using XGB with label encoder
    if model_info['label_encoder'] is not None:
        prediction = model_info['label_encoder'].inverse_transform(prediction_encoded)
    else:
        prediction = prediction_encoded
    
    proba = model.predict_proba(input_df)
    
    # Store results in session state
    st.session_state.prediction_made = True
    st.session_state.prediction = prediction[0]
    st.session_state.proba = proba[0]
    st.session_state.inputs = inputs.copy()

# Display results if prediction has been made
if st.session_state.prediction_made:
    predicted_risk_level = st.session_state.prediction
    
    st.subheader(f"Predicted Risk Level: :{risk_colors[predicted_risk_level]}[{predicted_risk_level}]")
    
    st.divider()
    
    # RISK TRAJECTORY SIMULATION
    st.subheader("Risk Trajectory Simulation")
    st.write("Adjust vital signs below to see how interventions might change the patient's risk level:")
    
    # Create tabs for different simulation modes
    tab1, tab2 = st.tabs(["Manual Adjustment", "Intervention Scenarios"])
    
    with tab1:
        st.write("**Simulate changes in vital signs:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current Risk Level", predicted_risk_level, 
                     delta=None, delta_color="off")
            
            st.write("**Current Vitals:**")
            st.write(f"- Respiratory Rate: {st.session_state.inputs['Respiratory_Rate']}")
            st.write(f"- Oxygen Saturation: {st.session_state.inputs['Oxygen_Saturation']}%")
            st.write(f"- Heart Rate: {st.session_state.inputs['Heart_Rate']}")
            st.write(f"- Temperature: {st.session_state.inputs['Temperature']}°F")
            st.write(f"- Systolic BP: {st.session_state.inputs['Systolic_BP']}")
        
        with col2:
            st.write("**Adjust vitals:**")
            
            sim_inputs = {
                'Respiratory_Rate': st.slider('Sim: Respiratory Rate', 0.0, 30.0, st.session_state.inputs['Respiratory_Rate'], key='sim_rr'),
                'Oxygen_Saturation': st.slider('Sim: Oxygen Saturation', 60.0, 100.0, st.session_state.inputs['Oxygen_Saturation'], key='sim_o2'),
                'O2_Scale': st.slider('Sim: O2 Scale', 0.0, 2.0, st.session_state.inputs['O2_Scale'], step=1.0, key='sim_o2scale'),
                'Systolic_BP': st.slider('Sim: Systolic BP', 40.0, 160.0, st.session_state.inputs['Systolic_BP'], key='sim_bp'),
                'Heart_Rate': st.slider('Sim: Heart Rate', 50.0, 170.0, st.session_state.inputs['Heart_Rate'], key='sim_hr'),
                'Temperature': st.slider('Sim: Temperature °F', 90.0, 120.0, st.session_state.inputs['Temperature'], key='sim_temp'),
                'Consciousness': st.selectbox('Sim: Consciousness', ['A', 'C', 'P', 'U', 'V'], 
                                             index=['A', 'C', 'P', 'U', 'V'].index(st.session_state.inputs['Consciousness']), key='sim_cons'),
                'On_Oxygen': st.selectbox('Sim: On Oxygen', ["No", 'Yes'], 
                                         index=["No", 'Yes'].index(st.session_state.inputs['On_Oxygen']), key='sim_onO2')
            }
            
            sim_input_df = prepare_input(sim_inputs, model_info)
            sim_prediction_encoded = model.predict(sim_input_df)
            
            # Decode if using label encoder
            if model_info['label_encoder'] is not None:
                sim_prediction = model_info['label_encoder'].inverse_transform(sim_prediction_encoded)
            else:
                sim_prediction = sim_prediction_encoded
            
            sim_proba = model.predict_proba(sim_input_df)
            sim_risk_level = sim_prediction[0]
            
            # Show simulated risk with comparison
            if sim_risk_level != predicted_risk_level:
                # Determine if risk got better or worse for delta color
                risk_order = {"Low": 0, "Normal": 1, "Medium": 2, "High": 3}
                is_improvement = risk_order.get(sim_risk_level, 2) < risk_order.get(predicted_risk_level, 2)
                
                st.metric("Simulated Risk Level", sim_risk_level, 
                         delta=f"Changed from {predicted_risk_level}",
                         delta_color="inverse" if is_improvement else "normal")
            else:
                st.metric("Simulated Risk Level", sim_risk_level, 
                         delta="No change")
            
            # Show what changed
            changes = []
            if sim_inputs['Respiratory_Rate'] != st.session_state.inputs['Respiratory_Rate']:
                changes.append(f"RR: {st.session_state.inputs['Respiratory_Rate']} → {sim_inputs['Respiratory_Rate']}")
            if sim_inputs['Oxygen_Saturation'] != st.session_state.inputs['Oxygen_Saturation']:
                changes.append(f"O2 Sat: {st.session_state.inputs['Oxygen_Saturation']}% → {sim_inputs['Oxygen_Saturation']}%")
            if sim_inputs['Heart_Rate'] != st.session_state.inputs['Heart_Rate']:
                changes.append(f"HR: {st.session_state.inputs['Heart_Rate']} → {sim_inputs['Heart_Rate']}")
            if sim_inputs['Temperature'] != st.session_state.inputs['Temperature']:
                changes.append(f"Temp: {st.session_state.inputs['Temperature']}°F → {sim_inputs['Temperature']}°F")
            if sim_inputs['Systolic_BP'] != st.session_state.inputs['Systolic_BP']:
                changes.append(f"BP: {st.session_state.inputs['Systolic_BP']} → {sim_inputs['Systolic_BP']}")
            
            if changes:
                st.write("**Changes made:**")
                for change in changes:
                    st.write(f"- {change}")
    
    with tab2:
        st.write("**Common clinical interventions:**")
        
        scenario = st.selectbox("Select intervention scenario:", [
            "Oxygen supplementation",
            "Fever reduction (antipyretics)",
            "Fluid resuscitation (low BP)",
            "Respiratory support",
            "Combined: O2 + fever control"
        ])
        
        scenario_inputs = st.session_state.inputs.copy()
        
        if scenario == "Oxygen supplementation":
            st.info("Simulating supplemental oxygen administration")
            scenario_inputs['Oxygen_Saturation'] = min(100.0, st.session_state.inputs['Oxygen_Saturation'] + 8)
            scenario_inputs['On_Oxygen'] = 'Yes'
            scenario_inputs['O2_Scale'] = min(2.0, st.session_state.inputs['O2_Scale'] + 1.0)
            
        elif scenario == "Fever reduction (antipyretics)":
            st.info("Simulating antipyretic medication (e.g., acetaminophen)")
            scenario_inputs['Temperature'] = max(98.0, st.session_state.inputs['Temperature'] - 2.0)
            scenario_inputs['Heart_Rate'] = max(60.0, st.session_state.inputs['Heart_Rate'] - 10)
            
        elif scenario == "Fluid resuscitation (low BP)":
            st.info("Simulating IV fluid administration")
            scenario_inputs['Systolic_BP'] = min(140.0, st.session_state.inputs['Systolic_BP'] + 15)
            scenario_inputs['Heart_Rate'] = max(60.0, st.session_state.inputs['Heart_Rate'] - 5)
            
        elif scenario == "Respiratory support":
            st.info("Simulating respiratory support (bronchodilators, positioning)")
            scenario_inputs['Respiratory_Rate'] = max(12.0, st.session_state.inputs['Respiratory_Rate'] - 4)
            scenario_inputs['Oxygen_Saturation'] = min(100.0, st.session_state.inputs['Oxygen_Saturation'] + 5)
            
        elif scenario == "Combined: O2 + fever control":
            st.info("Simulating both oxygen supplementation and fever reduction")
            scenario_inputs['Oxygen_Saturation'] = min(100.0, st.session_state.inputs['Oxygen_Saturation'] + 8)
            scenario_inputs['On_Oxygen'] = 'Yes'
            scenario_inputs['Temperature'] = max(98.0, st.session_state.inputs['Temperature'] - 2.0)
            scenario_inputs['Heart_Rate'] = max(60.0, st.session_state.inputs['Heart_Rate'] - 10)
        
        scenario_df = prepare_input(scenario_inputs, model_info)
        scenario_prediction_encoded = model.predict(scenario_df)
        
        # Decode if using label encoder
        if model_info['label_encoder'] is not None:
            scenario_prediction = model_info['label_encoder'].inverse_transform(scenario_prediction_encoded)
        else:
            scenario_prediction = scenario_prediction_encoded
            
        scenario_risk = scenario_prediction[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Before Intervention", predicted_risk_level)
        with col2:
            if scenario_risk != predicted_risk_level:
                # Determine if risk got better or worse
                risk_order = {"Low": 0, "Normal": 1, "Medium": 2, "High": 3}
                is_improvement = risk_order.get(scenario_risk, 2) < risk_order.get(predicted_risk_level, 2)
                
                st.metric("After Intervention", scenario_risk,
                         delta=f"Changed from {predicted_risk_level}",
                         delta_color="inverse" if is_improvement else "normal")
            else:
                st.metric("After Intervention", scenario_risk, delta="No change")
