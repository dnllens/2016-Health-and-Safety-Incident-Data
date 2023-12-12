import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('Random_forest.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

clf1 = data["model"]

le_site_type = data["le_site_type"]
le_Incident_area = data["le_Incident_area"]
le_Work_process = data["le_Work_process"]
le_Dropped_Object_incident = data["le_Dropped_Object_incident"]
le_High_Potential_Incident = data["le_High_Potential_Incident"]
le_Region = data["le_Region"]

def show_predict_page():
    st.title("INCIDENT PREDICTION APPLICATION")

    st.write("""## PLEASE PROVIDE THE NECESSARY INFORMATIONS BELOW""")

    Site_Type = (
        "Operation site", "Project site", "Others", "Survey phase"
    )

    Incident_area = (
        'Nacelle', 'Excavations & civil works', 'Administration',
    'Installation vessel – heavy installations (WTG, foundations, offshore substation)',
    'Foundation external (excluding boatlanding and TP)',
    'Harbour, quay and pontoons',
    'Helicopter hoisting and landing area', 'Access roads',
    'Public road/area', 'Office', 'Vessel – other', 'Yaw gear space',
    'Transition piece area', 'Turbine tower', 'Warehouse',
    'CTV (Crew transfer vessel)', 'Substation work and cable areas',
    'Turbine assembly area', 'Car park', 'Access ladders',
    'Boatlanding', 'Hub and blades', 'Turbine Tower', 'Workshop',
    'Foundation internal', 'Storage', 'Survey vessel',
    'Company vehicle', 'Accommodation platform',
    'Accommodation vessel', 'Tug',
    'Installation vessel – cables (array, export)',
    'Turbine/substation outside (not dedicated work areas)', 'Barge',
    'Diving vessel', 'Kitchen & canteen',
    'Other - if activities cannot be covered by one of the other work processes',
    'Other - if activities cannot be covered by one of the other incident area',
    'Staircase', 'Substation HV areas (>1000 V)',
    'Harbour, Quay and pontoons', 'SOV (Service operation vessel)'

    )

    Work_process = (
        'Working with hand tools/power tools', 'Manual handling',
        'O&M building maintenance', 'Office work', 'Maritime operations',
        'Working at heights', 'Transfer by helicopter',
        'Operating plant and machinery',
        'Vessel operation (including jack-ups and barges)',
        'Lifting operations',
        'Working on energized systems (electrical, hydraulical, pneumatic)',
        'Civil works',
        'Other - if activities cannot be covered by one of the other work processes',
        'Transfer by vessel', 'Vessel mobilization',
        'Working with chemicals and hazardous substances',
        'Transit (vessel)', 'Training/drills/team building events',
        'Business travels',
        'Surveys (geophysical, environmental, meteorological)',
        'Rigging/slinging', 'Working in confined spaces',
        'Catering/cleaning', 'Replacing major components', 'Hot works',
        'Diving operations'
    )

    Dropped_Object_incident = (
        'Yes', 'No'
    )

    High_Potential_Incident = (
        'Yes', 'No'
    )

    Region = (
        'UK', 'EU'
    )

    Site_Type = st.selectbox("What sites did the incident happen", Site_Type)
    Incident_area = st.selectbox("What Area did the Incident occur", Incident_area)
    Work_process = st.selectbox("What work process was going on", Work_process)
    Dropped_Object_incident = st.selectbox("Was it a dropped object incident", Dropped_Object_incident)
    High_Potential_Incident = st.selectbox("Was it a high potential incident", High_Potential_Incident)
    Region = st.selectbox("What region did the incident happen", Region)

    ok = st.button("Calculate Success Rate")

    if ok:
        J = np.array([[Site_Type, Incident_area, Work_process, Dropped_Object_incident, High_Potential_Incident, Region]])
        J[:, 0] = le_site_type.transform(J[:, 0])
        J[:, 1] = le_Incident_area.transform(J[:, 1])
        J[:, 2] = le_Work_process.transform(J[:, 2])
        J[:, 3] = le_Dropped_Object_incident.transform(J[:, 3])
        J[:, 4] = le_High_Potential_Incident.transform(J[:, 4])
        J[:, 5] = le_Region.transform(J[:, 5])
        J = J.astype(float)

        st.write(f'The transformed input array (J) is {J}')

        predictions = clf1.predict(J)
        prediction = predictions[0]

        if prediction == 'Serious incident':
            st.write(f' ## This was a Serious Incident')
        else:
            st.write(f"## This is a slight Incident ")








