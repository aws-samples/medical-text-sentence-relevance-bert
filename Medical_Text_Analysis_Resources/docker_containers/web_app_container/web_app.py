import streamlit as st
import time
import json
import boto3
import os
import requests
import pandas

def predict(statement,input_paragraph,ip_address):

    ip_address=ip_address
    headers = {
        'Content-Type': 'application/json',
    }

    data = {"input_sentence":statement,"input_paragraph":input_paragraph}
    cmd=f'''http://{ip_address}:80/invocations'''

    data_2=json.dumps(data)

    response = requests.post(cmd, headers=headers, data=data_2) 
    body_str = response.text
    return(body_str)

def main(model_ip=None):
    st.markdown("""
    # Healthcare Sentence Relevance.   
     """) 

    about_button = st.sidebar.button("About")
    demo_button = st.sidebar.button("Demo")

    if about_button == True:
        st.markdown("""
        ### Background
        This demo uses a BERT based model to determine the relevence of a summary statement in regards to the document it is summarizing.
        This model is used to produce semantically meaningful sentence embeddings that can be used to determine 
        how similar the summary statement is to each of the sentences that make up the corresponding document. The sample text is taken from https://www.mtsamples.com/
        
        ### How It Works
        Given a statement and the corresponding document, 
        the model will attempt to go through the document to find the top 5 sentences that are the closest related to the statement. 
        In some cases, this functionality can serve as a proxy to determine sentence correctness as well.

        ### How To Use
        First, select whether you would like to write your own statement and document or if you would rather test a preloaded sample.
        After you submit the statement and corresponding document, 
        the sentences will be embedded and the distance between the input statement and each sentence within the document will be calculated. 
        The top 5 sentences from the document that have the minimum distance to the input statement will be returned.

        """)
    else:
        st.markdown("""
        This demo uses a BERT based model to determine the relevence of a summary statement in regards to the document it is summarizing.
        """)
        radio_button = st.radio("",("Write my own","Test a sample"))

        options_1=[
            "",
            "A 23-year-old white female presents with complaint of allergies",
            "Consult for laparoscopic gastric bypass."
                ]

        options=[
            "",
            ("SUBJECTIVE:,  This 23-year-old white female presents with complaint of allergies.  She used to have allergies when she lived in Seattle but she thinks they are worse here.  In the past, she has tried Claritin, and Zyrtec.  Both worked for short time but then seemed to lose effectiveness.  She has used Allegra also.  She used that last summer and she began using it again two weeks ago.  It does not appear to be working very well.  She has used over-the-counter sprays but no prescription nasal sprays.  She does have asthma but doest not require daily medication for this and does not think it is flaring up.,MEDICATIONS: , Her only medication currently is Ortho Tri-Cyclen and the Allegra.,ALLERGIES: , She has no known medicine allergies.,OBJECTIVE:,Vitals:  Weight was 130 pounds and blood pressure 124/78.,HEENT:  Her throat was mildly erythematous without exudate.  Nasal mucosa was erythematous and swollen.  Only clear drainage was seen.  TMs were clear.,Neck:  Supple without adenopathy.,Lungs:  Clear.,ASSESSMENT:,  Allergic rhinitis.,PLAN:,1.  She will try Zyrtec instead of Allegra again.  Another option will be to use loratadine.  She does not think she has prescription coverage so that might be cheaper.,2.  Samples of Nasonex two sprays in each nostril given for three weeks.  A prescription was written as well."),
            ("PAST MEDICAL HISTORY:, He has difficulty climbing stairs, difficulty with airline seats, tying shoes, used to public seating, and lifting objects off the floor.  He exercises three times a week at home and does cardio.  He has difficulty walking two blocks or five flights of stairs.  Difficulty with snoring.  He has muscle and joint pains including knee pain, back pain, foot and ankle pain, and swelling.  He has gastroesophageal reflux disease.,PAST SURGICAL HISTORY:, Includes reconstructive surgery on his right hand 13 years ago.  ,SOCIAL HISTORY:, He is currently single.  He has about ten drinks a year.  He had smoked significantly up until several months ago.  He now smokes less than three cigarettes a day.,FAMILY HISTORY:, Heart disease in both grandfathers, grandmother with stroke, and a grandmother with diabetes.  Denies obesity and hypertension in other family members.,CURRENT MEDICATIONS:, None.,ALLERGIES:,  He is allergic to Penicillin.,MISCELLANEOUS/EATING HISTORY:, He has been going to support groups for seven months with Lynn Holmberg in Greenwich and he is from Eastchester, New York and he feels that we are the appropriate program.  He had a poor experience with the Greenwich program.  Eating history, he is not an emotional eater.  Does not like sweets.  He likes big portions and carbohydrates.  He likes chicken and not steak.  He currently weighs 312 pounds.  Ideal body weight would be 170 pounds.  He is 142 pounds overweight.  If ,he lost 60% of his excess body weight that would be 84 pounds and he should weigh about 228.,REVIEW OF SYSTEMS: ,Negative for head, neck, heart, lungs, GI, GU, orthopedic, and skin.  Specifically denies chest pain, heart attack, coronary artery disease, congestive heart failure, arrhythmia, atrial fibrillation, pacemaker, high cholesterol, pulmonary embolism, high blood pressure, CVA, venous insufficiency, thrombophlebitis, asthma, shortness of breath, COPD, emphysema, sleep apnea, diabetes, leg and foot swelling, osteoarthritis, rheumatoid arthritis, hiatal hernia, peptic ulcer disease, gallstones, infected gallbladder, pancreatitis, fatty liver, hepatitis, hemorrhoids, rectal bleeding, polyps, incontinence of stool, urinary stress incontinence, or cancer.  Denies cellulitis, pseudotumor cerebri, meningitis, or encephalitis.,PHYSICAL EXAMINATION:, He is alert and oriented x 3.  Cranial nerves II-XII are intact.  Afebrile.  Vital Signs are stable."),
            ("HISTORY OF PRESENT ILLNESS: , I have seen ABC today.  He is a very pleasant gentleman who is 42 years old, 344 pounds.  He is 5\'9\'.  He has a BMI of 51.  He has been overweight for ten years since the age of 33, at his highest he was 358 pounds, at his lowest 260.  He is pursuing surgical attempts of weight loss to feel good, get healthy, and begin to exercise again.  He wants to be able to exercise and play volleyball.  Physically, he is sluggish.  He gets tired quickly.  He does not go out often.  When he loses weight he always regains it and he gains back more than he lost.  His biggest weight loss is 25 pounds and it was three months before he gained it back.  He did six months of not drinking alcohol and not taking in many calories.  He has been on multiple commercial weight loss programs including Slim Fast for one month one year ago and Atkin\'s Diet for one month two years ago.,PAST MEDICAL HISTORY: , He has difficulty climbing stairs, difficulty with airline seats, tying shoes, used to public seating, difficulty walking, high cholesterol, and high blood pressure.  He has asthma and difficulty walking two blocks or going eight to ten steps.  He has sleep apnea and snoring.  He is a diabetic, on medication.  He has joint pain, knee pain, back pain, foot and ankle pain, leg and foot swelling.  He has hemorrhoids.,PAST SURGICAL HISTORY: , Includes orthopedic or knee surgery.,SOCIAL HISTORY: , He is currently single.  He drinks alcohol ten to twelve drinks a week, but does not drink five days a week and then will binge drink.  He smokes one and a half pack a day for 15 years, but he has recently stopped smoking for the past two weeks.,FAMILY HISTORY: , Obesity, heart disease, and diabetes.  Family history is negative for hypertension and stroke.,CURRENT MEDICATIONS:,  Include Diovan, Crestor, and Tricor.,MISCELLANEOUS/EATING HISTORY:  ,He says a couple of friends of his have had heart attacks and have had died.  He used to drink everyday, but stopped two years ago.  He now only drinks on weekends.  He is on his second week of Chantix, which is a medication to come off smoking completely.  Eating, he eats bad food.  He is single.  He eats things like bacon, eggs, and cheese, cheeseburgers, fast food, eats four times a day, seven in the morning, at noon, 9 p.m., and 2 a.m.  He currently weighs 344 pounds and 5\'9\".  His ideal body weight is 160 pounds.  He is 184 pounds overweight.  If he lost 70% of his excess body weight that would be 129 pounds and that would get him down to 215.,REVIEW OF SYSTEMS: , Negative for head, neck, heart, lungs, GI, GU, orthopedic, or skin.  He also is positive for gout.  He denies chest pain, heart attack, coronary artery disease, congestive heart failure, arrhythmia, atrial fibrillation, pacemaker, pulmonary embolism, or CVA.  He denies venous insufficiency or thrombophlebitis.  Denies shortness of breath, COPD, or emphysema.  Denies thyroid problems, hip pain, osteoarthritis, rheumatoid arthritis, GERD, hiatal hernia, peptic ulcer disease, gallstones, infected gallbladder, pancreatitis, fatty liver, hepatitis, rectal bleeding, polyps, incontinence of stool, urinary stress incontinence, or cancer.  He denies cellulitis, pseudotumor cerebri, meningitis, or encephalitis.,PHYSICAL EXAMINATION:  ,He is alert and oriented x 3.  Cranial nerves II-XII are intact.  Neck is soft and supple.  Lungs:  He has positive wheezing bilaterally.  Heart is regular rhythm and rate.  His abdomen is soft.  Extremities:  He has 1+ pitting edema.,IMPRESSION/PLAN:,  I have explained to him the risks and potential complications of laparoscopic gastric bypass in detail and these include bleeding, infection, deep venous thrombosis, pulmonary embolism, leakage from the gastrojejuno-anastomosis, jejunojejuno-anastomosis, and possible bowel obstruction among other potential complications.  He understands.  He wants to proceed with workup and evaluation for laparoscopic Roux-en-Y gastric bypass.  He will need to get a letter of approval from Dr. XYZ.  He will need to see a nutritionist and mental health worker.  He will need an upper endoscopy by either Dr. XYZ.  He will need to go to Dr. XYZ as he previously had a sleep study.  We will need another sleep study.  He will need H. pylori testing, thyroid function tests, LFTs, glycosylated hemoglobin, and fasting blood sugar.  After this is performed, we will submit him for insurance approval.")
                ]

        if radio_button == "Test a sample":
            prompt_3 = st.selectbox(label='Select Sample Input Statement:', options=options_1)
            assert isinstance(prompt_3, str)
            sample_index = options_1.index(prompt_3)
            if sample_index != 0:
                prompt_4 = st.text_area(
                    label='Corresponding Sample Document:',
                    value= options[sample_index]
                )
                assert isinstance(prompt_4, str)
        else:
            prompt_1 = st.text_area(
                label='Write a statement:'
            )
            assert isinstance(prompt_1, str)

            prompt_2 = st.text_area(
                label='Write a corresponding document:'
            
            )
            assert isinstance(prompt_2, str)

        to_submit=st.button('Submit')
        print(to_submit)

        if to_submit==True and radio_button == "Test a sample": 
            if prompt_3 and prompt_4:
                with st.spinner('Calculating Sentence Similarities...'):
                    text = predict(
                        prompt_3, prompt_4,model_ip
                    )
            the_df=pandas.read_json(text)
            st.markdown("""
                # Results
            """)
            st.dataframe(the_df,width=700)
        elif to_submit==True and radio_button == "Write my own":
            if prompt_1 and prompt_2:
                with st.spinner('Calculating Sentence Similarities...'):
                    text = predict(
                        prompt_1, prompt_2,model_ip
                    )
            the_df=pandas.read_json(text)
            st.markdown("""
                # Results
            """)
            st.dataframe(the_df)

def check_password():
    password = st.text_input("Enter password",type="password")
    #to_submit=st.button('Submit')
    #print(password)
    flag=0
    #print(flag)
    true_pass=os.environ["WEB_APP_PASSWORD"]
    #print(true_pass)
    if password != true_pass:
        #st.error("the password you entered is incorrect")
        flag=1
    #print(flag)
    return(flag)


if __name__ == "__main__":
    debug = os.getenv('DASHBOARD_DEBUG', 'false') == 'true'
    if debug:
        #main()
        pass
    else:
        try:
            result=check_password() #uncomment to remove password protection
            #print(result)
            #result=0
            if result==0:
                model_ip=os.environ["MODEL_IP"]
                main(model_ip=model_ip)
        except Exception as e:
            st.error('Internal error occurred.')

