import streamlit as st
import re
import pandas as pd
import numpy as np
import textstat
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings
import joblib
from transformers import AutoTokenizer 
warnings.filterwarnings('ignore')


try:
    from googletrans import Translator
    translator = Translator()
    st.session_state.googletrans_available = True
except ImportError:
    st.session_state.googletrans_available = False
    st.warning("`googletrans` library not found. Language translation of T5 output will be skipped. Please install it: `pip install googletrans==3.1.0a0`")

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Medical Text Simplifier", layout="wide")

# -------------------------------
# Session State Initialization
# -------------------------------
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'googletrans_available' not in st.session_state: 
    st.session_state.googletrans_available = False


@st.cache_data
def load_local_dataset():
    """Load Med-EASi dataset from local folder after downloading via huggingface-cli"""
    try:
        import pandas as pd
        import os

        data_dir = "./med-easi"
        train_path = os.path.join(data_dir, "train.csv")

        if not os.path.exists(train_path):
            st.error("❌ Local dataset not found! Please download using: huggingface-cli download cbasu/Med-EASi")
            return None

        with st.spinner("📥 Loading Med-EASi dataset from local folder..."):
            df = pd.read_csv(train_path)

            
            if 'Expert' in df.columns and 'Simple' in df.columns:
                df = df[['Expert', 'Simple']].rename(
                    columns={'Expert': 'original_text', 'Simple': 'simplified_text'}
                )
            else:
                st.error(f"⚠️ Could not find 'Expert' and 'Simple' columns. Found: {list(df.columns)}")
                return None

            df = df.dropna()
            st.success(f"✅ Loaded local dataset with {len(df)} samples!")
            return df

    except Exception as e:
        st.error(f"❌ Error loading local dataset: {str(e)}")
        return None


@st.cache_data
def create_sample_dataset():
    """Create high-quality sample dataset"""
    data = {
        'original_text': [
            "The patient presented with acute myocardial infarction requiring immediate percutaneous coronary intervention with stent placement.",
            "Chronic obstructive pulmonary disease exacerbation with severe dyspnea, hypoxemia, and respiratory acidosis.",
            "Patient exhibits sinus tachycardia and orthopnea with elevated serum troponin levels indicative of cardiac injury.",
            "Cerebrovascular accident involving left middle cerebral artery resulting in right-sided hemiplegia and aphasia.",
            "Hypertensive crisis with posterior reversible encephalopathy syndrome requiring immediate antihypertensive therapy.",
            "Diabetic ketoacidosis with severe metabolic derangement including hyperglycemia, ketonemia, and profound acidosis.",
            "Acute kidney injury secondary to septic shock with oliguria and elevated creatinine requiring dialysis.",
            "Bilateral pneumonia with extensive consolidation requiring broad-spectrum antibiotics and supplemental oxygen.",
            "Gastroesophageal reflux disease causing erosive esophagitis and progressive dysphagia.",
            "Acute appendicitis with perforation and peritonitis requiring emergent laparoscopic appendectomy.",
            "Atrial fibrillation with rapid ventricular response requiring rate control and anticoagulation.",
            "Anaphylactic reaction to penicillin with bronchospasm, angioedema, and hypotension requiring epinephrine.",
            "Deep vein thrombosis with pulmonary embolism requiring systemic anticoagulation therapy.",
            "Severe migraine with aura refractory to analgesics requiring intravenous dihydroergotamine.",
            "Acute decompensated heart failure with pulmonary edema requiring diuretic therapy.",
            "Community-acquired pneumonia complicated by respiratory distress requiring mechanical ventilation.",
            "Type 2 diabetes mellitus inadequately controlled requiring insulin therapy.",
            "Osteoarthritis with severe joint space narrowing causing functional impairment.",
            "Peptic ulcer disease with active hemorrhage requiring endoscopic hemostasis.",
            "Major depressive disorder with suicidal ideation requiring psychiatric hospitalization.",
            "Acute ischemic stroke treated with mechanical thrombectomy.",
            "Hepatic encephalopathy with asterixis requiring lactulose and rifaximin therapy.",
            "Nephrotic syndrome with massive proteinuria and peripheral edema.",
            "Acute pancreatitis with necrosis requiring nutritional support.",
            "Hyperthyroidism causing thyrotoxicosis requiring antithyroid medication.",
            "Severe asthma exacerbation requiring high-dose systemic corticosteroids.",
            "Bacterial meningitis requiring empiric antimicrobial therapy with ceftriaxone.",
            "Acute cholecystitis requiring emergent cholecystectomy.",
            "Pulmonary fibrosis causing severe dyspnea requiring supplemental oxygen.",
            "Subarachnoid hemorrhage requiring neurosurgical intervention.",
        ],
        'simplified_text': [
            "The patient had a severe heart attack needing emergency treatment to open blocked arteries with a tube.",
            "Long-term lung disease got worse with very bad breathing problems and low oxygen levels.",
            "Patient has fast heartbeat and trouble breathing when lying down. Blood tests show heart damage.",
            "Stroke in left brain caused right side paralysis and speech problems.",
            "Dangerously high blood pressure caused brain swelling needing immediate treatment.",
            "Diabetes complication with very high blood sugar and dangerous acid levels.",
            "Kidneys failed due to severe infection with low urine output needing dialysis machine.",
            "Infection in both lungs needing strong antibiotics and extra oxygen.",
            "Stomach acid damaged throat causing swallowing problems.",
            "Burst appendix with belly infection needing emergency surgery.",
            "Irregular heartbeat with fast pulse needing medicine to slow heart and prevent clots.",
            "Severe allergic reaction to penicillin causing breathing problems and low blood pressure.",
            "Blood clot in leg moved to lung needing blood thinners.",
            "Severe headache not helped by pain medicine needing IV treatment.",
            "Heart failure with fluid in lungs needing water pills.",
            "Lung infection causing breathing failure needing breathing machine.",
            "Type 2 diabetes not controlled with pills needing insulin shots.",
            "Severe arthritis with worn cartilage causing disability.",
            "Stomach ulcer bleeding needing procedure to stop bleeding.",
            "Severe depression with thoughts of suicide needing hospital care.",
            "Stroke from blocked artery treated with procedure to remove clot.",
            "Brain confusion from liver failure needing medicines to reduce toxins.",
            "Kidney disease with protein in urine and body swelling.",
            "Severe pancreas inflammation needing feeding support.",
            "Overactive thyroid causing severe symptoms needing medicine.",
            "Severe asthma attack needing high-dose steroid medicine.",
            "Brain infection needing immediate antibiotic treatment.",
            "Infected gallbladder needing emergency surgery.",
            "Lung scarring causing breathing problems needing oxygen.",
            "Brain bleeding needing surgery.",
        ]
    }
    
    df = pd.DataFrame(data)
    return df

@st.cache_data
def prepare_dataset(df):
    """Calculate readability scores and create target classes"""
    df['original_fre'] = df['original_text'].apply(textstat.flesch_reading_ease)
    df['simplified_fre'] = df['simplified_text'].apply(textstat.flesch_reading_ease)
    df['original_grade'] = df['original_text'].apply(textstat.flesch_kincaid_grade)
    df['simplified_grade'] = df['simplified_text'].apply(textstat.flesch_kincaid_grade)
    
    
    df['readability_class'] = df['original_fre'].apply(
        lambda x: 'Easy' if x > 60 else ('Medium' if x > 30 else 'Difficult')
    )
    
    return df


medical_dict = {
    # Cardiovascular
    "myocardial infarction": "heart attack",
    "percutaneous coronary intervention": "procedure to open blocked arteries",
    "stent placement": "tube insertion",
    "hypertension": "high blood pressure",
    "tachycardia": "fast heartbeat",
    "bradycardia": "slow heartbeat",
    "atrial fibrillation": "irregular heartbeat",
    "cardiac arrest": "heart stopped",
    
    # Respiratory
    "chronic obstructive pulmonary disease": "long-term lung disease",
    "dyspnea": "shortness of breath",
    "hypoxemia": "low oxygen levels",
    "respiratory acidosis": "dangerous acid from breathing problems",
    "pneumonia": "lung infection",
    "bronchospasm": "airway tightening",
    "mechanical ventilation": "breathing machine",
    
    # Neurological
    "cerebrovascular accident": "stroke",
    "hemiplegia": "paralysis on one side",
    "aphasia": "speech problems",
    "encephalopathy": "brain swelling",
    "cephalgia": "headache",
    "migraine": "severe headache",
    
    # Metabolic
    "diabetic ketoacidosis": "diabetes complication",
    "hyperglycemia": "high blood sugar",
    "hypoglycemia": "low blood sugar",
    "ketonemia": "toxic acids",
    
    # Renal
    "acute kidney injury": "sudden kidney failure",
    "oliguria": "low urine output",
    "creatinine": "kidney function marker",
    "dialysis": "kidney machine",
    
    # GI
    "gastroesophageal reflux disease": "acid reflux",
    "esophagitis": "throat inflammation",
    "dysphagia": "swallowing difficulty",
    "appendicitis": "inflamed appendix",
    "peritonitis": "belly infection",
    
    # General
    "requiring": "needing",
    "manifested by": "shown by",
    "indicative of": "showing",
    "secondary to": "caused by",
    "refractory to": "not helped by",
    "empiric": "immediate",
    "emergent": "emergency",
    "laparoscopic": "minimally invasive",
}

def extract_features(texts, vectorizer=None):
    """Extract TF-IDF and linguistic features """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            min_df=1,
            stop_words='english',
            sublinear_tf=True
        )
        tfidf_features = vectorizer.fit_transform(texts).toarray()
    else:
        tfidf_features = vectorizer.transform(texts).toarray()

    # Linguistic features 
    linguistic_features = []
    for text in texts:
        word_count = len(text.split())
        sentence_count = textstat.sentence_count(text)
        syllable_count = textstat.syllable_count(text)

        features = [
            word_count,
            len(text),
            sentence_count,
            syllable_count,
            textstat.difficult_words(text),
            # textstat.flesch_reading_ease(text),
            # textstat.flesch_kincaid_grade(text),
            textstat.smog_index(text),
            
            word_count / max(sentence_count, 1), # avg words per sentence
            syllable_count / max(word_count, 1), # avg syllables per word
            len(re.findall(r'\b\w{12,}\b', text)), # count of long words
            text.count(','), # count of commas
        ]
        linguistic_features.append(features)
    
    linguistic_features = np.array(linguistic_features)
    X = np.hstack([tfidf_features, linguistic_features])
    return X, vectorizer


# -------------------------------
# 4. Simplification Functions
# -------------------------------
def rule_based_simplify(text):
    """Enhanced rule-based simplification"""
    simplified = text.lower()
    
    # Replace medical terms
    for term, simple in sorted(medical_dict.items(), key=lambda x: len(x[0]), reverse=True):
        simplified = re.sub(rf"\b{term}\b", simple, simplified, flags=re.IGNORECASE)
    
    # Remove parenthetical content
    simplified = re.sub(r'\s*\([^)]*\)', '', simplified)
    
    # Clean spaces
    simplified = re.sub(r'\s+', ' ', simplified).strip()
    
    # Capitalize
    return simplified.capitalize() if simplified else simplified

@st.cache_resource
def load_t5_model():
    """Load T5 model for text simplification, ensuring English output."""
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        
        model_name = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        
        
        pipe = pipeline(
            "text2text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            device=-1, # -1 for CPU, 0 for first GPU
            num_beams=4, 
            early_stopping=True,
        )
        
        return pipe
    except Exception as e:
        st.warning(f"⚠️ T5 model loading failed: {e}. T5 simplification will be unavailable.")
        return None

def model_simplify(text, pipe):
    """Model-based simplification"""
    if pipe is None:
        return text
    
    try:
        # The prompt "simplify: " is crucial for T5.
        # Ensure do_sample is False for deterministic output.
        result = pipe(
            f"simplify: {text}", 
            max_length=150, 
            min_length=20, 
            do_sample=False, # Ensure deterministic output
            num_beams=4, 
            early_stopping=True,
        )
        
        generated_text = result[0]["generated_text"].strip()
        
        # Regex to remove common non-English prefixes for "simplified" or similar
        generated_text = re.sub(r"^(Einfacher: |Simple: |Simplifier: |Vereinfachen: |Simplified: )", "", generated_text, flags=re.IGNORECASE).strip()

        return generated_text
    except Exception as e:
        st.error(f"Error during T5 simplification: {e}")
        return text # Return original text on error

def translate_to_english(text):
    """Translate text to English using googletrans."""
    if not st.session_state.googletrans_available:
        return text
    try:
        # Detect language first to avoid unnecessary translation if already English
        detected = translator.detect(text)
        if detected.lang != 'en':
            translation = translator.translate(text, dest='en')
            return translation.text
        return text
    except Exception as e:
        st.warning(f"❌ Error during translation to English: {e}. Returning original text.")
        return text


st.title(" Medical Text Simplifier")
st.markdown("**Professional ML-based medical text simplification system**")


st.sidebar.title("📂 Dataset Setup")

dataset_choice = st.sidebar.radio(
    "Select Dataset Source:",
    [" HuggingFace (Med-EASi)", " Sample Dataset (30 examples)"]
)

if dataset_choice == " HuggingFace (Med-EASi)":
    if st.sidebar.button("📥 Load HuggingFace Dataset", type="primary"):
        df = load_local_dataset()
        if df is not None:
            st.session_state.df = prepare_dataset(df)
            st.session_state.dataset_loaded = True
            st.sidebar.success(f"✅ Loaded {len(st.session_state.df)} samples!")
        else:
            st.sidebar.warning("Using sample dataset instead")
            st.session_state.df = prepare_dataset(create_sample_dataset())
            st.session_state.dataset_loaded = True
else: # " Sample Dataset"
    if not st.session_state.dataset_loaded or st.session_state.df is None or len(st.session_state.df) == 0:
        st.session_state.df = prepare_dataset(create_sample_dataset())
        st.session_state.dataset_loaded = True
    st.sidebar.success("✅ Using sample dataset")

if st.session_state.dataset_loaded and st.session_state.df is not None:
    st.sidebar.metric("Dataset Size", len(st.session_state.df))
    st.sidebar.metric("Avg Original FRE", f"{st.session_state.df['original_fre'].mean():.1f}")
    st.sidebar.metric("Avg Simplified FRE", f"{st.session_state.df['simplified_fre'].mean():.1f}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🤖 Train Model", "✨ Simplify Text", "📈 Evaluation"])


with tab1:
    if st.session_state.dataset_loaded and st.session_state.df is not None:
        st.header("📊 Dataset Overview")
        
        df = st.session_state.df
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Samples", len(df))
        with col2:
            improvement = (df['simplified_fre'] - df['original_fre']).mean()
            st.metric("Avg FRE Gain", f"+{improvement:.1f}")
        with col3:
            grade_drop = (df['original_grade'] - df['simplified_grade']).mean()
            st.metric("Avg Grade Drop", f"-{grade_drop:.1f}")
        with col4:
            success = (df['simplified_fre'] > df['original_fre']).sum() / len(df) * 100
            st.metric("Success Rate", f"{success:.0f}%")
        
        st.subheader("Sample Data")
        st.dataframe(df[['original_text', 'simplified_text', 'readability_class']].head(10))
        
        st.subheader("Readability Distribution")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Class distribution
        df['readability_class'].value_counts().plot(kind='bar', ax=ax1, 
                                                     color=['#ff6b6b', '#ffd93d', '#6bcf7f'])
        ax1.set_title("Readability Classes", fontweight='bold')
        ax1.set_xlabel("Class")
        ax1.set_ylabel("Count")
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
        
        # FRE comparison
        ax2.scatter(df['original_fre'], df['simplified_fre'], alpha=0.6, s=80)
        ax2.plot([0, 100], [0, 100], 'r--', label='No Change')
        ax2.set_xlabel("Original FRE")
        ax2.set_ylabel("Simplified FRE")
        ax2.set_title("Simplification Effect", fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        st.pyplot(fig)
    else:
        st.info("👈 Please load a dataset from the sidebar")


with tab2:
    st.header("🤖 Model Training")
    
    if st.session_state.dataset_loaded and st.session_state.df is not None:
        st.markdown("""
        **ML Pipeline:**
        - **Features**: TF-IDF (100) + Bigrams + Linguistic (12) = ~112 features
        - **Model**: Gradient Boosting Classifier
        - **Task**: Classify readability (Easy/Medium/Difficult)
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Size", 0.15, 0.35, 0.2, 0.05)
        with col2:
            n_estimators = st.slider("Trees", 50, 200, 100, 10)
        with col3:
            max_depth = st.slider("Max Depth", 5, 15, 8)
        
        if st.button("🚀 Train Model", type="primary"):
            df = st.session_state.df
            
            with st.spinner("Extracting features..."):
                # Step 1: TF-IDF + linguistic features
                X, vectorizer = extract_features(df['original_text']) # This now extracts 12 linguistic features
                y = df['readability_class'].values
            
            with st.spinner("Splitting dataset..."):
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            
            with st.spinner("Scaling features..."):
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            
            with st.spinner("Training Gradient Boosting..."):
                model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=0.1,
                    max_depth=max_depth,
                    random_state=42
                )
                model.fit(X_train_scaled, y_train)
            
            
            joblib.dump((model, vectorizer, scaler), 'model.joblib')
            
            st.session_state.model_trained = True
            
            # Evaluate model
            train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
            test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
            y_pred = model.predict(X_test_scaled)
            
            st.success("✅ Model trained and saved successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Train Accuracy", f"{train_acc:.1%}")
            with col2:
                st.metric("Test Accuracy", f"{test_acc:.1%}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=np.unique(y), yticklabels=np.unique(y), ax=ax)
            ax.set_ylabel('True Label', fontweight='bold')
            ax.set_xlabel('Predicted Label', fontweight='bold')
            st.pyplot(fig)
    else:
        st.info("👈 Please load a dataset first")

# -------------------------------
# TAB 3: SIMPLIFICATION
# -------------------------------
with tab3:
    st.header("✨ Simplify Medical Text")
    
    # Load T5 model
    t5_pipe = load_t5_model()
    
    # Load ML model if exists
    ml_model_loaded = os.path.exists('model.joblib') # Changed to .joblib
    ml_model, ml_vectorizer, ml_scaler = None, None, None 
    if ml_model_loaded:
        try:
            ml_model, ml_vectorizer, ml_scaler = joblib.load('model.joblib') 
            st.success("✅ ML model loaded")
        except Exception as e:
            st.error(f"❌ Error loading ML model: {e}")
            ml_model_loaded = False
    else:
        st.info("ℹ️ Train ML model for readability classification (optional)")
    
    user_input = st.text_area(
        "Enter medical text:",
        height=120,
        value="The patient presented with acute myocardial infarction requiring immediate percutaneous coronary intervention."
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        use_t5 = st.checkbox("Use T5 Model", value=(t5_pipe is not None))
        if st.session_state.googletrans_available:
            translate_t5_output = st.checkbox("Translate T5 Output to English (Google Translate)", value=False)
        else:
            translate_t5_output = False
            st.markdown("<small style='color: gray;'>Google Translate not available.</small>", unsafe_allow_html=True)
            
    
    if st.button("✨ Simplify Text", type="primary"):
        if user_input.strip():
            with st.spinner("Processing..."):
                # Step 1: Rule-based simplification
                rule_simplified = rule_based_simplify(user_input)
                
                # Step 2: Model-based simplification
                if use_t5 and t5_pipe:
                    t5_output = model_simplify(rule_simplified, t5_pipe)
                    if translate_t5_output and st.session_state.googletrans_available:
                        final_simplified = translate_to_english(t5_output)
                        if t5_output != final_simplified:
                            st.info("ℹ️ T5 output was translated to English using Google Translate.")
                    else:
                        final_simplified = t5_output
                else:
                    final_simplified = rule_simplified
                
                # Step 3: ML Prediction
                if ml_model_loaded and ml_model is not None: # Check if model is actually loaded
                    # Use saved vectorizer
                    X_tfidf = ml_vectorizer.transform([user_input]).toarray()
                    
                    word_count = len(user_input.split())
                    sentence_count = textstat.sentence_count(user_input)
                    syllable_count = textstat.syllable_count(user_input)

                    linguistic_features_single = [
                        word_count,
                        len(user_input),
                        sentence_count,
                        syllable_count,
                        textstat.difficult_words(user_input),
                        # textstat.flesch_reading_ease(user_input),
                        # textstat.flesch_kincaid_grade(user_input),
                        textstat.smog_index(user_input),
                        word_count / max(sentence_count, 1), # avg words per sentence
                        syllable_count / max(word_count, 1), # avg syllables per word
                        len(re.findall(r'\b\w{12,}\b', user_input)), # count of long words
                        user_input.count(','), # count of commas
                    ]
                    
                    X_ling = np.array([linguistic_features_single]) 
                    
                    X_input = np.hstack([X_tfidf, X_ling])
                    
                    X_scaled = ml_scaler.transform(X_input)
                    
                    prediction = ml_model.predict(X_scaled)[0]
                    proba = ml_model.predict_proba(X_scaled)[0] 
                else:
                    prediction = "N/A"
                    proba = None
                
                # Step 4: Metrics
                orig_fre = textstat.flesch_reading_ease(user_input)
                rule_fre = textstat.flesch_reading_ease(rule_simplified)
                final_fre = textstat.flesch_reading_ease(final_simplified)
                
                orig_grade = textstat.flesch_kincaid_grade(user_input)
                final_grade = textstat.flesch_kincaid_grade(final_simplified)
            
            st.markdown("---")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if ml_model_loaded and ml_model is not None:
                    st.metric("ML Prediction", prediction)
            with col2:
                st.metric("Original FRE", f"{orig_fre:.1f}")
            with col3:
                st.metric("Simplified FRE", f"{final_fre:.1f}")
            with col4:
                improvement = final_fre - orig_fre
                st.metric("Improvement", f"+{improvement:.1f}", delta=f"+{improvement:.1f}")
            
            # Display text comparison
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📄 Original")
                st.info(user_input)
                st.caption(f"Grade Level: {orig_grade:.1f}")
            
            with col2:
                st.markdown("### ✅ Simplified")
                st.success(final_simplified)
                st.caption(f"Grade Level: {final_grade:.1f}")
            
            # Display improvement message
            if improvement > 10:
                st.success(f"🎉 Excellent! Readability improved by {improvement:.1f} points!")
            elif improvement > 5:
                st.info(f"✅ Good! Readability improved by {improvement:.1f} points")
            
            # Show simplification steps
            with st.expander("🔍 View Simplification Steps"):
                st.markdown("**Step 1: Rule-Based Simplification**")
                st.write(rule_simplified)
                st.caption(f"FRE: {rule_fre:.1f}")
                
                if use_t5 and t5_pipe:
                    st.markdown("**Step 2: T5 Model Refinement**")
                    st.write(t5_output) # Show raw T5 output before external translation
                    st.caption(f"FRE: {textstat.flesch_reading_ease(t5_output):.1f}")
                    if translate_t5_output and t5_output != final_simplified:
                        st.markdown("**Step 3: Google Translate (Ensuring English Output)**")
                        st.write(final_simplified)
                        st.caption(f"FRE: {final_fre:.1f}")
        else:
            st.warning("Please enter some text")

# -------------------------------
# TAB 4: EVALUATION
# -------------------------------
with tab4:
    if st.session_state.dataset_loaded and st.session_state.df is not None:
        st.header("📈 System Evaluation")
        
        df = st.session_state.df
        df['improvement'] = df['simplified_fre'] - df['original_fre']
        df['grade_reduction'] = df['original_grade'] - df['simplified_grade']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg FRE Gain", f"+{df['improvement'].mean():.2f}")
        with col2:
            st.metric("Avg Grade Drop", f"-{df['grade_reduction'].mean():.2f}")
        with col3:
            success = (df['improvement'] > 5).sum() / len(df) * 100
            st.metric("Success Rate", f"{success:.0f}%")
        with col4:
            target = (df['simplified_grade'] <= 8).sum() / len(df) * 100
            st.metric("Target Grade ≤8", f"{target:.0f}%")
        
        st.subheader("Improvement Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df['improvement'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(df['improvement'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {df["improvement"].mean():.1}')
        
        ax.set_xlabel('FRE Improvement')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Readability Improvements', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("👈 Please load a dataset first")

# Footer
st.markdown("---")
st.caption("""
**Project:** Medical Text Readability Simplifier  
**Author:** Gayathiri SB  
**Tech Stack:** Python, scikit-learn, TF-IDF, Gradient Boosting, T5, Streamlit  
**Dataset:** HuggingFace Med-EASi (4000+ samples) or Sample Dataset (30 samples)
""")