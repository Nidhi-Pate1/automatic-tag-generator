import streamlit as st
import spacy
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AutoTag Studio", page_icon="💡", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #f1f3f6; color: #1e1b4b; }
    
    .app-title {
        font-size: 38px; font-weight: 700;
        background: linear-gradient(135deg, #06b6d4 0%, #4f46e5 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
    }
    
    .input-header { color: #64748b; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 10px; }

    .stTextArea textarea { 
        border-radius: 20px; border: none;
        background-color: #f1f3f6; padding: 25px;
        box-shadow: inset 8px 8px 16px #e2e4e8, inset -8px -8px 16px #ffffff;
    }

    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.8em; font-weight: 700;
        background: linear-gradient(135deg, #06b6d4 0%, #4f46e5 100%);
        color: white; border: none; transition: 0.3s;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3);
    }

    /* THE FIX: Unified Result Suite */
    .result-suite {
        padding: 30px; border-radius: 25px;
        background: #ffffff;
        box-shadow: 20px 20px 60px #e2e4e8, -20px -20px 60px #ffffff;
        border: 1px solid #f8fafc;
        margin-top: 0px !important;
    }
    
    .tag-chip {
        display: inline-block; padding: 7px 16px; border-radius: 8px;
        background: #f8fafc; color: #4f46e5; border: 1px solid #e5e7eb;
        margin: 4px; font-weight: 600; font-size: 13px;
    }
    
    .google-card {
        padding: 15px; border-radius: 12px; background: #fafafa;
        border: 1px dashed #cbd5e1; margin-top: 20px;
    }

    /* FORCING EMPTY BLOCKS TO DISAPPEAR */
    div[data-testid="stVerticalBlock"] > div:empty { display: none !important; }
    .element-container { margin-bottom: 0px !important; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    nltk.download('punkt_tab', quiet=True)
    return nlp, model

NLP, BRAIN = load_all_models()

def mmr_logic(doc_emb, cand_embs, words, top_n=5, diversity=0.5):
    word_doc_sim = cosine_similarity(cand_embs, doc_emb)
    word_word_sim = cosine_similarity(cand_embs)
    selected = [np.argmax(word_doc_sim)]
    remaining = [i for i in range(len(words)) if i not in selected]
    for _ in range(top_n - 1):
        if not remaining: break
        rel = word_doc_sim[remaining]
        div_scores = np.max(word_word_sim[remaining][:, selected], axis=1)
        mmr = (1 - diversity) * rel - diversity * div_scores.reshape(-1, 1)
        best = remaining[np.argmax(mmr)]
        selected.append(best)
        remaining.remove(best)
    return [words[i] for i in selected]

def generate_tags_full_logic(text):
    doc = NLP(text)
    candidates = list(set([
        chunk.text.title() for chunk in doc.noun_chunks 
        if not chunk.root.is_stop and len(chunk.text) > 3 
        and not any(x in chunk.text.lower() for x in ['nisha', 'also', 'boast', 'apart'])
    ]))
    if len(candidates) < 2: return candidates
    d_emb = BRAIN.encode([text])
    c_emb = BRAIN.encode(candidates)
    return mmr_logic(d_emb, c_emb, candidates)

st.markdown('<h1 class="app-title">Automatic Tag Generator</h1>', unsafe_allow_html=True)

col_input, col_output = st.columns([1.2, 1], gap="large")

with col_input:
    st.markdown('<p class="input-header">Your Input</p>', unsafe_allow_html=True)
    user_article = st.text_area("", height=420, placeholder="Paste your article content here...", label_visibility="collapsed")
    st.write("")
    btn = st.button("Generate Tags")

with col_output:
    st.markdown('<p class="input-header">Results</p>', unsafe_allow_html=True)
    
    if btn and user_article.strip():
        with st.spinner("Calculating semantics..."):
            tags = generate_tags_full_logic(user_article)
            first_sent = nltk.sent_tokenize(user_article)[0]
            
            badge_html = "".join([f'<span class="tag-chip">{t}</span>' for t in tags])
            
            st.markdown(f"""
                <div class="result-suite">
                    <h4 style="margin-bottom:15px; color:#1e1b4b; font-size:18px;">🏷️ Semantic Keywords</h4>
                    {badge_html}
                    <div class="google-card">
                        <h4 style="margin-bottom:10px; color:#1e1b4b; font-size:16px;">🔍 Search Preview</h4>
                        <p style="color:#1a0dab; font-size:18px; font-weight:600; margin-bottom:2px;">{tags[0]} | Insights</p>
                        <p style="color:#4b5563; font-size:13px; line-height:1.5;">{first_sent[:150]}...</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Input content on the left to see your optimization suite.")

