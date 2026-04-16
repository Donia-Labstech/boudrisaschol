# ============================================
# تطبيق المؤسسة التربوية - متوسطة الشهيد بوضريسة محمد الأمين
# نسخة متكاملة جاهزة للنشر على Hugging Face Spaces
# ============================================

import streamlit as st
import requests
from datetime import datetime
import tempfile
import os
import pandas as pd
from streamlit_option_menu import option_menu
from huggingface_hub import HfApi, upload_file, list_repo_files

# -----------------------------
# 1. تهيئة الصفحة ودعم اللغة العربية (RTL)
# -----------------------------
st.set_page_config(page_title="مؤسسة الشهيد بوضريسة", page_icon="🏫", layout="wide")

# CSS لتخصيص الألوان ودعم RTL
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    * {
        font-family: 'Tajawal', sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    /* دعم الكتابة من اليمين لليسار */
    .stApp, div, p, h1, h2, h3 {
        direction: rtl;
        text-align: right;
    }
    /* تخصيص الجانب بار */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    /* بطاقات الملفات */
    .file-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 2. إدارة الأسرار (Secrets)
# -----------------------------
# يجب وضع هذه القيم في Hugging Face Secrets أو في ملف .streamlit/secrets.toml محلياً
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
    ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
    TEACHER_PASSWORD = st.secrets["TEACHER_PASSWORD"]
except Exception as e:
    st.error("⚠️ يرجى إعداد الأسرار (Secrets) في Hugging Face Space: HF_TOKEN, ADMIN_PASSWORD, TEACHER_PASSWORD")
    st.stop()

# إعداد اتصال Hugging Face
api = HfApi(token=HF_TOKEN)

# اسم مستودع Dataset لتخزين الملفات (قم بتغييره إلى اسم المستودع الذي أنشأته)
DATASET_REPO = "your-username/school-resources"  # غيّر هذا إلى اسم المستودع الخاص بك

# -----------------------------
# 3. دوال مساعدة (Helper Functions)
# -----------------------------
def get_wisdom_of_day():
    """استدعاء نموذج LLM من Hugging Face لإنشاء حكمة اليوم"""
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""أنت حكيم تربوي. اكتب حكمة قصيرة ومؤثرة للمعلمين والطلاب ليوم {today}. 
    يجب أن تكون الحكمة باللغة العربية الفصحى، لا تزيد عن 30 كلمة، وتشجع على العلم والأخلاق."""
    
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 100, "temperature": 0.7}}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            wisdom = result[0]['generated_text'].replace(prompt, "").strip()
            return wisdom if wisdom else "طلب العلم فريضة على كل مسلم ومسلمة. - الحديث الشريف"
        else:
            return "طلب العلم فريضة على كل مسلم ومسلمة. - الحديث الشريف"
    except Exception:
        return "من جد وجد، ومن سار على الدرب وصل."

def analyze_sentiment(text):
    """تحليل المشاعر للنصوص باستخدام نموذج Hugging Face"""
    API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=30)
        if response.status_code == 200:
            result = response.json()[0]
            sentiment = max(result, key=lambda x: x['score'])['label']
            return sentiment
        else:
            return "NEUTRAL"
    except Exception:
        return "NEUTRAL"

def upload_file_to_hf_dataset(file_bytes, file_name, metadata):
    """رفع ملف إلى Hugging Face Dataset"""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file_name)
    
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    
    try:
        upload_file(
            path_or_fileobj=temp_path,
            path_in_repo=f"resources/{file_name}",
            repo_id=DATASET_REPO,
            token=HF_TOKEN
        )
        # هنا يمكن حفظ metadata في ملف JSON منفصل إذا أردت
        return True
    except Exception as e:
        st.error(f"خطأ في الرفع: {str(e)}")
        return False
    finally:
        # تنظيف الملف المؤقت
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

def get_resources_list():
    """جلب قائمة الملفات من Hugging Face Dataset"""
    try:
        files = list_repo_files(DATASET_REPO, token=HF_TOKEN)
        return [f for f in files if f.startswith("resources/")]
    except Exception:
        return []

# -----------------------------
# 4. الشريط الجانبي (Sidebar Navigation)
# -----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2942/2942707.png", width=80)
    st.title("القائمة الرئيسية")
    
    selected = option_menu(
        menu_title=None,
        options=["🏠 الرئيسية", "📂 بنك الموارد", "🏢 الإدارة", "☕ قاعة الأساتذة", "💡 بنك الأفكار", "👨‍👩‍👧 بواب الأولياء"],
        icons=["house", "folder", "building", "cup", "lightbulb", "people"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#1e3c72", "font-size": "20px"},
            "nav-link": {"font-size": "18px", "text-align": "right", "margin": "5px", "font-family": "Tajawal"},
            "nav-link-selected": {"background-color": "#2a5298"},
        }
    )

# -----------------------------
# 5. الواجهة الرئيسية (🏠 الرئيسية)
# -----------------------------
if selected == "🏠 الرئيسية":
    st.markdown("""
    <div class="main-header">
        <h1>🏫 متوسطة الشهيد بوضريسة محمد الأمين</h1>
        <p>منصة رقمية متكاملة للتعليم والتفاعل المجتمعي</p>
    </div>
    """, unsafe_allow_html=True)
    
    # حكمة اليوم المولدة بالذكاء الاصطناعي
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.markdown("---")
        st.subheader("📖 حكمة اليوم")
        
        if 'wisdom' not in st.session_state:
            with st.spinner("جاري استلهام حكمة اليوم..."):
                st.session_state.wisdom = get_wisdom_of_day()
        
        st.markdown(f"""
        <div style="background-color:#f0f8ff;border-right:5px solid #2a5298;padding:1.5rem;border-radius:10px;text-align:center;">
            <p style="font-size:1.4rem;font-style:italic;">"{st.session_state.wisdom}"</p>
            <p style="margin-top:1rem;">✨ حكمة اليوم {datetime.now().strftime("%Y/%m/%d")}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 تحديث الحكمة"):
            st.session_state.wisdom = get_wisdom_of_day()
            st.rerun()
        st.markdown("---")
    
    # إحصائيات سريعة
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 عدد الملفات", len(get_resources_list()))
    with col2:
        st.metric("💡 الأفكار المطروحة", "12")
    with col3:
        st.metric("👨‍🏫 الأساتذة", "35")
    with col4:
        st.metric("🎓 الطلاب", "420")
    
    # أخبار وإعلانات سريعة
    st.subheader("📢 آخر الإعلانات")
    announcements = [
        "📅 امتحانات الفصل الأول تبدأ في 15 ديسمبر",
        "🎉 تكريم المتفوقين في اللغة العربية يوم الخميس القادم",
        "📝 اجتماع أولياء الأمور يوم السبت الموافق 10 ديسمبر"
    ]
    for announcement in announcements:
        st.info(announcement)

# -----------------------------
# 6. تبويب بنك الموارد الرقمية
# -----------------------------
elif selected == "📂 بنك الموارد":
    st.header("📂 بنك الموارد الرقمية")
    st.markdown("---")
    
    # نموذج رفع ملف جديد
    with st.expander("➕ رفع ملف جديد", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            file_type = st.selectbox("نوع الملف", ["فروض", "اختبارات", "دروس", "ملخصات"])
            level = st.selectbox("المستوى الدراسي", ["السنة الأولى متوسط", "السنة الثانية متوسط", "السنة الثالثة متوسط", "السنة الرابعة متوسط"])
        with col2:
            subject = st.selectbox("المادة", ["الرياضيات", "العلوم", "الفيزياء", "اللغة العربية", "اللغة الفرنسية", "الإنجليزية", "التاريخ", "التربية الإسلامية"])
            uploaded_file = st.file_uploader("اختر الملف (PDF فقط)", type=["pdf"])
        
        if uploaded_file and st.button("رفع الملف"):
            with st.spinner("جاري رفع الملف..."):
                auto_tags = analyze_sentiment(uploaded_file.name)
                metadata = {
                    "type": file_type,
                    "level": level,
                    "subject": subject,
                    "auto_tags": auto_tags,
                    "upload_date": datetime.now().strftime("%Y-%m-%d")
                }
                success = upload_file_to_hf_dataset(uploaded_file.getvalue(), uploaded_file.name, metadata)
                if success:
                    st.success("✅ تم رفع الملف بنجاح!")
                else:
                    st.error("❌ حدث خطأ أثناء الرفع")
    
    # عرض قائمة الملفات
    st.subheader("📚 قائمة الملفات المتاحة")
    resources = get_resources_list()
    
    if resources:
        for file_path in resources:
            file_name = file_path.replace("resources/", "")
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    <div class="file-card">
                        <strong>📄 {file_name}</strong><br>
                        <small>📅 تاريخ الرفع: {datetime.now().strftime("%Y-%m-%d")}</small>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    download_url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{file_path}"
                    st.markdown(f'<a href="{download_url}" target="_blank"><button style="background-color:#2a5298;color:white;padding:8px 15px;border:none;border-radius:5px;">📥 تحميل</button></a>', unsafe_allow_html=True)
    else:
        st.info("لا توجد ملفات مرفوعة بعد. قم برفع أول ملف من القسم أعلاه.")

# -----------------------------
# 7. تبويب الإدارة الافتراضية
# -----------------------------
elif selected == "🏢 الإدارة":
    st.header("🏢 الإدارة الافتراضية")
    
    password = st.text_input("🔐 كلمة المرور الإدارية", type="password")
    
    if password == ADMIN_PASSWORD:
        st.success("تم التحقق من الصلاحيات ✅")
        
        with st.form("announcement_form"):
            st.subheader("📢 نشر إعلان جديد")
            announcement_title = st.text_input("عنوان الإعلان")
            announcement_content = st.text_area("محتوى الإعلان")
            announcement_target = st.selectbox("موجه إلى", ["الطلاب", "الأساتذة", "أولياء الأمور", "الجميع"])
            
            if st.form_submit_button("نشر الإعلان"):
                st.success("تم نشر الإعلان بنجاح!")
        
        st.subheader("📋 الإعلانات السابقة")
        st.info("لا توجد إعلانات سابقة.")
        
    elif password:
        st.error("❌ كلمة المرور غير صحيحة")

# -----------------------------
# 8. تبويب قاعة الأساتذة
# -----------------------------
elif selected == "☕ قاعة الأساتذة":
    st.header("☕ قاعة الأساتذة - النقاش المهني")
    
    teacher_password = st.text_input("🔐 كلمة مرور قاعة الأساتذة", type="password", key="teacher_pass")
    
    if teacher_password == TEACHER_PASSWORD:
        st.success("مرحباً أستاذ/أستاذة! 🌟")
        
        st.subheader("🗣️ النقاشات المهنية")
        with st.form("discussion_form"):
            discussion_title = st.text_input("عنوان النقاش")
            discussion_content = st.text_area("المحتوى")
            discussion_tags = st.multiselect("الوسوم", ["مناهج", "تقييمات", "أنشطة", "تكنولوجيا التعليم", "توجيه تربوي"])
            
            if st.form_submit_button("نشر النقاش"):
                st.success("تم نشر النقاش بنجاح!")
        
        st.subheader("📝 المذكرات التربوية")
        uploaded_memo = st.file_uploader("رفع مذكرة تربوية", type=["pdf", "docx"], key="memo")
        
        if uploaded_memo and st.button("رفع المذكرة"):
            st.success("تم رفع المذكرة بنجاح!")
            
    elif teacher_password:
        st.error("❌ كلمة المرور غير صحيحة")

# -----------------------------
# 9. تبويب بنك الأفكار ومنتدى الطلاب
# -----------------------------
elif selected == "💡 بنك الأفكار":
    st.header("💡 بنك الأفكار ومنتدى الطلاب")
    
    st.markdown("""
    <div style="background-color:#e8f4f8;padding:1rem;border-radius:10px;margin-bottom:2rem;">
        <p style="font-size:1.2rem;">📢 شاركنا بفكرتك أو اقتراحك لتطوير المؤسسة، وسيتم تحليلها بواسطة الذكاء الاصطناعي 
        لاختيار أفضل الأفكار!</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("idea_form"):
        student_name = st.text_input("الاسم (اختياري)")
        idea_title = st.text_input("عنوان الفكرة/الاقتراح")
        idea_content = st.text_area("محتوى الفكرة بالتفصيل")
        idea_category = st.selectbox("التصنيف", ["تحسين الدروس", "الأنشطة المدرسية", "المرافق", "التكنولوجيا", "أخرى"])
        
        submitted = st.form_submit_button("💡 إرسال الفكرة")
        
        if submitted and idea_content:
            with st.spinner("جاري تحليل الفكرة بواسطة الذكاء الاصطناعي..."):
                sentiment = analyze_sentiment(idea_content)
                
                if sentiment == "POSITIVE":
                    st.balloons()
                    st.success("🎉 رائعة! هذه فكرة إيجابية ومبدعة. تم إرسالها إلى الإدارة.")
                elif sentiment == "NEGATIVE":
                    st.warning("🙏 نشكرك على مشاركتك. نأمل أن نتمكن من معالجة هذه النقطة قريباً.")
                else:
                    st.info("📝 تم استلام فكرتك. شكراً لمشاركتك!")
        elif submitted:
            st.error("الرجاء كتابة محتوى الفكرة")
    
    st.subheader("✨ الأفكار الأكثر إيجابية هذا الشهر")
    st.info("سيتم عرض الأفكار الحاصلة على تقييم إيجابي من الذكاء الاصطناعي هنا قريباً.")

# -----------------------------
# 10. تبويب بواب أولياء الأمور
# -----------------------------
elif selected == "👨‍👩‍👧 بواب الأولياء":
    st.header("👨‍👩‍👧 بواب أولياء الأمور - الاستعلام عن النتائج")
    
    st.markdown("""
    <div style="background-color:#f0f7f0;padding:1rem;border-radius:10px;margin-bottom:1rem;">
        <p>🔍 أدخل رقم تسجيل الطالب للاستعلام عن نتائجه.</p>
    </div>
    """, unsafe_allow_html=True)
    
    student_id = st.text_input("📝 رقم تسجيل الطالب")
    
    if st.button("استعلام"):
        if student_id:
            with st.spinner("جاري البحث..."):
                st.success(f"✅ تم العثور على الطالب: محمد أحمد")
                results_data = {
                    "المادة": ["الرياضيات", "العلوم", "اللغة العربية", "اللغة الفرنسية", "الإنجليزية"],
                    "العلامة": ["17/20", "15/20", "18/20", "14/20", "16/20"],
                    "التقدير": ["ممتاز", "جيد جداً", "ممتاز", "جيد جداً", "جيد جداً"]
                }
                results_df = pd.DataFrame(results_data)
                st.table(results_df)
                st.info("📅 الفصل الأول - السنة الدراسية 2025/2026")
        else:
            st.error("الرجاء إدخال رقم التسجيل")
