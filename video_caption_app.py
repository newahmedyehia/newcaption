import streamlit as st
import whisper
import tempfile
import os

# تحميل نموذج Whisper
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# دالة لتحويل الفيديو إلى نص
def transcribe_video(video_file, model):
    # حفظ الملف المؤقت
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_file.read())
        temp_file_path = temp_file.name

    # استخراج النص من الفيديو
    result = model.transcribe(temp_file_path)
    
    # حذف الملف المؤقت
    os.unlink(temp_file_path)
    
    return result["text"]

# إعداد الصفحة الرئيسية لتطبيق Streamlit
st.title("تحويل الكلام في الفيديو إلى نص")
st.write("قم بتحميل ملف فيديو وسنقوم بتحويل الكلام فيه إلى نص باستخدام Whisper.")

# تحميل نموذج Whisper
model = load_whisper_model()

# واجهة تحميل الملف
uploaded_file = st.file_uploader("اختر ملف فيديو", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("بدء التحويل إلى نص"):
        with st.spinner("جاري تحويل الفيديو إلى نص..."):
            transcription = transcribe_video(uploaded_file, model)
        
        st.success("تم التحويل بنجاح!")
        st.write(transcription)
        
        # إنشاء زر لتنزيل النص
        st.download_button(
            label="تنزيل النص",
            data=transcription,
            file_name="transcription.txt",
            mime="text/plain"
        )
