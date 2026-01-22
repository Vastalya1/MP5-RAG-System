import google.generativeai as genai
genai.configure(api_key="AIzaSyBLJ_eYaCBQ6TY4RUGf_gelHyU1H4pPw1g")

for m in genai.list_models():
    print(m.name)
