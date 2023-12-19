import streamlit as st
import pandas as pd
import torch
# from transformers import AutoModel, AutoTokenizer
import re
from underthesea import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np

from model_loader import load_phobert_model, load_phobert_tokenizer

phobert = load_phobert_model()
tokenizer = load_phobert_tokenizer()

#-------------------------------------------------------------------------------------------

st.set_page_config(layout = "wide")

st.markdown(
    """
    <style>
        div[data-baseweb="input"] input {
            font-size: 20px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)
#-------------------------------------------------------------------------------------------

def sort_data_by_id():
    global data
    data.sort_values(by=["ID"], inplace=True)
    data.reset_index(drop=True, inplace=True)

# Load data from Excel if available
try:
    data = pd.read_excel("patient_data.xlsx")
except FileNotFoundError:
    # If the file is not found, create an empty DataFrame
    data = pd.DataFrame(columns=["ID", "Name", "Sex", "Height", "Weight", "Date of Birth", "Blood Type", "Allergy", "Symptoms", "Diseases", "Medicine"])

#-------------------------------------------------------------------------------------------
    
# Đọc dữ liệu từ file JSON
with open('data_pre/drug.json', 'r', encoding='utf-8') as file:
    data_drug = json.load(file)

drug_names = [item['name'] for item in data_drug]
drug_ids = [item['id'] for item in data_drug]

# documents_drug = [
#   str(item.get('uses', ''))
#   for item in data_drug
# ]

loaded_embedding_matrix = np.load('illness/embedding_matrix.npy')
df_illness = pd.read_json('data_pre/illness.json')

new_df = pd.read_json('data_pre/illness_2.json')
#-------------------------------------------------------------------------------------------

# def load_stopwords(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         stopwords = [line.strip() for line in file.readlines()]
#     return set(stopwords)

# def preprocess_text(text, stopwords_file_path= 'vietnamese-stopwords.txt'):
#     custom_stopwords = load_stopwords(stopwords_file_path)
#     # Chuyển về từ thường
#     text = str(text).lower()
#     # Xóa dấu câu, ký tự đặc biệt tiếng Việt
#     text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ', text)
#     # Tách từ và Xóa stopword tiếng Việt
#     words = word_tokenize(text)
#     filtered_words = [word for word in words if word not in custom_stopwords]

#     return ' '.join(filtered_words)


# Hàm xử lý đoạn văn bản để truy vấn
def process_text(text):
#   text= preprocess_text(text)
  inputs = tokenizer(text, return_tensors="pt", truncation=True)
  outputs = phobert(**inputs)
  embeddings = outputs.last_hidden_state.mean(dim=1)

  return embeddings

def tokenize_and_embed(text):
    # Split the text into overlapping windows
    window_size = 512  # Adjust the window size as needed
    stride = 256  # Adjust the stride as needed
    windows = [text[i:i+window_size] for i in range(0, len(text), stride)]

    # Tokenize and embed each window
    embeddings = []
    for window in windows:
        tokens = tokenizer(window, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = phobert(**tokens)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).numpy().squeeze())

    # Average the embeddings of all windows
    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        return avg_embedding
    else:
        return np.zeros(768)  # Return a zero vector if there are no windows
    

#-------------------------------------------------------------------------------------------

# # Hàm dự đoán bệnh dựa trên triệu chứng
# def perform_diagnosis(symptoms_1):
#     # Thực hiện dự đoán bệnh ở đây
#     # Đây là nơi để tích hợp logic của bạn
#     # Ví dụ đơn giản: Trả về danh sách bệnh
#     list_Diseases = []
#     A = "Bệnh A cho triệu chứng " + symptoms_1
#     B = "Bệnh B cho triệu chứng " + symptoms_1
#     list_Diseases = [A, B]
#     return list_Diseases


# Load the document_embeddings
document_embeddings = torch.load('drug/document_embeddings.pt')

# # Hàm gợi ý thuốc dựa trên bệnh
# def suggest_medicine(disease):
#     # Thực hiện gợi ý thuốc ở đây
#     # Đây là nơi để tích hợp logic của bạn
#     # Ví dụ đơn giản: Trả về danh sách thuốc cho mỗi bệnh
#     list_Medicine = []
#     A = "Thuốc 1 cho " + disease
#     B = "Thuốc 2 cho " + disease
#     list_Medicine = [A, B]
#     return list_Medicine

# Hàm khuyến nghị thuốc dựa trên đoạn văn bản đầu vào
def suggest_medicine(query_text, top_k=5):
    # Truy vấn cho tất cả các biến
    query_embedding = process_text(query_text)

    # Tạo danh sách khuyến nghị
    recommendations = []
    for i in range(len(drug_ids)):
        similarity_scores = cosine_similarity(query_embedding.detach().numpy(), document_embeddings[i].detach().numpy())[0]
        recommendations.append((drug_ids[i], drug_names[i], data_drug[i]['uses'],data_drug[i]['formulationAndBrand'], data_drug[i]['warning'], data_drug[i]['caution'],similarity_scores[0]))

    # Sắp xếp kết quả theo điểm số giảm dần
    sorted_recommendations = sorted(recommendations, key=lambda x: x[6], reverse=True)

    # Chọn top k recommendations
    top_k_recommendations = sorted_recommendations[:top_k]

    list_Medicine = top_k_recommendations 
    return list_Medicine


def recommend(query_text, top_k=5):
    # Tokenize and embed the user input
    user_input_embedding = tokenize_and_embed(query_text)
    
    # Normalize the user input embedding
    user_input_embedding = user_input_embedding / np.linalg.norm(user_input_embedding)

    if np.isnan(loaded_embedding_matrix).any():
        raise ValueError("Loaded embedding matrix contains NaN values.")

    # Calculate cosine similarity between user input and medicine vectors
    user_similarity = cosine_similarity(user_input_embedding.reshape(1, -1), loaded_embedding_matrix)

    # Get the indices of the top N most similar medicines
    top_indices = user_similarity.argsort()[0][-top_k:][::-1]

    # Return the top N recommended medicines
    top_recommendations = [(new_df.iloc[i]['id'],new_df.iloc[i]['title'], df_illness.iloc[i]['symptoms_free'], user_similarity[0, i]) for i in top_indices]

    return top_recommendations


#-------------------------------------------------------------------------------------------

# Giao diện người dùng Streamlit
def main():
    global data  # Declare data as a global variable
    list_id_benhnhan = list((data["ID"].unique()))
    st.markdown("<h1 style='text-align: center;'>Hệ thống Dự đoán Bệnh và Gợi ý Thuốc</h1>", unsafe_allow_html=True)



    section = st.sidebar.selectbox("Mục lục 1 ", ["Tạo mới thông tin bệnh nhân","Tra cứu thông tin bệnh nhân","Dự đoán bệnh", "Gợi ý thuốc","Dữ liệu bệnh nhân"])
    
    # Hiển thị chọn ID bệnh nhân hoặc tạo mới
    
    
    if section =="Tạo mới thông tin bệnh nhân":
        # Create new patient information
        st.write("## Tạo mới thông tin bệnh nhân.")
        
        # Example: Add code for entering new patient information
        name = st.text_input("Họ và tên:")
        col_sex, col_height, col_weight = st.columns(3)
        col_date_of_birth, col_blood_type, col_ = st.columns(3)
        with col_sex:
            sex = st.selectbox("Giới tính:", ["Nam", "Nữ"])
        with col_height:
            height = st.number_input("Chiều cao (cm):")
        with col_weight:
            weight = st.number_input("Cân nặng (kg):")
        with col_date_of_birth:
            date_of_birth = st.date_input("Ngày sinh:")
        with col_blood_type:
            blood_type = st.text_input("Nhóm máu:")
        allergy = st.text_input("Dị ứng:")

        if st.button("Lưu thông tin Bệnh nhân") and name:
            # Tạo ID tự động bằng cách sử dụng timestamp
            patient_id = len(data) + 1

            # Tạo DataFrame mới với thông tin bệnh nhân
            new_patient_info = pd.DataFrame({
                "ID": [patient_id],
                "Name": [name],
                "Sex": [sex],
                "Height": [height],
                "Weight": [weight],
                "Date of Birth": [date_of_birth],
                "Blood Type": [blood_type],
                "Allergy": [allergy],
                "Symptoms": [""],
                "Diseases": [""],
                "Medicine": [""]
            })

            # Concatenate the new patient info with the existing DataFrame
            data = pd.concat([data, new_patient_info], ignore_index=True)
            # sort_data_by_id()
            # Lưu DataFrame vào file Excel
            data.to_excel("patient_data.xlsx", index=False)
            st.success("Thông tin Bệnh nhân đã được lưu. ID: {}".format(patient_id))
            # data = data.dropna(how='all')
        else: 
            st.warning("Hãy nhập Họ và tên")
        # Bước tiếp theo nếu đã chọn ID bệnh nhân
    elif section == "Tra cứu thông tin bệnh nhân":
        st.write("## Bạn đang ở phần Tra cứu thông tin bệnh nhân.")
        input_id = st.text_input("Nhập ID bệnh nhân:")
        # Hiển thị toàn bộ thông tin bệnh nhân và các dự đoán, gợi ý thuốc
        if input_id.isdigit():
            patient_info = data[data["ID"] == int(input_id)]
            if not patient_info.empty:
                st.subheader("Thông tin bệnh nhân:")
                st.table(patient_info.iloc[:1, :-3])

                # Display "Diagnosis" in a separate section
                st.subheader("Tiền sử điều trị:")
                st.table(patient_info.iloc[:, -3:])
            else:
                st.warning("Không có thông tin cho ID bệnh nhân đã chọn.")
        else:
            st.warning("Hãy nhập ID")

        
    

    elif section == "Dự đoán bệnh":
        st.write("## Bạn đang ở phần Dự đoán bệnh.")
        input_id=""
        input_id = st.text_input("Nhập ID bệnh nhân:")

        if input_id.isdigit():
            patient_info = data[data["ID"] == int(input_id)]
            if not patient_info.empty:

                # Display "Diagnosis" in a separate section
                st.subheader("Tiền sử điều trị:")
                st.table(patient_info.iloc[:, -3:])
            else:
                st.warning("Không có thông tin cho ID bệnh nhân đã chọn.")
        else:
            st.warning("Hãy nhập ID")

        # Nhập triệu chứng từ người dùng
        symptoms = st.text_input("Nhập triệu chứng :")

        if st.button("Dự đoán bệnh"):

            # Kiểm tra xem người dùng đã nhập triệu chứng hay chưa
            if symptoms:
                # Thực hiện dự đoán bệnh và hiển thị kết quả
                diseases_result = recommend(symptoms)

                st.subheader("Bệnh được dự đoán:")
                
                diseases_list = []
                columns_diseases = ['ID', 'Name', 'symptoms_free','Score']
                diseases_df = pd.DataFrame(diseases_result, columns=columns_diseases)
                for disease in diseases_result:
                    diseases_list.append(disease[1])
                    # st.text(f"{disease[1]} - {disease[2]}")
                st.table(diseases_df)

                # Gợi ý thuốc dựa trên bệnh
                st.subheader("Gợi ý thuốc:")
                medicine_result_1 = suggest_medicine(diseases_result[0][1])
                medicine_list_1 = []
                columns_1 = ['ID', 'Name','Usage', 'formulationAndBrand', 'warning','caution', 'Score']
                medicine_df_1 = pd.DataFrame(medicine_result_1, columns=columns_1)
                for medicine_1 in medicine_result_1:
                    medicine_list_1.append(medicine_1[1])
                st.table(medicine_df_1)
                
                if input_id.isdigit():
                    input_id_0= [int(input_id)]
                else :
                    input_id_0 =0
                
                patient_info = data[data["ID"] == int(input_id)]
                if not patient_info.empty:
                    # Lưu kết quả vào DataFrame
                    new_data = pd.DataFrame({
                        "ID": input_id_0,
                        "Name": patient_info['Name'].values[0],
                        "Sex": patient_info['Sex'].values[0],
                        "Height": patient_info['Height'].values[0],
                        "Weight": patient_info['Weight'].values[0],
                        "Date of Birth": patient_info['Date of Birth'].values[0],
                        "Blood Type": patient_info['Blood Type'].values[0],
                        "Allergy": patient_info['Allergy'].values[0],
                        "Symptoms": [symptoms],
                        "Diseases": [', '.join(diseases_list)],
                        "Medicine": [', '.join(medicine_list_1)]
                    })
                    
                    data = pd.concat([data, new_data], ignore_index=True)
                    # sort_data_by_id()
                    # Lưu DataFrame vào file Excel
                    data.to_excel("patient_data.xlsx", index=False)
        
                    st.success("Dữ liệu đã được lưu thành công ")

            else:
                st.warning("Vui lòng nhập triệu chứng trước khi dự đoán.")

    elif section == "Gợi ý thuốc":
        st.write("## Bạn đang ở phần Gợi ý thuốc.")
        input_id=""
        input_id = st.text_input("Nhập ID bệnh nhân:")

        if input_id.isdigit():
            patient_info = data[data["ID"] == int(input_id)]
            if not patient_info.empty:

                # Display "Diagnosis" in a separate section
                st.subheader("Tiền sử điều trị:")
                st.table(patient_info.iloc[:, -3:])
            else:
                st.warning("Không có thông tin cho ID bệnh nhân đã chọn.")
        else:
            st.warning("Hãy nhập ID")

        # Nhập bệnh từ người dùng
        disease_input = st.text_input("Nhập tên bệnh:")

        if st.button("Gợi ý thuốc"):
            # Kiểm tra xem người dùng đã nhập tên bệnh hay chưa
            if disease_input:
                # Gợi ý thuốc dựa trên bệnh
                medicine_result = suggest_medicine(disease_input)
                st.subheader(f"Gợi ý thuốc cho {disease_input}")
                medicine_list = []
                columns = ['ID', 'Name','Usage', 'formulationAndBrand', 'warning','caution', 'Score']
                medicine_df = pd.DataFrame(medicine_result, columns=columns)
                for medicine in medicine_result:
                    medicine_list.append(medicine[1])
                    # st.text(f"{medicine[1]} - {medicine[2]}")
                st.table(medicine_df)
                if input_id.isdigit():
                    input_id_0= [int(input_id)]
                    

                patient_info = data[data["ID"] == int(input_id)]
                if not patient_info.empty:
                    # Lưu kết quả vào DataFrame
                    new_data = pd.DataFrame({
                        "ID": input_id_0,
                        "Name": patient_info['Name'].values[0],
                        "Sex": patient_info['Sex'].values[0],
                        "Height": patient_info['Height'].values[0],
                        "Weight": patient_info['Weight'].values[0],
                        "Date of Birth": patient_info['Date of Birth'].values[0],
                        "Blood Type": patient_info['Blood Type'].values[0],
                        "Allergy": patient_info['Allergy'].values[0],
                        # "Symptoms": [symptoms],
                        "Diseases": [disease_input],
                        "Medicine": [', '.join(medicine_list)]
                    })
                    
                    data = pd.concat([data, new_data], ignore_index=True)
                    # sort_data_by_id()
                    # Lưu DataFrame vào file Excel
                    data.to_excel("patient_data.xlsx", index=False)
        
                    st.success("Dữ liệu đã được lưu thành công ")

            else:
                st.warning("Vui lòng nhập tên bệnh trước khi gợi ý thuốc.")
    elif section == "Dữ liệu bệnh nhân":
        st.subheader("Toàn bộ dữ liệu")
        st.table(data)

if __name__ == "__main__":
    main()
