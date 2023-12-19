import streamlit as st
import pandas as pd

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

# Hàm dự đoán bệnh dựa trên triệu chứng
def perform_diagnosis(symptoms_1):
    # Thực hiện dự đoán bệnh ở đây
    # Đây là nơi để tích hợp logic của bạn
    # Ví dụ đơn giản: Trả về danh sách bệnh
    list_Diseases = []
    A = "Bệnh A cho triệu chứng " + symptoms_1
    B = "Bệnh B cho triệu chứng " + symptoms_1
    list_Diseases = [A, B]
    return list_Diseases

# Hàm gợi ý thuốc dựa trên bệnh
def suggest_medicine(disease):
    # Thực hiện gợi ý thuốc ở đây
    # Đây là nơi để tích hợp logic của bạn
    # Ví dụ đơn giản: Trả về danh sách thuốc cho mỗi bệnh
    list_Medicine = []
    A = "Thuốc 1 cho " + disease
    B = "Thuốc 2 cho " + disease
    list_Medicine = [A, B]
    return list_Medicine

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
                diseases_result = perform_diagnosis(symptoms)

                st.subheader("Bệnh được dự đoán:")
                for disease in diseases_result:
                    st.text(disease)
                
                # Gợi ý thuốc dựa trên bệnh
                medicine_result = suggest_medicine(diseases_result[0])
                st.subheader("Gợi ý thuốc:")
                for medicine in medicine_result:
                    st.text(medicine)

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
                        "Diseases": [', '.join(diseases_result)],
                        "Medicine": [', '.join(medicine_result)]
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
                for medicine in medicine_result:
                    st.text(medicine)
                    
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
                        "Medicine": [', '.join(medicine_result)]
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
