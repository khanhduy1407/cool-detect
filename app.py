from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
# Định nghĩa class
class_name = ['ok', 'xxx']

# Load model đã train
my_model = load_model("cool_model")


newTrain_data_folder = 'data/New_Train/'
newTrain_folder_list = ['OK', 'XXX']
# Tạo các thư mục nếu chưa tồn tại
for folder in newTrain_folder_list:
    os.makedirs(os.path.join(newTrain_data_folder, folder), exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files or 'label' not in request.form:
        return jsonify({"error": "No file or label part"}), 400

    file = request.files['file']
    label = request.form['label']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if label not in newTrain_folder_list:
        return jsonify({"error": "Invalid label"}), 400

    # Lưu tệp vào thư mục tương ứng
    file_path = os.path.join(newTrain_data_folder, label, file.filename)
    file.save(file_path)

    return jsonify({"success": f"File saved to {file_path}"}), 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Kiểm tra xem file có được gửi hay không
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        # Nếu người dùng không chọn file, trình duyệt có thể gửi một file rỗng không tên
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file:
            # Lưu file tạm thời
            temp_file_path = os.path.join('temp', file.filename)
            file.save(temp_file_path)

            # Đọc ảnh
            image_org = cv2.imread(temp_file_path)

            # Xử lý trường hợp không thể đọc ảnh
            if image_org is None:
                os.remove(temp_file_path)
                return jsonify({"error": "Unable to read the image file"}), 400

            # Resize
            image = image_org.copy()
            image = cv2.resize(image, (200, 200))

            # Chuyển đổi sang tensor
            image = np.expand_dims(image, axis=0)

            # Dự đoán
            predict = my_model.predict(image)
            predicted_class = class_name[np.argmax(predict)]

            # Xóa file tạm
            os.remove(temp_file_path)

            # Trả về kết quả
            return jsonify({"result": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#Tim transparents

def calculate_angle_with_vertical_edge(left_vertical_edge_point1, left_vertical_edge_point2):
    # Tính vector từ điểm đầu đến điểm cuối của cạnh dọc
    vertical_edge_vector = np.array(left_vertical_edge_point2) - np.array(left_vertical_edge_point1)

    # Tính góc giữa cạnh dọc và trục Tung
    angle_radians = np.arctan2(vertical_edge_vector[1], vertical_edge_vector[0])
    angle_degrees = np.degrees(angle_radians)

    # Chuyển đổi góc về khoảng từ 0 đến 180 độ
    angle_degrees = angle_degrees if angle_degrees >= 0 else 180 + angle_degrees

    return angle_degrees
def detect_transparent_regions(image_data):
    # Đọc ảnh từ dữ liệu nhận được
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # Kiểm tra kích thước của hình ảnh
    heightBG, widthBG, _ = image.shape

    # Kiểm tra xem hình ảnh có kênh alpha không (có phải hình ảnh có phần trong suốt hay không)
    if image.shape[2] == 4:
        # Khởi tạo một danh sách để lưu trữ kết quả
        transparent_regions_info = []

        # Tìm các vùng trong suốt
        transparent_regions = cv2.inRange(image[:, :, 3], 0, 0)

        # Tìm các contour của các vùng trong suốt
        contours, _ = cv2.findContours(transparent_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sắp xếp các contour theo vị trí x của hộp chứa tối thiểu
        contours = sorted(contours, key=lambda x: cv2.minAreaRect(x)[0][0])

        # Biến đếm số thứ tự cho các ô
        order_number = 1

        # Lặp qua từng contour để lấy thông tin về vị trí, kích thước và độ nghiêng
        for contour in contours:

            # Tính toán hình chữ nhật bao quanh contour
            x, y, w, h = cv2.boundingRect(contour)

            # Tính toán hộp chứa tối thiểu và góc quay của contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Tính toán tọa độ của hai điểm đối diện
            x_min = min(box[:, 0])
            x_max = max(box[:, 0])
            y_min = min(box[:, 1])
            y_max = max(box[:, 1])

            # Kiểm tra kích thước của hộp chứa
            if w >= 150 and h >= 150:

                # Tính toán tâm của hình chữ nhật
                x_center = x + w // 2
                y_center = y + h // 2

                # Tính toán tâm của hình chữ nhật
                x = int((x_min + x_max) / 2)
                y = int((y_min + y_max) / 2)

                # Xác định hướng của ô trống
                orientation = "Ngang" if w > h else "Dọc"

                # Kiểm tra rect[1][0] và rect[1][1] để gán width và height phù hợp với hướng
                if orientation == "Ngang":
                    if rect[1][0] > rect[1][1]:
                        width = rect[1][0]
                        height = rect[1][1]
                    else:
                        width = rect[1][1]
                        height = rect[1][0]
                else:  # orientation == "Dọc"
                    if rect[1][0] > rect[1][1]:
                        height = rect[1][0]
                        width = rect[1][1]
                    else:
                        height = rect[1][1]
                        width = rect[1][0]

                # Tính toán góc nghiêng của trục Oxy
                angle_radians = np.arctan2(y_center, x_center)
                angle_degrees = np.degrees(angle_radians)

                # Sắp xếp lại các điểm đỉnh của hình chữ nhật theo chiều kim đồng hồ bắt đầu từ góc dưới bên trái
                sorted_box = sorted(box, key=lambda x: x[0] + x[1])
                # Kiểm tra hướng để xác định cạnh dọc bên trái
                if orientation == "Ngang":
                    left_vertical_edge_point1 = sorted_box[0]  # Điểm đỉnh thứ nhất (D12)
                    left_vertical_edge_point2 = sorted_box[1]  # Điểm đỉnh thứ tư (D13)
                else:  # orientation == "Dọc"
                    left_vertical_edge_point1 = sorted_box[0]  # Điểm đỉnh thứ hai (D12)
                    left_vertical_edge_point2 = sorted_box[2]  # Điểm đỉnh thứ ba (D13)

                # Chuyển đổi sang kiểu dữ liệu thông thường để tránh lỗi JSON serializable
                left_vertical_edge_point1 = left_vertical_edge_point1.tolist()
                left_vertical_edge_point2 = left_vertical_edge_point2.tolist()

                angle = calculate_angle_with_vertical_edge(left_vertical_edge_point1, left_vertical_edge_point2)

                # Tính bán kính của contour
                r = max(width, height) / 2

                # Nếu contour gần giống với một hình tròn, in ra bán kính
                if abs(width - height) <= 10:
                    transparent_regions_info.append({
                        "order_number": order_number,
                        "type": "circle",
                        "x": x_center,
                        "y": y_center,
                        "width": r + r ,
                        "height": r + r ,
                        "rotation": 0,
                        "left_vertical_edge": {
                             "start_point": {
                                "x": x_center - r ,
                                "y": y_center - r
                            },
                            "end_point": {
                                "x": left_vertical_edge_point2[0],
                                "y": left_vertical_edge_point2[1]
                            }
                        }
                    })
                else:
                    # In ra thông tin về vị trí, kích thước và độ nghiêng của ô trống
                    # Thêm các thông tin vào danh sách thông tin vùng trong suốt
                    transparent_regions_info.append({
                        "order_number": order_number,
                        "type": "rectangle",
                        "x": x_center,
                        "y": y_center,
                        "width": width,
                        "height": height,
                        "rotation": -(90 - angle),
                        "left_vertical_edge": {
                            "start_point": {
                                "x": left_vertical_edge_point1[0],
                                "y": left_vertical_edge_point1[1]
                            },
                            "end_point": {
                                "x": left_vertical_edge_point2[0],
                                "y": left_vertical_edge_point2[1]
                            }
                        }
                    })

                # Tăng số thứ tự cho ô tiếp theo
                order_number += 1

        total_transparent_regions = len(transparent_regions_info)

        return transparent_regions_info, total_transparent_regions, heightBG, widthBG

    else:
        return None, 0, heightBG, widthBG

@app.route('/detect_transparent_regions', methods=['POST'])
def process_image():
    # Nhận dữ liệu ảnh từ yêu cầu POST
    image_file = request.files['image']
    image_data = image_file.read()

    # Phát hiện các vùng trong suốt và lấy thông tin về chúng
    transparent_regions_info, total_transparent_regions, image_height, image_width = detect_transparent_regions(image_data)
    if total_transparent_regions == 0:
        response = {
            "BG_width": image_width,
            "BG_height": image_height,
            "Total_trans": 0
        }
    else:
        response = {
            "BG_width": image_width,
            "BG_height": image_height,
            "Total_trans": total_transparent_regions,
            "transparent_regions": transparent_regions_info
        }
    # Sắp xếp lại danh sách các vùng trong suốt theo order_number
    if "transparent_regions" in response:
        # Sắp xếp các vùng trong suốt nếu chúng tồn tại
        response["transparent_regions"] = sorted(response["transparent_regions"], key=lambda x: x["order_number"])

    return jsonify(response)


if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')
    app.run()
