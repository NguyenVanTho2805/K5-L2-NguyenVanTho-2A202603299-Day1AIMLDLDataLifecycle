# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU (ô setup in `Device: cpu`)

**Python / PyTorch / Ultralytics:** Python 3.13.18 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không. Chạy nguyên bản notebook, không đổi checkpoint, ngưỡng hay mã nguồn.

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…

   **Trả lời:** `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

   **Trả lời:** Cả bức ảnh, vì record chỉ có một lớp xếp hạng, không có box hay polygon chỉ vị trí.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

   **Trả lời:** Đi kèm taxonomy và checkpoint (ImageNet-1K, 1000 lớp cố định); model chỉ chọn trong danh sách đó.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

   **Trả lời:** Để tra đúng bảng lớp: id và tên chỉ có nghĩa trong một taxonomy cụ thể (id 468 không tồn tại trong COCO-80, chỉ có id 0–79).

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

   **Trả lời:** Tiêu chí chọn một lớp (vật chính, vật lớn nhất hoặc chủ đề chính); ca mơ hồ thì escalation.

6. Vì sao không được lấy `score` của model làm ground truth?

   **Trả lời:** `score` là độ tin cậy do model tự tính, không phải nhãn do người xác nhận; model có thể tự tin mà vẫn sai.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…

   **Trả lời:** `class_name`=bowl ; `score`=0.701283 ; `bbox_xyxy`=[155.02, 168.67, 182.38, 184.18] ; `bbox_width`=27.36 ; `bbox_height`=15.51

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

   **Trả lời:** Góc trên-trái ở (155.02, 168.67), góc dưới-phải ở (182.38, 184.18); tính bằng pixel, gốc (0, 0) ở góc trên-trái ảnh. Box rộng 27.36 px, cao 15.51 px.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

   **Trả lời:** Chọn hai ngưỡng 0.35 và 0.60. Ngưỡng 0.35 (ngưỡng tạo JSON) có 11 prediction; ngưỡng 0.60 chỉ còn 6 prediction (person 0.91, bowl 0.72, bowl 0.70, oven 0.69, oven 0.63, person 0.61), mất 5 hộp score thấp (bowl 0.50, bowl 0.46, cup 0.45, cup 0.38, bowl 0.38). Hộp `bowl` (score 0.701283) còn ở cả hai ngưỡng.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

   **Trả lời:** Kiểm tra thêm các hộp score thấp (có thể sai lớp, thừa hoặc trùng).

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

   **Trả lời:** Box ôm sát phần nhìn thấy của vật, mỗi vật một box riêng.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

   **Trả lời:** `person` 0.61 nằm sát mép trái ảnh nên bị cắt một phần. Tình huống này thường gặp nên viết thành quy tắc trong guideline: box ôm sát phần còn nhìn thấy trong khung ảnh; chỉ hỏi Lab Coach (escalation) khi không đoán được đó là vật gì.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…

   **Trả lời:** `instance_id`=kitchen-002 ; `class_name`=bowl ; `score`=0.735743 ; `polygon_point_count`=67 ; 5 điểm đầu của `polygon_xy`=[[53.0, 344.0], [52.0, 345.0], [50.0, 345.0], [49.0, 346.0], [48.0, 346.0]]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

   **Trả lời:** Polygon bám biên thật của vật, còn box chỉ là hình chữ nhật bao quanh; polygon chiếm khoảng 80% diện tích box (khoảng 2325 / 2902 px²), phần còn lại là nền.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

   **Trả lời:** `instance_id` chỉ để phân biệt từng instance trong output lab; nó không cho biết loại vật (class) và không theo dõi vật qua nhiều frame (tracking).

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

   **Trả lời:** Biên polygon bám sát mép nhìn thấy của vật, không lấn sang nền hay vật khác.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

   **Trả lời:** `bowl` 0.74 (`kitchen-002`) và `bowl` 0.53 nằm sát nhau ở góc dưới-trái, chỗ hai vật chạm nhau khó vẽ biên (đoạn biên thẳng đứng x=32 của `kitchen-002` có thể là chỗ bát kia che). Đây là ca thường gặp nên dùng guideline: vẽ theo phần nhìn thấy và tách thành hai instance; chỉ escalation khi không phân định được biên.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho cả ảnh | `traffic` top-1 là `cab`, score chỉ 0.51 | Chọn một lớp theo guideline; ca mơ hồ thì hỏi | Lớp đúng taxonomy và đúng quy tắc chọn |
| Phát hiện vật thể | Mỗi vật một lớp và một box | `bowl` chỉ 27×15 px, lệch vài px là sai nhiều; `person` 0.61 sát mép trái ảnh, bị cắt một phần | Vẽ box ôm sát, mỗi vật một box | Box sát, đúng lớp, không sót, không trùng |
| Instance segmentation | Mỗi instance một lớp và một polygon | `kitchen-002` biên bậc thang; `bowl` 0.74 và `bowl` 0.53 chạm nhau ở góc dưới-trái, khó vẽ biên; mặt nạ `person` 0.43 phủ cả mặt bàn (nghi sai lớp) | Vẽ polygon bám mép; tách vật chạm nhau | Biên không lấn nền hay dính hai vật; trả về làm lại nếu sai |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

   **Trả lời:** Chỉ dùng ba ảnh COCO công khai của notebook, không tải ảnh cá nhân hay dữ liệu nội bộ, và không ghi họ tên, MSSV vào báo cáo hay output.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

   **Trả lời:** Dừng lại và báo Lab Coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
