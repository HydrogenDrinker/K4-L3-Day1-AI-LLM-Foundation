# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi tăng temperature từ 0.0 lên 1.5, phản hồi của mô hình dịch chuyển rõ rệt từ tính đơn định, quy chuẩn sang tính ngẫu nhiên và sáng tạo cao. Ở mức 0.0 và 0.5, câu trả lời tập trung vào các sự thật phổ biến, chuẩn xác, văn phong mạch lạc và lặp lại cấu trúc ổn định giữa các lần chạy. Khi tăng lên 1.0 và đặc biệt 1.5, mô hình lựa chọn các từ ngữ bất ngờ hơn nhưng bắt đầu xuất hiện hiện tượng bịa đặt thông tin (hallucination, ví dụ nhầm lẫn vị trí địa lý hoặc nhân vật lịch sử) và tính liên kết giữa các câu giảm rõ rệt.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Đối với chatbot hỗ trợ khách hàng, tôi sẽ đặt temperature ở mức thấp, khoảng từ 0.1 đến 0.3. Mức nhiệt độ này đảm bảo câu trả lời có tính nhất quán, chính xác cao và bám sát thông tin cơ sở dữ liệu/chính sách của doanh nghiệp, đồng thời giảm thiểu tối đa rủi ro mô hình hallucinate (bịa đặt giá tiền, khuyến mãi hay điều khoản bảo hành gây hiểu lầm cho người dùng).

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Tổng lượng token output mỗi ngày là: 10.000 x 3 x 350 = 10.500.000 token (10.500 đơn vị 1K token). Chi phí output hàng ngày với GPT-4o ($0.010/1K) là 10.500 x $0.010 = $105/ngày (~$3.150/tháng); với GPT-4o-mini ($0.0006/1K) là 10.500 x $0.0006 = $6.30/ngày (~$189/tháng). Như vậy, GPT-4o đắt hơn GPT-4o-mini đúng 16.67 lần (~16.7 lần) cho workload này. GPT-4o hoàn toàn xứng đáng với chi phí khi cần phân tích hợp đồng pháp lý phức tạp, tư vấn tài chính chuyên sâu hoặc lập luận logic nhiều bước đòi hỏi tính chuẩn xác tuyệt đối. Ngược lại, GPT-4o-mini là lựa chọn tối ưu cho các tác vụ hỏi đáp thông tin cơ bản (FAQ), phân loại ý định (intent classification) hoặc tóm tắt hội thoại ngắn với lưu lượng lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Hai phản hồi có sự phân hóa sâu sắc: persona giáo viên tiểu học sử dụng phép ẩn dụ trực quan gần gũi ("cuốn sổ tay số chung", "các trang sổ kết nối"), cấu trúc câu ngắn dạng gạch đầu dòng dễ hiểu và từ vựng giản dị; trong khi persona chuyên gia tài chính sử dụng thuật ngữ học thuật chuyên sâu như "công nghệ sổ cái phân tán (DLT)", "ký số", "tính bất biến", "mật mã học". System prompt đóng vai trò như một cơ chế định hướng ngữ cảnh toàn cục (system steering), chi phối trực tiếp không gian phân phối từ vựng, độ phức tạp của cú pháp, độ dài phản hồi và góc nhìn tiếp cận vấn đề của mô hình mà không cần thay đổi prompt câu hỏi của người dùng.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Thử nghiệm trên đoạn văn tiếng Việt 123 từ: công thức ước lượng (123 / 0.75) cho ra 164 token, trong khi tiktoken với bảng mã o200k_base (GPT-4o) đếm được 147 token (chênh lệch -10.4%) và bảng mã cl100k_base (GPT-4) đếm tới 269 token (chênh lệch +64.0%). Tiếng Việt thường tốn nhiều token hơn tiếng Anh cùng độ dài vì các thuật toán phân tách token Byte-Pair Encoding (BPE) được huấn luyện chủ yếu trên kho ngữ liệu tiếng Anh, nơi các từ hoặc gốc từ đã có sẵn trong từ điển; trong khi tiếng Việt có nhiều ký tự có dấu thanh và nguyên âm/phụ âm đặc thù (ă, â, đ, ê, ô, ơ, ư...) trong mã hóa UTF-8 chiếm 2-3 byte và hiếm gặp trong bộ từ điển BPE gốc, khiến tokenizer phải chia nhỏ một từ tiếng Việt thành nhiều subword hoặc byte riêng biệt (hiện tượng token inflation).

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng và phát huy hiệu quả cao nhất trong các ứng dụng có tính tương tác trực tiếp với người dùng như giao diện chatbot trò chuyện, trợ lý ảo hoặc công cụ hỗ trợ viết nội dung, vì nó tối ưu hóa chỉ số Time-To-First-Token (TTFT), mang lại cảm giác phản hồi tức thì và cho phép người dùng đọc dần nội dung ngay khi từng từ được sinh ra thay vì phải chờ đợi toàn bộ câu trả lời hoàn tất. Ngược lại, non-streaming lại phù hợp hơn đối với các tác vụ xử lý ngầm ở phía backend (background workers), các pipeline xử lý dữ liệu hàng loạt (batch ETL), hoặc khi hệ thống yêu cầu cấu trúc dữ liệu hoàn chỉnh (như JSON schema hoặc tool/function call arguments) trước khi có thể parse cú pháp và chuyển tiếp dữ liệu sang dịch vụ tiếp theo.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> So với delay cố định, exponential backoff tăng dần khoảng thời gian chờ theo cấp số nhân sau mỗi lần thất bại (ví dụ 0.1s -> 0.2s -> 0.4s -> 0.8s...), giúp giãn mật độ request theo thời gian và tạo ra khoảng lặng cần thiết cho hệ thống server/mạng kịp giải phóng tài nguyên và tự phục hồi sau sự cố. Nếu hàng nghìn client cùng gặp lỗi và retry với delay cố định giống nhau (ví dụ đều chờ 1 giây), hiện tượng "thundering herd" (cơn bão retry) sẽ xuất hiện: tất cả các client sẽ đồng loạt gửi lại request tại cùng một thời điểm, tiếp tục tạo ra đỉnh tải đột biến (traffic spike) đánh sập máy chủ một lần nữa, khiến tình trạng tắc nghẽn trở nên trầm trọng hơn và hệ thống không thể hồi phục.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona là: "Bạn là trợ giảng thân thiện, chuyên môn cao của khóa học AI LLM; trả lời cô đọng, súc tích bằng tiếng Việt chuẩn mực, tập trung giải thích bản chất kỹ thuật kèm ví dụ code minh họa". Lựa chọn từ ngữ "cô đọng, súc tích" là cốt lõi để hạn chế độ dài output của trợ lý, giúp phản hồi tập trung vào trọng tâm, đồng thời tiết kiệm đáng kể số lượng token tích lũy trong lịch sử hội thoại nhiều lượt; cụm từ "bằng tiếng Việt chuẩn mực kèm ví dụ code minh họa" đảm bảo trợ lý duy trì ngôn ngữ tiếng Việt nhất quán ngay cả khi người dùng đưa vào các thuật ngữ lập trình tiếng Anh, đồng thời luôn kết nối kiến thức lý thuyết với code thực tế.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của trợ lý hiện tại là cửa sổ ngữ cảnh dạng trượt (sliding window) chỉ lưu giữ tối đa 3 lượt hội thoại gần nhất (6 messages) và hoàn toàn không có bộ nhớ dài hạn (persistent memory), khiến trợ lý nhanh chóng quên các thông tin, ngữ cảnh hoặc ràng buộc quan trọng mà người dùng đã thiết lập ở đầu cuộc trò chuyện. Một giải pháp cải thiện cụ thể là kỹ thuật "tóm tắt ngữ cảnh định kỳ kết hợp tệp lưu trữ" (context summarization with persistent state): khi lịch sử đạt đến ngưỡng giới hạn, hệ thống gọi một hàm ngầm dùng model mini để tóm tắt các lượt hội thoại trước đó thành một đoạn bullet-point ngắn gọn và đưa vào làm bối cảnh bổ sung trong system prompt; đồng thời lưu trữ lịch sử và profile người dùng vào cơ sở dữ liệu (như SQLite hoặc Vector DB) để có thể phục hồi và truy vấn ngữ cảnh liên quan (RAG) qua các phiên làm việc khác nhau.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
