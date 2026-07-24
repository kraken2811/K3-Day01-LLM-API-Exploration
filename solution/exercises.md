# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
Khi temperature tăng, câu trả lời thường trở nên sáng tạo và đa dạng hơn, nhưng cũng có thể ít nhất quán và ít chính xác hơn. Ở temperature thấp, phản hồi thường ngắn gọn, ổn định và tập trung vào thông tin cốt lõi. Ở temperature cao, model có xu hướng diễn đạt tự do hơn, thêm biến thể và đôi khi đưa ra ý tưởng ít chắc chắn hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
Tôi sẽ chọn temperature khoảng 0.2 đến 0.4 vì đây là mức đủ tự nhiên để câu trả lời không bị cứng nhắc, nhưng vẫn giữ được độ nhất quán, an toàn và phù hợp với ngữ cảnh hỗ trợ khách hàng.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
Với 10.000 người dùng mỗi ngày, mỗi người gọi 3 lần, tổng là 30.000 lượt gọi. Nếu mỗi lần đầu ra khoảng 350 token, workload này sẽ tạo khoảng 10,5 triệu token đầu ra mỗi ngày. Nếu GPT-4o đắt khoảng 5 đến 10 lần GPT-4o-mini, thì chi phí cho workload này cũng sẽ cao hơn tương ứng 5 đến 10 lần. GPT-4o xứng đáng khi dùng cho các tác vụ cần chất lượng cao, suy luận tốt và phản hồi chính xác như hỗ trợ kỹ thuật phức tạp hoặc phân tích tài liệu. GPT-4o-mini phù hợp hơn cho chatbot đơn giản, tóm tắt nhanh, trả lời FAQ hoặc các tình huống không cần độ chính xác cực kỳ cao.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
Với system prompt là giáo viên tiểu học, câu trả lời sẽ ngắn hơn, dùng ngôn ngữ đơn giản và ví dụ gần gũ như việc “giữ sổ sách chung cho nhiều người”. Với system prompt là chuyên gia tài chính, câu trả lời sẽ dài hơn, dùng nhiều thuật ngữ chuyên môn như “cơ sở dữ liệu phân tán”, “thỏa thuận thông minh” và “không cần trung gian”. Hai phản hồi khác nhau rõ ở mức độ chi tiết, độ khó của từ vựng và cách trình bày ví dụ. System prompt ảnh hưởng rất mạnh đến hành vi model vì nó hướng model vào một vai trò, phong cách và mức độ chuyên môn nhất định.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
Ví dụ với một đoạn văn tiếng Việt khoảng 100 từ, tiktoken có thể tính được khoảng 140 token, còn ước lượng theo công thức số từ chia cho 0.75 cho ra khoảng 133 token. Chênh lệch giữa hai con số này khoảng 5% đến 10%, tùy vào cách tokenizer chia cụm từ và dấu câu. Tiếng Việt thường tốn nhiều token hơn tiếng Anh cùng độ dài vì hệ thống tokenizer thường chia các từ và ký tự tiếng Việt thành nhiều đơn vị nhỏ hơn, đồng thời các dấu câu, từ ghép và cách viết không dùng khoảng trắng rõ ràng cũng làm tăng số token.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
Streaming rất quan trọng khi người dùng cần thấy phản hồi ngay lập tức, như trong chatbot tương tác, trợ lý ảo, hoặc các công cụ hỗ trợ nhập liệu thời gian thực. Nó làm cho trải nghiệm cảm giác nhanh hơn và giúp người dùng thấy hệ thống đang làm việc thay vì chờ đợi một câu trả lời lâu. Non-streaming phù hợp hơn khi câu trả lời dài, cần được hoàn chỉnh trước khi hiển thị, hoặc khi ứng dụng ưu tiên sự rõ ràng và dễ kiểm soát hơn là tốc độ phản hồi ban đầu.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
Exponential backoff tốt hơn delay cố định vì nó giúp phân tán lại tải mà không làm cho hệ thống bị tấn công bởi một đợt retry đồng loạt. Khi API bị quá tải, mỗi client sẽ chờ lâu hơn dần theo cấp số nhân, nên ít có khả năng gây ra một “sóng” retry khiến hệ thống bị quá tải thêm. Nếu hàng nghìn client cùng retry sau một khoảng delay cố định giống nhau, chúng có thể đồng loạt gửi lại yêu cầu cùng lúc, tạo ra một đợt tải cực lớn và làm tình trạng quá tải trở nên tồi tệ hơn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
Tôi chọn persona là một trợ lý học tập tiếng Việt thân thiện, ngắn gọn và dễ hiểu. System prompt có thể là: “Bạn là một trợ lý học tập tiếng Việt, trả lời ngắn gọn, rõ ràng và dễ hiểu. Luôn giải thích bằng ví dụ đơn giản, ưu tiên tiếng Việt, và không đưa ra thông tin không chắc chắn.” Việc yêu cầu “trả lời ngắn gọn” giúp phản hồi không dài dòng và dễ sử dụng, còn việc chỉ định “ưu tiên tiếng Việt” giúp câu trả lời phù hợp với người dùng trong môi trường học tập.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
Hạn chế lớn nhất là trợ lý chưa có bộ nhớ dài hạn, nên nó không thể nhớ sở thích hoặc tình huống trước đó của người dùng qua nhiều phiên làm việc. Một cải thiện cụ thể là thêm cơ chế lưu ngắn gọn lịch sử hội thoại vào cơ sở dữ liệu hoặc session store, rồi tự động đưa lại thông tin đó vào mỗi lần gọi tiếp theo. Cách triển khai khá đơn giản: lưu các lượt trao đổi dưới dạng JSON, gắn với một user_id và chèn lại vào prompt trước khi gọi API.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
