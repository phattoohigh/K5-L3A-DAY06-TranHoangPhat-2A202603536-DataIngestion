# Báo cáo cá nhân - Day 6 Data Ingestion Engineering

## Dữ liệu và pipeline

- **Corpus:** ba báo cáo kinh tế World Bank năm 2024 cho Viet Nam, Indonesia và Thailand, tổng cộng 183 trang vật lý và 151 trang nội dung hợp lệ. Trang bìa, mục lục, trang trắng, trang phân cách và bìa sau không được tính. Nguồn, URL, quyền sử dụng, phạm vi trang và SHA-256 được ghi trong `data/manifest.json`.
- **Chuẩn hóa:** trích xuất theo trang bằng `pypdf`, chuẩn hóa Unicode NFC và whitespace, giữ `document_id`, `source`, `page`, `country`, `year`, `topic`, `document_type`, `language`. Header/footer lặp lại được phát hiện ở biên trang và loại trước khi chunk. PDF đều có text layer; không cần OCR.
- **Embedding:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 chiều, vector được normalize. Câu hỏi tiếng Việt có `retrieval_query` tiếng Anh được dịch thủ công để phù hợp corpus, không đưa answer hoặc gold source/page vào query.
- **Vector store:** PostgreSQL 16 + pgvector 0.8.7, cosine similarity, exact search, dữ liệu lưu bền vững. Upsert dùng stable chunk ID và xóa chunk cũ của cùng document/strategy để tránh nhân đôi.
- **Hybrid retrieval:** lấy top-20 dense candidates rồi rerank bằng IDF-weighted lexical overlap. Dense weight 0.35 được chọn trên dev; cùng cơ chế áp dụng cho A/B/C. Chi phí dịch vụ bằng 0 vì embedding chạy local.

## Cấu hình và benchmark

- **A:** fixed-size gần 800 ký tự, overlap 100, dịch điểm cắt về whitespace để không cắt giữa từ; không metadata filter.
- **B:** sentence-aware chunking, tối đa 650 ký tự; không metadata filter.
- **C:** chunks và vectors giống hệt B; áp dụng `country` filter khi filter được cung cấp trước trong benchmark.
- **Benchmark cuối:** 15 câu dev và 6 câu test v4. Test v4 dùng các trang evidence chưa xuất hiện trong các test pilot. Có câu direct, paraphrase, condition và ba câu test có country filter. Evidence được kiểm tra trực tiếp trong PDF trước khi chạy.
- **Quy tắc chọn:** ưu tiên dev Evidence Hit@3, sau đó grounding, Hit@1, coherence và tài nguyên. Kích thước 650 và hybrid dense weight 0.35 được chọn chỉ trên dev; v4 test được chạy một lần và không dùng để tune tiếp.

## Kết quả retrieval

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Tổng thời gian chạy | Chi phí |
|---|---|---:|---:|---:|---:|---:|
| A | Dev | 6/15 | 8/15 | 721 | 12.57 giây (toàn pipeline) | 0 |
| B | Dev | 9/15 | 11/15 | 915 | 12.57 giây (toàn pipeline) | 0 |
| C | Dev | 9/15 | 11/15 | 915 | 12.57 giây (toàn pipeline) | 0 |
| A | Test | 1/6 | 1/6 | 721 | 11.68 giây (toàn pipeline) | 0 |
| B | Test | 3/6 | 3/6 | 915 | 11.68 giây (toàn pipeline) | 0 |
| C | Test | 3/6 | 3/6 | 915 | 11.68 giây (toàn pipeline) | 0 |

Thời gian trên là phép đo một lần trên máy local và bao gồm đọc PDF, chunk, embedding, upsert và query; không nên xem là benchmark hiệu năng tổng quát.

## Audit năm khía cạnh

| Strategy | Coherence trung bình /2 (10 vị trí) | Metadata đủ và đúng | Filter compliance | Grounding 0/1/2 và trung bình /2 |
|---|---:|---:|---:|---:|
| A | 1.0 | 15/15 | N/A | 1/4/1; 1.00 |
| B | 1.8 | 15/15 | N/A | 0/3/3; 1.50 |
| C | 1.8 | 15/15 | 3/3 câu filter (9/9 top-3 records) | 0/3/3; 1.50 |

- **Retrieval accuracy:** B/C cải thiện mạnh so với A trên dev và test. Test nhỏ tạo dao động lớn; kết quả không chứng minh ưu thế tổng quát ngoài corpus.
- **Chunk coherence:** A thường bắt đầu hoặc kết thúc giữa ý. B/C giữ câu hoàn chỉnh hơn; hai lỗi còn lại đến từ text extraction làm header xen vào câu hoặc mất vài từ ở biên.
- **Metadata usefulness:** 45/45 records được kiểm tra có `source`, `page`, `country`, `year`, `topic` đầy đủ và khớp manifest. C tuân thủ cả ba country filter; B không áp dụng filter theo thiết kế.
- **Grounding:** B/C có evidence đầy đủ cho VN12, ID13 và TH12. Các câu còn lại thường truy xuất đúng chủ đề nhưng thiếu số liệu cụ thể, nên chỉ đạt mức 1 hoặc 0.
- **Data strategy impact:** A->B cho thấy sentence-aware chunking tăng coherence và retrieval. B->C không tăng aggregate Hit@3 trên test v4 vì dense/lexical retrieval đã trả về đúng quốc gia ở nhiều câu, nhưng C đảm bảo phạm vi filter và tăng tính an toàn khi các báo cáo có nội dung tương tự.

Các file audit: `starter/results/coherence_audit.csv`, `starter/results/metadata_audit.csv`, `starter/results/grounding_audit.csv`. Trace đầy đủ nằm trong `starter/results/dev/` và `starter/results/test/`.

## Ví dụ lỗi và phân tích

1. **VN13:** top-3 có nội dung về under-execution của đầu tư công nhưng không chứa đồng thời `VND580 trillion` và `73.5 percent`. Lỗi là retrieval theo chủ đề đúng nhưng không đủ evidence định lượng.
2. **TH13:** top-3 nhắc budget delay và fiscal consolidation nhưng bỏ lỡ trang 24 chứa `3.48 trillion baht` và `18.7 percent of GDP`. Lexical reranking ưu tiên các đoạn có nhiều từ về fiscal budget nhưng không đúng con số được hỏi.
3. **ID12:** câu hỏi yêu cầu hai tỷ lệ trong cùng câu; retrieved chunks nói về revenue pressure nhưng không chứa đủ cả `7.7` và `10.9`. Đây là ví dụ grounding không đầy đủ dù topic match.

## Kết luận và giới hạn

C được chọn vì đồng hạng cao nhất với B về Hit@3, đạt Hit@1 tốt nhất trên dev, giữ coherence cao và bảo đảm metadata-filter compliance. Hybrid reranking cải thiện top-rank precision nhưng không giải quyết mọi câu hỏi định lượng.

Giới hạn chính gồm corpus chỉ có ba báo cáo cùng năm, test v4 chỉ có sáu câu, câu hỏi tiếng Việt trong khi corpus tiếng Anh, exact evidence metric nghiêm ngặt, và audit thủ công do tác giả thực hiện với hỗ trợ AI nên có nguy cơ thiên lệch. Các test pilot trước đó đã bị xem và không được báo như holdout độc lập. Nếu mở rộng, cần thêm nhiều quốc gia/năm, benchmark lớn hơn, đánh giá nhiều người chấm và thử embedding model mạnh hơn với cùng protocol dev/test.

## Mở rộng tùy chọn: hybrid retrieval

Trên cùng strategy C và cùng 15 câu dev, dense-only đạt Hit@1 `8/15`, Hit@3 `11/15`; hybrid đạt Hit@1 `9/15`, Hit@3 `11/15`. Hybrid cải thiện vị trí top-1 nhưng không tăng coverage top-3, vì vậy được giữ như một cải tiến precision có giới hạn thay vì tuyên bố tốt hơn tổng quát. Bảng ablation nằm tại `artifacts/hybrid_ablation.csv`.

## Sử dụng AI

AI được dùng để giải thích yêu cầu, hỗ trợ review code, đề xuất câu hỏi, kiểm tra trace. Tác giả kiểm tra evidence trong PDF, giữ log chạy thật và không tạo số liệu benchmark chưa chạy.
