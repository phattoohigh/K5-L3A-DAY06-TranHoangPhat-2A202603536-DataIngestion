# Day 06 · Data Ingestion Engineering

**Lab cá nhân** · **4 giờ** · **Dữ liệu tự chọn**

Biến tài liệu thành kho tri thức có thể truy xuất cho AI agent, rồi đo xem cách chuẩn bị dữ liệu nào hiệu quả nhất.

| 📄 Dữ liệu | 🧩 Chiến lược | 🎯 Benchmark |
|---|---|---|
| Tối thiểu **30 trang nội dung** | Tối thiểu **3 cấu hình** | Tối thiểu **20 câu hỏi** |

## Bắt đầu nhanh

Dùng [starter PostgreSQL + pgvector](starter/README.md) để dựng database và hoàn thiện các hàm TODO, hoặc tự chọn vector store tương đương. Starter chỉ là khung: **bạn tự hiện thực pipeline và benchmark**.

## 1. Xây pipeline

![Đọc tài liệu, chuẩn hóa, chia chunk, tạo vector, lưu và kiểm tra](assets/ingestion-pipeline.svg)

Chọn tài liệu được phép sử dụng; giữ **ID tài liệu, nguồn, trang/vị trí** và metadata phù hợp. Dùng embedding model thật qua API hoặc chạy local; lưu và truy xuất bằng pgvector, Qdrant, Chroma hoặc một store tương đương.

## 2. So sánh và đánh giá

![Ba chiến lược được đánh giá trên cùng corpus và benchmark dev/test](assets/benchmark-workflow.svg)

Dùng cùng dữ liệu, model và câu hỏi cho cả ba cấu hình. Tách **15 câu dev / 5 câu test** trước khi chạy; chọn cấu hình trên dev, rồi đánh giá test sau khi cố định. Mỗi câu hỏi cần có đoạn bằng chứng và trang nguồn; thiết kế bộ câu hỏi và đánh giá theo [BENCHMARK.md](BENCHMARK.md).

Đánh giá theo năm khía cạnh: **độ chính xác truy xuất**, **tính mạch lạc của chunk**, **tính hữu dụng của metadata**, **chất lượng grounding** và **tác động của chiến lược dữ liệu**. Giải thích lựa chọn bằng kết quả, ví dụ lỗi và các đánh đổi.

## 3. Chuẩn bị bài nộp

1. Chạy pipeline và lưu cấu hình, kết quả top-3, trace cùng nhãn đánh giá thủ công.
2. Điền [thông tin học viên](STUDENT.md), [nhật ký thí nghiệm](artifacts/experiment_log.csv) và bản sao [mẫu báo cáo cá nhân](report/individual/MSSV_HoTen.md).
3. Kiểm tra mã nguồn, cách chạy, manifest dữ liệu và checklist trong [SUBMISSION.md](SUBMISSION.md).

## Tài liệu hướng dẫn

| Bạn cần tìm | Tài liệu |
|---|---|
| Setup baseline PostgreSQL + pgvector | [starter/README.md](starter/README.md) |
| Các bước làm và cách tính chỉ số | [GUIDE.md](GUIDE.md) |
| Tiến độ trong 4 giờ | [CHECKPOINTS.md](CHECKPOINTS.md) |
| Năm tiêu chí benchmark và bộ câu hỏi | [BENCHMARK.md](BENCHMARK.md) |
| Tiêu chí chấm điểm | [RUBRIC.md](RUBRIC.md) |
| Quy tắc sử dụng dữ liệu, AI và API | [RULES.md](RULES.md) |
| Cách nộp bài | [SUBMISSION.md](SUBMISSION.md) |

[data/format-example.json](data/format-example.json) chỉ minh họa định dạng, không phải corpus đủ điều kiện. Phần xử lý và benchmark do bạn tự hiện thực; không bắt buộc xây chatbot.

## Bài làm - Trần Hoàng Phát (2A202603536)

### Corpus và cấu hình

- Ba báo cáo World Bank 2024 cho Viet Nam, Indonesia và Thailand; 151 trang nội dung.
- Embedding local: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 chiều).
- PostgreSQL + pgvector, cosine exact search, hybrid reranking top-20.
- A: fixed 800/overlap 100; B: sentence-aware 650; C: B + country filter.

### Cách chạy

```bash
cd starter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python scripts/download_corpus.py
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py
```

Chạy dev hoặc test vào thư mục riêng:

```bash
BENCHMARK_SPLIT=dev RESULTS_DIRECTORY=results/dev python -m src.run
BENCHMARK_SPLIT=test RESULTS_DIRECTORY=results/test python -m src.run
```

Để chạy offline sau khi model đã được tải một lần:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
BENCHMARK_SPLIT=dev RESULTS_DIRECTORY=results/dev python -m src.run
```

### Artifacts

- Manifest: `data/manifest.json`
- Benchmark cuối: `benchmark/questions.json`
- Experiment log: `artifacts/experiment_log.csv`
- Hybrid ablation: `artifacts/hybrid_ablation.csv`
- Dev/test metrics và top-3 trace: `starter/results/dev/`, `starter/results/test/`
- Audit thủ công: `starter/results/coherence_audit.csv`, `metadata_audit.csv`, `grounding_audit.csv`
- Báo cáo cá nhân: `report/individual/2A202603536_TranHoangPhat.md`

Kết quả cuối: B/C đạt dev Evidence Hit@3 `11/15`; trên holdout v4, B/C đạt Hit@1 `3/6` và Evidence Hit@3 `3/6`. Test v4 được chạy một lần sau khi khóa cấu hình hybrid trên dev.
