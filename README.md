# Hybrid RAG

2024 ICT 캡스톤디자인

## Experiments
- Benchmark dataset: **ARAGOG** [[Paper](https://arxiv.org/abs/2404.01037)] [[Code](https://github.com/predlico/ARAGOG)]
  - 107 QA pairs from 16 AI-arxiv papers
- Evaluation metric: **RAGAS** [[Paper](https://arxiv.org/abs/2309.15217)] [[Code](https://github.com/explodinggradients/ragas)]
  - Generation
    - Faithfulness: 
    - Answer relevancy: 
    - Answer correctness: 
    - Answer similarity: 
  - Retrieval
    - Context precision: 
    - Context recall: 

## Results
| RAGAS                     | Faithfulness | Answer relevancy | Answer correctness | Answer similarity | Context precision | Context recall |
|---------------------------|--------------|------------------|--------------------|-------------------|-------------------|----------------|
| Vector RAG                | 0.8935       | 0.9013           | 0.6863             | 0.9242            | 0.9021            | 0.8603         |
| Graph RAG                 |              |                  |                    |                   |                   |                |
| Hybrid RAG (w/ concat)    |              |                  |                    |                   |                   |                |
| Hybrid RAG (w/ summarize) |              |                  |                    |                   |                   |                |

## Quickstart
### 1. Clone this repo
```bash
git clone https://github.com/winun1127/Hybrid-RAG.git
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Fill in secretes
```bash
cp .env.example .env
```
Now go to your .env file and fill in the values.

### 4. Preprocessing
```bash
python preprocess.py --dataset ARAGOG
```

### 5. Run the main loop
```bash
python main.py --name hybrid --dataset ARAGOG
```