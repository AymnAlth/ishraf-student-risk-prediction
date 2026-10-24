# Contributing to Ishraf Student Attendance Risk AI System

Thank you for your interest in contributing to the Ishraf Early Warning AI System!

## Development Workflow

1. **Clone & Set Up Environment**:
   ```bash
   git clone <repository_url>
   cd ishraf_ml_coursework
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run Tests**:
   Before committing, verify all unit and integration tests pass:
   ```bash
   pytest tests/ -v
   ```

3. **Zero-Leakage Contract**:
   Any changes to features or preprocessing **MUST** adhere to `ml/student-risk/DATASET_CONTRACT.md`:
   - Never expose `student_key`, `snapshot_date`, or future attendance labels to the model.
   - Always evaluate models using chronological, student-grouped partitions.

4. **Code Style**:
   - Follow PEP 8 guidelines.
   - Include type annotations on all function signatures.
   - Add unit tests for any new feature engineering transformations.
