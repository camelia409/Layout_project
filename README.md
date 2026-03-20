# Tamil Nadu Floor Plan Generator

## Setup (run once)
Double-click setup.bat

## Activate environment (every session)
Double-click activate.bat  OR  run: venv\Scripts\activate.bat

## Run order
1. python inspect_seeds.py
2. python db\build_db.py
3. python db\validate_db.py
4. python generate_training_data.py
5. Upload training_data\floor_plan_samples.parquet to Google Colab
6. Download trained models into models\

## Project structure
seeds\           11 CSV seed files from Fabricate AI
db\              SQLite database scripts and floorplan.db
training_data\   Generated parquet training data
models\          Trained ML models (.h5, .pkl)
engine\          Floor plan generation logic
renderer\        2D drawing engine
tests\           Test scripts
