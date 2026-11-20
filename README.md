# AGRO-SENTINEL starter demonstrator

## What this version does
- Accepts manual environmental readings or generates clearly labelled sample readings.
- Applies transparent, illustrative heat-exposure screening rules.
- Explains the reason for the category and provides conservative safety prompts.
- Saves timestamped records locally to CSV.
- Displays an illustrative GIS station marker.
- Provides a downloadable CSV log.

## Important limitations
- No physical sensors or ESP32 are connected in this version.
- Sample readings are not real measurements.
- The screening thresholds are illustrative and require review against applicable occupational safety guidance.
- The break planner is not a certified work-rest schedule.
- This is not a medical diagnostic tool.
- It does not calculate validated WBGT and does not claim validated prediction performance.

## Run on Windows
1. Extract this ZIP.
2. Open the extracted folder in VS Code.
3. Open Terminal > New Terminal.
4. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

5. Install dependencies:

   ```powershell
   py -m pip install -r requirements.txt
   ```

6. Start the dashboard:

   ```powershell
   streamlit run app.py
   ```

7. Streamlit will provide a local browser address, usually http://localhost:8501.

## Before public deployment
- Replace illustrative coordinates with verified station coordinates.
- Keep demo/simulated data visibly labelled.
- Add the actual hardware data interface only after sensor readings are tested against reference equipment.
- Review the screening rules with a competent occupational-health/safety source.
- Do not claim clinical diagnosis, validated WBGT, or validated predictive performance.
