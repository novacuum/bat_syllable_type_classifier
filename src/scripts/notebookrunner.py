import asyncio
import sys

if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from engine.settings import RESULTS_FOLDER
from utils.experiment import create_result_report_notebook

report_folder = RESULTS_FOLDER / 'report'

for result_file in RESULTS_FOLDER.glob('testing_*.json'):
    html_file = report_folder / f'{result_file.stem}.html'
    print(html_file.stem)
    if not html_file.exists():
        try:
            create_result_report_notebook(result_file.stem)
        except Exception as e:
            print(e)
