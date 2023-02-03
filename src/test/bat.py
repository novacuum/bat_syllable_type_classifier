from datetime import datetime
import models.bat as bat

batService = bat.BatService()
print(batService.to_bubble_days(1, datetime(2015, 6, 8, 20, 00, 00)))
