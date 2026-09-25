# Email dataset

Place a genuine email dataset here and train with:

```powershell
python train.py --data data/raw/emails.csv
```

CSV must contain `label` plus `body` (or `text`, `message`, `content`). `subject` is optional. Labels may be `spam`/`ham`, `spam`/`legitimate`, or `1`/`0`. The SMS `SMSSpamCollection.csv` is intentionally not copied here: it is useful for testing the mechanics, but it is not an email dataset and must not be reported as email performance.
