# EC2 & S3 Infrastructure

## EC2 Instance

- **Region:** us-east-2 (Ohio)
- **Instance type:** t3.small (on-demand)
- **AMI:** Amazon Linux 2023
- **IAM role:** `ec2-scraper-role` (has S3 access)
- **Storage:** 20 GB gp3

### SSH access

```bash
ssh -i ~/path/to/scraper-key.pem ec2-user@<PUBLIC_IP>
```

Get the public IP from: **EC2 Console** (us-east-2) → **Instances** → `swe-scraper`

### Key paths on EC2

```
~/research/
├── .env                  # S3_BUCKET config
├── .venv/                # Python environment
├── scraper/              # Scraper code
├── data/scraped/         # Local data (synced to S3)
└── logs/                 # Scraper logs
```

### Common commands

```bash
cd ~/research && source .env && source .venv/bin/activate

# Run a full scrape now
nohup ./scraper/run_daily.sh >> logs/cron.log 2>&1 &

# Run a quick test
python3 scraper/scrape_linkedin_swe.py --test

# Check scrape progress
tail -f logs/scrape_$(date +%Y-%m-%d).log

# Check cron job
crontab -l

# Manual S3 sync
aws s3 sync data/scraped/ $S3_BUCKET/scraped/
```

## S3 Bucket

- **Bucket:** `s3://swe-labor-research`
- **Region:** us-east-1

### Structure

```
s3://swe-labor-research/
├── scraped/
│   ├── 2026-03-05_swe_jobs.csv
│   ├── 2026-03-05_non_swe_jobs.csv
│   ├── 2026-03-05_manifest.json
│   └── _seen_job_ids.json
└── scraper_status.json
```

### Access from local machine

```bash
# List recent scrapes
aws s3 ls s3://swe-labor-research/scraped/

# Download all data locally
aws s3 sync s3://swe-labor-research/scraped/ data/scraped/

# Download a specific day
aws s3 cp s3://swe-labor-research/scraped/2026-03-05_swe_jobs.csv data/scraped/

# Check scraper status
aws s3 cp s3://swe-labor-research/scraper_status.json - | python3 -m json.tool
```

## Schedule

The scraper runs daily at 6 AM UTC via cron on the EC2 instance. After each run it syncs new files to S3. The parquet file is not synced — rebuild locally with:

```bash
python3 scraper/harmonize.py
```
