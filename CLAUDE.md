# Claude instructions

Read `AGENTS.md` first — it has the project context, pipeline status, data sources, and rules.

## Claude-specific behavior

- Use DuckDB for parquet/CSV inspection.
- After completing work, update `AGENTS.md` with what changed. The next agent (Claude or Codex) depends on it being current.

## EC2 access

The scraper runs on EC2. To get fresh data:

```bash
ssh-keygen -t rsa -f /tmp/ec2_temp_key -N "" -q
aws ec2-instance-connect send-ssh-public-key \
  --instance-id i-09cd22a3840d802ce --instance-os-user ec2-user \
  --ssh-public-key file:///tmp/ec2_temp_key.pub --region us-east-2
ssh -i /tmp/ec2_temp_key -o StrictHostKeyChecking=no ec2-user@18.216.89.129 "<command>"
```

Then: `aws s3 sync s3://swe-labor-research/scraped/ data/scraped/`
