# Replicate training commands

Use this with `REPLICATE_API_TOKEN` set in your shell.

## Option 1: Official FLUX trainer

Endpoint model: `ostris/flux-dev-lora-trainer`

```bash
curl -s -X POST https://api.replicate.com/v1/trainings \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "26dce37a",
    "destination": "mihai-chindris/image-generator-v2",
    "input": {
      "trigger_word": "mihai",
      "steps": 1600,
      "learning_rate": 0.00015,
      "lora_rank": 16,
      "input_images": "https://YOUR_PUBLIC_FILE_URL/replicate_bundle_v2.zip"
    }
  }'
```

## Option 2: Fast FLUX trainer

Endpoint model: `replicate/fast-flux-trainer`

```bash
curl -s -X POST https://api.replicate.com/v1/trainings \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "mihai-chindris/image-generator-v2-fast",
    "input": {
      "trigger_word": "mihai",
      "type": "subject",
      "steps": 1400,
      "input_images": "https://YOUR_PUBLIC_FILE_URL/replicate_bundle_v2.zip"
    }
  }'
```

Notes:

- Upload the zip to a public URL first (or use the Replicate web uploader).
- Keep trigger word exactly `mihai`.
- Start with one baseline run, then run sweep variants from `replicate-v2-run-plan.md`.
