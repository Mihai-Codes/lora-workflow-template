# Publishing Plan (code-only)

## GitHub (Mihai Codes org)

Recommended: publish this project as a workflow repo without personal data.

1. Use `README-public.md` as repository README.
2. Keep `.gitignore` as-is.
3. Verify no files under data/output/checkpoints are tracked.
4. Push only workflow scripts and docs.

## Hugging Face

Recommended: do not publish personal-face LoRA weights publicly.

Safer alternatives:

- Publish a Space or repo with training workflow docs only.
- Publish a template model card with no weights.

If you keep an existing personal model on HF:

- Prefer switching visibility to **private** first.
- Keep or delete based on your risk tolerance; if uncertain, keep private.

## Existing HF model decision

For `mihai-chindris/image-generator`:

- If you do not actively need public access, set it to **private** now.
- Delete only if you are sure you never need it again.

Reason: it is identity-linked and publicly downloadable; private mode gives you immediate risk reduction without irreversible loss.
