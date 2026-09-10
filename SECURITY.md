# Security

This service is built for one person on one machine. It has no login, so its
safety rests on not being reachable by anyone else, and on handling the input it
does receive carefully. This file says what is in place and what to know.

## Reachability

- The container publishes its port on `127.0.0.1` only, so the page is reachable
  from this machine and not from the network. Changing the `ports` line in the
  compose file to `"8080:5000"` opens it to everyone on the network, who can then
  upload audio, read every transcript by name, change the settings (including
  the Ollama address that transcripts are sent to) and reset them.
- State-changing requests that a page on another website starts through your
  browser are refused, using the browser's `Sec-Fetch-Site` and `Origin`
  headers. The JSON routes also insist on a JSON content type.
- The page is served with a Content Security Policy: scripts run only from the
  page's own nonce-marked script, nothing may frame the page, and the page can
  only talk to its own origin. `X-Frame-Options`, `X-Content-Type-Options` and
  a referrer policy are set alongside.

## What leaves the machine

- Transcription runs entirely locally. Model weights are baked into the image,
  pinned by commit, and the container runs with Hugging Face's offline mode and
  telemetry switched off.
- Transcript analysis talks to the Ollama address in Settings, on this machine
  by default, and to Google's Gemini API only when Cloud Gemini is chosen.
- Nothing else is contacted at runtime, and the page loads no third-party
  resources.

## Secrets

- The Gemini API key is entered in Settings and saved to `./data/settings.json`
  with owner-only permissions. It is never logged and never sent back to the
  page; the page only learns whether a key is set.
- `.env` is ignored by git and by the Docker build context. Only values that
  differ from the defaults are saved from the page.

## Input handling

- Uploads are limited to 500 MB and to audio and video extensions, stored under
  a sanitised name, transcribed, and deleted. Transcripts are served from the
  outputs folder only, with a real-path check against traversal.
- Everything the page shows that came from the server, a setting or a model is
  written as text, never as markup. The analysis renderer escapes markdown
  before it emits any tag.
- The container runs as an unprivileged user, so a flaw in a media decoder does
  not have root inside it.
- Rate limits apply to uploads and analyses.

## Known limits

- The prompt-injection filter removes a few phrases and adds a safety prefix. A
  hostile recording can still steer a summary; that is a property of language
  models, not something this code can close.
- Decoding untrusted audio is a risk by nature. Keep the image rebuilt so the
  system packages stay current.
- The app runs on Werkzeug's development server, which suits a personal tool on
  localhost and is not meant for the open internet.

## Reporting

If you find a problem, open an issue on the repository.
