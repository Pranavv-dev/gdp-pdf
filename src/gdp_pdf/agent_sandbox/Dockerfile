# Same image as the Harbor task environment, without the verifier's judge
# client (here the judge runs on the host) and without the PDF (copied in
# per sample).
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
