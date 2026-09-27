# The service in a box: the Jev API from your own weights, one `docker compose up` away.
#
#   docker compose up -d                      # GPU, weights from the Hub on first start
#   docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d     # no GPU
#
# The image holds the code and the libraries and nothing else: the weights (7.5 GB) live in a
# volume mounted at /data/hf, downloaded once and kept across rebuilds, or come from a directory
# mounted at /models. The container runs `typecastlm-serve` with no arguments and takes every
# setting from TYPECASTLM_* in its environment — see .env.example and docs/DOCKER.md.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TYPECASTLM_HOST=0.0.0.0 \
    TYPECASTLM_PORT=8000

# The heavy layers first and on their own, so a change to the package rebuilds seconds, not
# gigabytes — and torch, the one wheel that is most of the image, on a layer of its own, so a
# broken download of anything else does not fetch it again.
WORKDIR /app
# torch's pin is repeated here on purpose: a layer that read it from requirements.txt would be
# rebuilt — 900 MB, twenty minutes — whenever any other line of that file changed.
RUN pip install --retries 10 --timeout 120 torch==2.10.0
COPY docker/requirements.txt docker/requirements.txt
RUN pip install --retries 10 --timeout 120 -r docker/requirements.txt

# A C compiler, for one thing: the linear-attention kernels are Triton, and Triton builds a
# launcher stub with `cc` the first time a kernel runs. Without it the service loads, answers
# /health, and fails its first question on a GPU with "Failed to find C compiler".
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Then the package itself, from this tree rather than from PyPI: the image tracks the checkout.
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src src
RUN pip install --no-deps . && rm -rf /root/.cache

# Not root: the process reads weights and writes the cache, nothing else.
RUN useradd --create-home --uid 1000 typecast \
    && mkdir -p /data/hf /models \
    && chown -R typecast:typecast /data
USER typecast
# Settings chosen in /admin live in the volume, beside the weights, so a restart keeps them.
# (Set here, below the library layers: an ENV above them would rebuild all of them.)
# USER and LOGNAME let the container run under any --user: torch asks for a user name, and a uid
# with no passwd entry has none.
ENV TYPECASTLM_CONFIG=/data/hf/typecastlm-config.json \
    USER=typecast LOGNAME=typecast
VOLUME ["/data/hf"]
EXPOSE 8000

# Healthy means loaded: /health exists only once the weights are in memory, and a cold start
# with a download is minutes, hence the long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import os, sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('TYPECASTLM_PORT', '8000'), timeout=4).status == 200 else 1)"

# An entrypoint, so that flags can follow the image name: docker run … typecastlm --backend jev
ENTRYPOINT ["typecastlm-serve"]
