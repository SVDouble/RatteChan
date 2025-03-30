###########
# BUILDER #
###########

FROM python:3.12-slim AS builder
WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt .

# Install build dependencies and create wheels for your Python packages
RUN set -ex \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

#########
# FINAL #
#########

FROM python:3.12-slim

LABEL maintainer="Valentin Safronov <svdouble@hotmail.com>"

# Create a non-root user and directory for the app
RUN groupadd -r app && useradd --no-log-init -r -g app app

ENV HOME=/home/app
ENV APP_HOME=$HOME
WORKDIR $APP_HOME

# Install runtime dependencies
RUN set -ex \
    && apt-get update \
    && apt-get install -y --no-install-recommends libpcre3 mime-support netcat-traditional bc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/requirements.txt .
RUN pip install --no-cache /wheels/* \
    && rm -rf /wheels/

# Copy the application source code
COPY --chown=app:app ./ "$APP_HOME"

# Expose port 80
EXPOSE 80

# Set Python path and switch to the non-root user
ENV PYTHONPATH=$APP_HOME
USER root

# Start the application
ENTRYPOINT ["python", "-m", "app"]
