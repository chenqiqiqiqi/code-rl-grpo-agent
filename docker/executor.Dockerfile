# Candidate build: Linux image must be built, digest-pinned and probe-validated.
# No GPU, host directories, dataset answers or credentials enter this image.
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
ARG SANDBOX_COMMIT=469de0845cb7d665bc68044d541aa3aa28c5b30a
COPY docker/vendor/sandboxfusion.tar.gz /tmp/sandboxfusion.tar.gz
RUN echo '44801855cbbee603501da86988dc4bf6561eb046e9f89c61930be6a62eb25f80  /tmp/sandboxfusion.tar.gz' | sha256sum -c - \
    && mkdir /opt/SandboxFusion \
    && tar -xzf /tmp/sandboxfusion.tar.gz -C /opt/SandboxFusion --strip-components=1 \
       SandboxFusion-$SANDBOX_COMMIT/LICENSE \
       SandboxFusion-$SANDBOX_COMMIT/sandbox/configs/run_config.py \
       SandboxFusion-$SANDBOX_COMMIT/sandbox/configs/__init__.py \
       SandboxFusion-$SANDBOX_COMMIT/sandbox/runners/base.py \
       SandboxFusion-$SANDBOX_COMMIT/sandbox/runners/types.py \
       SandboxFusion-$SANDBOX_COMMIT/sandbox/runners/isolation.py \
       SandboxFusion-$SANDBOX_COMMIT/sandbox/utils/common.py \
       SandboxFusion-$SANDBOX_COMMIT/sandbox/utils/execution.py \
    && rm /tmp/sandboxfusion.tar.gz \
    && pip install --no-cache-dir pydantic==2.6.4 pydantic_core==2.16.3 annotated-types==0.7.0 typing_extensions==4.15.0 structlog==23.3.0 psutil==5.9.8 aiofiles==23.2.1 PyYAML==6.0.3 \
    && pip freeze > /opt/executor-requirements.lock
COPY docker/sandbox-config.yaml /opt/SandboxFusion/sandbox/configs/posttrain.yaml
COPY docker/worker.py /opt/worker.py
ENV SANDBOX_CONFIG=posttrain PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /tmp
ENTRYPOINT ["python", "-I", "/opt/worker.py"]
