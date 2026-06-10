# The uploader needs PDAL (a native lib with GDAL/curl VSI support, used to read
# point clouds over /vsigs and write COPC). PDAL is not a reliable pip wheel, so
# we build on micromamba and install it from conda-forge. All Python deps live
# in the conda base env — we deliberately do NOT create a uv venv, because a
# separate .venv would not see conda's pdal/python-pdal.
FROM mambaorg/micromamba:2.0-debian12-slim

# micromamba does not auto-activate the env in RUN steps; this activates `base`.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

# Binary deps that need conda: Python, PDAL (+ its GDAL/PROJ/curl), and uv.
RUN micromamba install -y -n base -c conda-forge \
    python=3.13 \
    pdal \
    python-pdal \
    uv \
    && micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER services/lib/ /app/lib/
COPY --chown=$MAMBA_USER:$MAMBA_USER services/uploader/ /app/uploader/

WORKDIR /app/uploader

# Install lib (with the extras the uploader needs) and the uploader itself into
# the conda base env via uv pip — NOT `uv sync`, which would create a .venv that
# can't see conda's pdal. `which python` is the conda interpreter.
RUN uv pip install --python "$(which python)" \
    --editable "/app/lib[firestore,gcs,raster,zarr]" \
    --editable /app/uploader

ENV PYTHONPATH=/app/uploader
ENV FUNCTION_TARGET=process_upload
ENV FUNCTION_SOURCE=uploader/main.py
ENV FUNCTION_SIGNATURE_TYPE=cloudevent
ENV CPL_MACHINE_IS_GCE=YES

# The base image's entrypoint activates the conda env, then runs the CMD.
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["functions-framework"]
