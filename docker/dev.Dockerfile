FROM python:3.10.7 as base

RUN pip install --upgrade pip

WORKDIR /app

# Copying it earlier & redundantly for better caching + faster builds
COPY Makefile poetry.lock pyproject.toml /app/

# Install poetry & cofig deps directly installed at system level
RUN make install-poetry
RUN poetry config virtualenvs.create false

RUN poetry install --all-extras --no-root

FROM base as dev

COPY . /app/

RUN make install-py-dev-req

RUN make dvc-pull

CMD ["poetry-shell"]

ENTRYPOINT [ "make" ]
