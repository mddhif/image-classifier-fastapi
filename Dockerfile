FROM python:3-alpine AS builder
 
WORKDIR /app

RUN apk add --no-cache gcc g++ musl-dev libffi-dev 
RUN python3 -m venv venv
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
 
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --only-binary=:all: -r requirements.txt
 
# Stage 2
FROM python:3-alpine AS runner
 
WORKDIR /app
 
COPY --from=builder /app/venv venv
COPY main.py main.py
 
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
 
EXPOSE 8000
 
CMD [ "uvicorn", "--host", "0.0.0.0", "main:app" ]
