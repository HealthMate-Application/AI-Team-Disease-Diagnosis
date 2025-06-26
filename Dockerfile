FROM python:3.11 

WORKDIR /app

RUN pip install pipenv 

COPY Pipfile* ./

# Force Pipenv to use the system Python
RUN pipenv install --deploy --ignore-pipfile --python $(which python)

COPY . .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["pipenv", "run", "uvicorn","disease_dignoses.api:app", "--host", "0.0.0.0","--port","8000"]