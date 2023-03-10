FROM python:3.8
COPY . /app
WORKDIR /app
EXPOSE 5000
RUN pip install -r requirements.txt
CMD gunicorn --bind=0.0.0.0 --timeout 600 app:app
#CMD python app.py
