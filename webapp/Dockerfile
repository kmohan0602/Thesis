FROM python:3.7-slim

WORKDIR /webapp

ADD . /webapp

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 5000

ENV NAME OpentoAll

CMD ["python","app.py"]
