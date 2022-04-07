# FROM python:3.7-slim

# WORKDIR /webapp

# ADD . /webapp

# RUN pip install --trusted-host pypi.python.org -r requirements.txt

# EXPOSE 5000

# ENV NAME OpentoAll

# CMD ["python","app.py"]


FROM python:3.7-slim
WORKDIR /docker_container
ADD . /docker_container
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install torch torchvision torchaudio
EXPOSE 5000
# CMD ["cd","webapp"]
# CMD ["python","app.py"]
# CMD ["cd webapp/ && python app.py"]
CMD cd webapp/ ; python app.py