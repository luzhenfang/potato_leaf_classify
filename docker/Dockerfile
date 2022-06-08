#FROM bitnami/pytorch:1.10.0
# FROM continuumio/anaconda3
FROM python:3.7
ENV PATH /usr/local/bin:$PATH

#COPY serv.py .
#COPY net.py .
#COPY classify_label.py .

RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN python3 -m pip install --upgrade pip

RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# flask  跨域
RUN pip3 install flask
RUN pip3 install flask-cors

#  高性能优化
RUN pip3 install gunicorn
RUN pip3 install gevent

EXPOSE 5000

WORKDIR /app

COPY . .

#CMD ["python3","/data/serv.py"]
CMD ["gunicorn","-w","5", "-b", "0.0.0.0:5000","--certfile=/data/server.crt" ,"--keyfile=/data/server.key", "--access-logfile","/data/logs.log","serv:app"]