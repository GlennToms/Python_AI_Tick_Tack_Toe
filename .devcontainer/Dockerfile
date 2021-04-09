FROM ubuntu

RUN apt update -y
RUN apt upgrade -y
RUN apt install python python3-pip -y
RUN apt update -y
RUN apt upgrade -y

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade tensorflow

# EXPOSE 8000
ENTRYPOINT ["/usr/bash"]
