FROM python

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

# EXPOSE 8000
ENTRYPOINT ["python run.py"]
