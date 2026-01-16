FROM ubuntu:oracular
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -yq install python3-pip python3-venv
COPY requirements.txt /model/
RUN python3 -m venv /model
RUN /model/bin/pip3 install -r /model/requirements.txt
COPY model.py /model/
ENTRYPOINT ["/model/bin/python3", "/model/model.py"]
CMD ["--input=/data/test.csv", "--output=/data/aki.csv"]
