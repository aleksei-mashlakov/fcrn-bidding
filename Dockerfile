FROM frolvlad/alpine-miniconda3
#FROM python:3.7

#RUN pip install --upgrade pip
RUN conda update conda

#COPY requirements.txt /requirements.txt

COPY . /app
COPY conda_env.yml /conda_env.yml
#COPY configuration.yml /app/configuration.yml

RUN conda env create -f /conda_env.yml && conda clean -afy
RUN echo "source activate fcrn_bidding" > ~/.bashrc
ENV PATH /opt/conda/envs/fcrn_bidding/bin:$PATH

RUN apk add --no-cache tzdata
ENV TZ=Europe/Helsinki
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
#RUN pip install -r /requirements.txt

WORKDIR /app

ENTRYPOINT ["python", "/app/src/fcrn_bidding/pipeline.py"]
