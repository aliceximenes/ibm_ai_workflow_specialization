FROM frolvlad/alpine-miniconda3

#RUN apk add musl-dev linux-headers g++
RUN pip install pandas==1.0.1 scikit-learn==0.23.2 Flask==1.1.1 joblib==0.14.1 cloudpickle==1.3.0 pickleshare==0.7.5 unittest2==1.1.0

RUN conda install -c conda-forge fbprophet==0.6

RUN mkdir app
WORKDIR app

COPY api/utils ./utils
COPY data ../data
RUN ls utils
COPY api/api.py .
COPY api/model.py .

EXPOSE 8050

RUN mkdir logs
RUN mkdir models

ENTRYPOINT ["python", "api.py"]