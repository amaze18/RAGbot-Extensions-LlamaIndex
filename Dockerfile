# start by pulling the python image
FROM python:3.9-bookworm
# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app
RUN pip install wheel numpy
# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

# copy every content from the local file to the image
COPY . /app

HEALTHCHECK CMD curl --fail https://bits-pilani.up.railway.app/

ENTRYPOINT ["streamlit", "run", "main.py"," --server.address=https://bits-pilani.up.railway.app/"]
